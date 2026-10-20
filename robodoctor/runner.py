"""Persistent GPT policy with grounding, absolute IK and dual-arm EEF tools."""
from .demonstration import add_arguments, validate, prepare as prepare_demonstration
import argparse
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from hybrid_rollout.robodojo.io import InputError, write_json
from hybrid_rollout.robodojo.prompt_context import CONTEXT_VERSION
from hybrid_rollout.robodojo.settings import MODEL as DEFAULT_MODEL, EFFORT, PROVIDER, DEFAULT_CODEX_IMAGE_MAX_EDGE
# Backward-compatible module constant for callers and rollout fixtures.
MODEL = DEFAULT_MODEL
from hybrid_rollout.robodojo.method import evaluation_method
from hybrid_rollout.robodojo.skill.transport import StdioAppServer
from .workspace import prepare_workspace, hybrid_skill_root
from .harness_snapshot import validate_harness, prompt_hashes
from hybrid_rollout.robodojo.skill.image_preview import image_max_edge, prepare_image
from .recovery import closed_failed_turn, oversized_request, recovery_message
from .rsi_memory import RSIMemory
from .memory import EpisodeMemory
from .task_experience import TaskExperience, validate_task_experience
from .schema import tool_specs

SKILL_ROOT = Path(__file__).parent/'prompts'
CONTROLLER_VERSION = 'grounded_direct_v2'
NATIVE_WORK_ITEMS = frozenset(('commandExecution', 'fileChange', 'imageView',
    'webSearch', 'mcpToolCall', 'collabAgentToolCall'))
# The provider rejects a request containing more than 50 images. Keep margin
# for native imageView calls and multiple tool results in one turn.
THREAD_IMAGE_LIMIT = 50
THREAD_IMAGE_ROTATE_AT = 36



def rejected_input(error, rollout, arguments, count):
    """Feedback only for InputError's pre-execution rejection contract.

    Never change arguments, clip a target, retry a handler or reset simulation.
    Other exceptions (including uncertain RPC outcomes) remain fatal.
    """
    packet = dict(error=str(error), error_type='recoverable_tool_input',
        no_execution=True, retryable=True, controller_version=getattr(rollout, 'control_version', CONTROLLER_VERSION),
        rejected_calls=count, step_id=rollout.tick, next_call=rollout.next_call(),
        correction='Correct the arguments and retry next_call in this same episode. '
                   'Do not reset or replay already executed actions.')
    if getattr(rollout, 'observation_path', None):
        packet['observation_path'] = str(rollout.observation_path)
    request = getattr(rollout, 'request', None)
    if isinstance(request, dict) and rollout.phase in ('act', 'execute'):
        packet['request_id'] = request.get('request_id')
        packet['current_eef'] = request.get('current_eef')
    return packet


def toml_value(value):
    """Codex -c values are TOML; JSON objects/quoted dotted keys are not equivalent."""
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml_value(v) for k, v in value.items()) + '}'
    return json.dumps(value)


def agent_config(audit, agent, *, effort=EFFORT, model=DEFAULT_MODEL, provider=PROVIDER):
    # Keep Codex's normal tool/skill configuration. These two capabilities are
    # essential, even if an inherited profile had disabled them previously.
    return dict(model=model, model_provider=provider, model_reasoning_effort=effort,
        sqlite_home=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_db'),
        log_dir=str(Path(os.environ.get('ROLLOUT_CODEX_STATE_DIR', os.environ.get('CODEX_HOME', str(audit))))/'runtime_logs'),
        default_permissions='rollout_agent', **{
            'features.shell_tool': True, 'features.view_image': True,
            'shell_environment_policy.inherit': 'all',
            'permissions.rollout_agent.extends': ':workspace',
            # The parent contains host-owned RPC logs and experiment artifacts.
            # Only the agent directory is a writable runtime workspace root.
            'permissions.rollout_agent.filesystem': {
                str(audit.parent): 'read', str(agent): 'write'},
        })


def content_items(packet, *, images, max_image_edge=DEFAULT_CODEX_IMAGE_MAX_EDGE):
    # This boundary is downstream of observation capture and policy inference.
    # Copy metadata only: never change packet, source PNG/NPZ or policy arrays.
    visible_packet = packet
    attachments = []
    if images:
        # Keep each service reply bounded to the latest three camera views.
        # Older observations remain on disk for recovery, but are never reattached.
        visible_packet = dict(packet, images=[])
        for item in packet.get('images', [])[-3:]:
            descriptor, payload = prepare_image(item, max_image_edge)
            visible_packet['images'].append(descriptor)
            data = base64.b64encode(payload).decode('ascii')
            attachments.append(dict(type='inputImage', imageUrl='data:image/png;base64,' + data))
    return [dict(type='inputText', text=json.dumps(visible_packet, separators=(',', ':'))), *attachments]


class CodexPolicy:
    def __init__(self, workspace, codex, *, timeout=900, transport_factory=StdioAppServer,
                 method='gpt_only', smoke_approach=False, tower_test=False, effort=EFFORT,
                 control_mode='0-shot', task_details=None, demo_video=None, demo_frames=24,
                 model=DEFAULT_MODEL, experience_enabled=False, task_experience=None, iteration=0,
                 harness_dir=None):
        if smoke_approach and tower_test:
            raise ValueError('Smoke approach and full tower test are mutually exclusive')
        if effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('effort must be low, medium, high, xhigh or max')
        validate(control_mode, task_details, demo_video, demo_frames, method)
        validate_task_experience(task_experience)
        if task_experience and method != 'gpt_only':
            raise ValueError('--task-experience requires direct control (gpt_only)')
        self.effort = effort
        self.model = model
        self.provider = os.environ.get("RSI_CODEX_PROVIDER", PROVIDER)
        self.method = evaluation_method(method)
        harness_files = validate_harness(harness_dir, self.method) if harness_dir is not None else None
        self.experience_enabled = bool(experience_enabled)
        if self.method != 'gpt_only' and (smoke_approach or tower_test):
            raise ValueError('Direct-only smoke/tower verifier cannot be used for hybrid')
        self.controller_version = ('rsi_policy_doctor_v1' if self.method == 'rsi_policy_doctor' else
            'grounded_hybrid_v1' if self.method == 'pi05_plus_gpt' else CONTROLLER_VERSION)
        skill_root = SKILL_ROOT
        self.image_max_edge = image_max_edge(os.environ.get(
            'CODEX_IMAGE_MAX_EDGE', '0'))
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        if self.method == 'pi05_plus_gpt':
            skill_root = hybrid_skill_root(self.workspace)
        elif self.method == 'rsi_policy_doctor':
            skill_root = Path(harness_dir) if harness_dir is not None else Path(__file__).parent/'rsi_prompts'
        write_json(self.workspace.parent/'controller_protocol.json', dict(
            controller_version=self.controller_version, context_version=CONTEXT_VERSION,
            recoverable_input_error_limit=None, eef_target_delta_limited=False, motion_planner='curobo',
            same_thread_except_oversized_request=True, simulator_reset_on_input_error=False,
            action_replay_on_input_error=False))
        self.agent_workspace = prepare_workspace(self.workspace, skill_root, method=self.method)
        if harness_files is not None:
            if prompt_hashes(self.agent_workspace) != harness_files:
                raise ValueError('Harness prompts changed while preparing the RSI workspace')
            write_json(self.workspace/'harness.json', dict(source=str(Path(harness_dir).resolve()),
                                                         files=harness_files))
            # Inject exactly the verified per-episode copy, never the direct prompt tree.
            skill_root = self.agent_workspace
        self.memory = RSIMemory(self.agent_workspace) if self.method == 'rsi_policy_doctor' else EpisodeMemory(self.agent_workspace)
        self.task_experience = (TaskExperience(self.workspace.parent, self.agent_workspace,
            source=task_experience, iteration=iteration) if self.method == 'gpt_only' else None)
        self.last_run_completed = False
        self.thread_usage = {}
        if self.method == 'gpt_only':
            self.demonstration, example_prompt = prepare_demonstration(
                self.workspace/'demonstration', control_mode, task_details, demo_video, demo_frames)
        else:
            self.demonstration, example_prompt = {'frames': []}, ''
        self.example_images_pending = True
        self.inject_latest_images_pending = False
        self.latest_image_items = []
        self.thread_image_count = 0
        self.thread_image_ids = set()
        self.thread_rotations = 0
        self.timeout = timeout
        skill = (skill_root/'SKILL.md').read_text()
        context = (skill_root/'context/teacher_context.md').read_text()
        self.prompt = (skill + '\n\n' + context + '\n\nAgent working directory: '
                       + str(self.agent_workspace)
                       + '\nResolve context/ and workspace.json relative to that directory.\n'
                       + '\nIMAGE CONTEXT BUDGET: Each current observation already includes the latest three RGB camera images. '
                       'Use those images directly. Do not repeatedly call imageView on old observations or '
                       'historical frames; only inspect a current workspace image when essential.\n')
        self.prompt += example_prompt
        if self.task_experience:
            self.prompt += self.task_experience.instructions()
        if smoke_approach:
            self.prompt += ('\nSMOKE TEST SCOPE: The loaded task is build_tower, but this run only tests '
                'visual grounding and reaching a clear pre-operation pose. You must inspect the actual '
                'head image, choose a visible tower block or board yourself, call grounding and '
                'robodojo_act with both absolute environment_origin targets and stop_on_reach=true. Select your own pixel and poses; no scripted '
                'target is supplied. Preserve the current gripper orientation for this initial '
                'approach, leave generous clearance above/beside the object, and keep grippers open. '
                'The host ends the episode after both measured poses arrive. Do not '
                'grasp, build the tower, or return the arm. Report approach arrival separately '
                'from native task success. This smoke scope takes precedence over task completion.\n')
        if tower_test:
            self.prompt += ('\nFULL TOWER TASK TEST: Execute the original build_tower instruction through '
                'native termination. This is NOT an approach-only smoke test. Independently select '
                'objects and placements from observations, use grounding and unified robodojo_act for clear '
                'pre-operation poses, then robodojo_act to orient, grasp, lift, place and release. '
                'Inspect each manipulation and re-ground new targets. Use grounded absolute '
                'waypoints to lift before transferring above a placement target, then fine EEF '
                'control to lower and release; choose offsets from the scene without a fixed margin. '
                'Build the required supported tower layers and reserve controls for releasing '
                'objects, retracting and returning both arms as task_context requires. Arrival, '
                'one grasp, one layer or your own visual judgment does not end this test. Continue '
                'until rollout_finished; only native task success passes. Native timeout or an '
                'operator budget is failure/incomplete, never a successful tower. No scripted '
                'pixel, pose or action sequence is supplied.\n')
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace/'SKILL.md').write_text(skill)
        (self.workspace/'PROMPT.md').write_text(self.prompt)
        specs = tool_specs(self.method)
        if not self.experience_enabled:
            specs = [item for item in specs if item["name"] != "update_experience"]
        write_json(self.workspace/'tools.json', specs)
        version = subprocess.run([codex, '--version'], capture_output=True, text=True, check=True).stdout.strip()
        config = agent_config(self.workspace, self.agent_workspace, effort=self.effort, model=self.model, provider=self.provider)
        argv = [codex, 'app-server', '--stdio', '--strict-config']
        for key, value in config.items():
            argv += ['-c', key + '=' + toml_value(value)]
        write_json(self.workspace/'launch.json', dict(argv=argv, config=config))
        self.transport = transport_factory(argv, self.workspace)
        self.thread_params = dict(
            cwd=str(self.agent_workspace), model=self.model, modelProvider=self.provider,
            config=dict(model_reasoning_effort=self.effort), developerInstructions=self.prompt,
            dynamicTools=specs, ephemeral=False,
            allowProviderModelFallback=False, approvalPolicy='never',
            approvalsReviewer='auto_review', permissions='rollout_agent',
            runtimeWorkspaceRoots=[str(self.agent_workspace)])
        try:
            self.transport.request('initialize', dict(
                clientInfo=dict(name='robodoctor-grounded-direct', version='1'),
                capabilities=dict(experimentalApi=True)), timeout)
            self.transport.notify('initialized', {})
            self._start_thread(timeout, codex=codex, version=version, control_mode=control_mode,
                               smoke_approach=smoke_approach, tower_test=tower_test, specs=specs)
        except BaseException:
            self.close()
            raise

    def _start_thread(self, timeout, *, codex=None, version=None, control_mode=None,
                      smoke_approach=False, tower_test=False, specs=None, reason=None):
        """Start a clean model context while preserving the physical episode."""
        response = self.transport.request('thread/start', self.thread_params, timeout)
        if response.get('model') != self.model or response.get('reasoningEffort') != self.effort:
            raise RuntimeError(f'Codex model/effort differs from the required {self.model}/{self.effort}: '
                               f'{response.get("model")}/{response.get("reasoningEffort")}')
        self.thread_id = response['thread']['id']
        self.thread_image_count = 0
        self.thread_image_ids.clear()
        worker_path = self.workspace/'worker.json'
        if worker_path.is_file():
            worker = json.loads(worker_path.read_text())
            worker.setdefault('thread_ids', []).append(self.thread_id)
            worker['thread_id'] = self.thread_id
            worker['thread_image_count'] = 0
            worker['thread_image_rotate_at'] = THREAD_IMAGE_ROTATE_AT
            worker['thread_rotation_reason'] = reason
            worker['thread_rotations'] = self.thread_rotations
            write_json(worker_path, worker)
        else:
            write_json(worker_path, dict(model=self.model, model_provider=self.provider,
                controller_version=self.controller_version, evaluation_method=self.method,
                control_mode=control_mode, demonstration=self.demonstration,
                task_experience_input=str(self.task_experience.prior) if self.task_experience and self.task_experience.prior else None,
                task_experience_output=str(self.task_experience.output) if self.task_experience else None,
                reasoning_effort=self.effort, context_version=CONTEXT_VERSION,
                codex_version=version, thread_id=self.thread_id, thread_ids=[self.thread_id],
                pid=self.transport.process.pid, codex_executable=str(Path(codex).resolve()) if codex else None,
                smoke_approach=smoke_approach, tower_test=tower_test, prompt_sha256=self.prompt_sha256,
                no_rollback=True, tools=[item['name'] for item in (specs or [])],
                service_tools_are_additive=True, agent_workspace=str(self.agent_workspace),
                native_tools='Codex defaults; shell and image viewing explicitly enabled',
                model_error_recovery=dict(revision='error_feedback_v2',
                    backoff_seconds=[10, 20, 30, 40, 50, 60], restart_simulator=False,
                    oversized_request_new_thread=True, replay_actions=False, attempt_limit=None),
                codex_image_max_edge=self.image_max_edge, policy_images_resized=False,
                baseline_full_conversation_available=False, thread_image_count=0,
                thread_image_rotate_at=THREAD_IMAGE_ROTATE_AT, thread_rotations=0))

    def _maybe_rotate_thread(self, reason):
        if self.thread_image_count < THREAD_IMAGE_ROTATE_AT:
            return False
        self.thread_rotations += 1
        self.inject_latest_images_pending = bool(self.latest_image_items)
        self._start_thread(self.timeout, reason=reason)
        return True

    def _turn(self, text):
        memory = self.memory.capture()
        inputs = [dict(type='text', text=text + '\nCurrent TODO snapshot:\n' + json.dumps(memory, ensure_ascii=False))]
        image_inputs = 0
        if self.example_images_pending:
            for frame in self.demonstration['frames']:
                inputs.append(dict(type='text', text=f"Demonstration at {frame['seconds']:.3f} seconds"))
                inputs.append(dict(type='localImage', path=frame['path']))
                image_inputs += 1
        if self.inject_latest_images_pending:
            for item in self.latest_image_items[-3:]:
                inputs.append(dict(type='text', text=f"Latest current observation: {item.get('camera', 'camera')}"))
                inputs.append(dict(type='localImage', path=item['path']))
                image_inputs += 1
            self.inject_latest_images_pending = False
        self.thread_image_count += image_inputs
        result = self.transport.request('turn/start', dict(threadId=self.thread_id,
            model=self.model, effort=self.effort, approvalPolicy='never', approvalsReviewer='auto_review',
            permissions='rollout_agent', cwd=str(self.agent_workspace),
            runtimeWorkspaceRoots=[str(self.agent_workspace)], input=inputs), self.timeout)
        self.example_images_pending = False
        return result['turn']['id']

    def _model_continue(self, rollout, old_turn, error, consecutive, serial):
        fresh_thread = oversized_request(error)
        delay = min(10 * (consecutive + 1), 60)
        record = dict(revision='error_feedback_v2', thread_id=self.thread_id,
            previous_turn_id=old_turn, error=error, step_id=rollout.tick,
            phase=rollout.phase, next_call=rollout.next_call(), delay_seconds=delay,
            consecutive_attempt=consecutive+1, status='waiting',
            fresh_thread=fresh_thread, physical_actions_replayed=0)
        path = self.workspace/f'model_error_{serial:04d}.json'
        write_json(path, record)
        print(json.dumps(dict(event='codex_model_error_wait', **record)), flush=True)
        # Preserve the simulator; operator signals and launcher budgets still apply.
        for _ in range(delay):
            time.sleep(1)
        if fresh_thread:
            # Retrying an oversized conversation can resend the same rejected
            # payload. Start with the same instructions/tools/workspace and only
            # the latest observation images plus compressed state.
            self.inject_latest_images_pending = bool(self.latest_image_items)
            self._start_thread(self.timeout, reason='provider_oversized_request')
        message = recovery_message(rollout, error, path, fresh_thread=fresh_thread)
        record.update(status='turn_start_requested', recovery_thread_id=self.thread_id,
                      model_feedback=message)
        write_json(path, record)  # Never replay a handler if transport ACK is uncertain.
        new_turn = self._turn(message)
        record.update(status='continue_sent', new_turn_id=new_turn)
        write_json(path, record)
        print(json.dumps(dict(event='codex_model_error_feedback_sent', thread_id=self.thread_id,
            old_turn_id=old_turn, turn_id=new_turn, step_id=rollout.tick)), flush=True)
        return new_turn

    def run(self, rollout, *, summary_only=False):
        self.last_run_completed = False
        self.memory.episode(rollout)
        if summary_only:
            if rollout.phase != 'done' or self.task_experience is None:
                raise ValueError('Task experience summary requires a finished direct episode')
            prompt = ('The physical episode is finished. Do not call any simulator service or execute '
                      'any more robot actions. Use only recorded evidence and your workspace files to '
                      'write the final task experience for the next iteration. Native outcome and a '
                      f'fallback evidence record are in {self.task_experience.output}. '
                      + self.task_experience.instructions())
        else:
            prompt = ('Act as the autonomous policy agent for this single simulation rollout. '
                      'Use your normal file, image, shell/code and planning tools as useful, alongside '
                      'the rollout service tools. Read workspace.json for paths and interpreter; '
                      'keep working notes and analysis in your agent workspace. Use English for every '
                      'public explanation, assessment, progress update, note, and final report. '
                      'The user authorizes sending this episode\'s three RGB images, proprio, robot-only FK, '
                      'task text, depth-grounded surface coordinates and same-episode history for online decisions. '
                      'First call: ' + json.dumps(rollout.next_call()))
        turn_id = self._turn(prompt)
        audit_prefix = 'experience_' if summary_only else ''
        call_index = 0
        errors = 0
        continuations = 0
        model_error = None
        model_consecutive = model_serial = 0
        completed_turns = 0
        thread_usage = self.thread_usage
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError('Codex made no completed service or native tool call within the configured timeout')
                event = self.transport.next_message(remaining)
            except TimeoutError:
                if not model_error:
                    raise
                # A missing terminal notification does not authorize a second active turn.
                response = self.transport.request('thread/read', dict(
                    threadId=self.thread_id, includeTurns=True), min(30, self.timeout))
                turn = closed_failed_turn(response, self.thread_id, turn_id)
                if turn is None:
                    # Provider error is pending, but the turn is still active.
                    # Wait rather than start a competing turn or end the episode.
                    deadline = time.monotonic() + 30
                    continue
                event = dict(method='turn/completed', params=dict(threadId=self.thread_id, turn=turn))
            method, params = event.get('method'), event.get('params', {})
            if method == 'thread/tokenUsage/updated' and params.get('threadId') == self.thread_id:
                usage = params.get('tokenUsage', {})
                thread_usage[self.thread_id] = usage
                total = {}
                for value in thread_usage.values():
                    for key, count in value.get('total', {}).items():
                        if isinstance(count, (int, float)):
                            total[key] = total.get(key, 0) + count
                write_json(self.workspace.parent/'token_usage.json', dict(
                    model=self.model, effort=self.effort, provider=self.provider, usage=dict(usage, total=total),
                    thread_usage=thread_usage,
                    total_tokens_include_cached_input=True, configured_limit=None))
                print(json.dumps(dict(event='token_usage', total=total)), flush=True)
            elif method == 'item/tool/call':
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    raise RuntimeError('Tool call belongs to another thread or turn')
                name, arguments = params['tool'], params['arguments']
                print(json.dumps(dict(event='tool_start', tool=name, call=call_index,
                                      phase=rollout.phase, step_id=rollout.tick)), flush=True)
                audit = dict(call_id=params['callId'], tool=name, arguments=arguments)
                write_json(self.workspace/f'{audit_prefix}call_{call_index:04d}_request.json', audit)
                handlers = dict(robodojo_start=rollout.start, robodojo_act=rollout.act,
                                grounding=rollout.grounding, get_depth=rollout.get_depth)
                if hasattr(rollout, 'joint_act'):
                    handlers['robodojo_joint_act'] = rollout.joint_act
                if self.method == 'pi05_plus_gpt':
                    handlers.update(update_plan=rollout.update_plan, pi05_infer=rollout.infer,
                                    review_pi05=rollout.review_pi05, pi05_execute=rollout.execute)
                elif self.method == 'rsi_policy_doctor':
                    handlers.update(policy_rollout=rollout.rollout, policy_continue=rollout.continue_policy)
                    if self.experience_enabled:
                        handlers['update_experience'] = rollout.update_experience
                self.memory.capture(step=rollout.tick)
                step_before = rollout.tick
                try:
                    if summary_only:
                        raise InputError('Task experience generation cannot call simulator services')
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as error:
                            raise InputError(f'Tool arguments must be a JSON object: {error}') from error
                    if name not in handlers:
                        raise InputError('Unknown host service; native Codex tools are executed by app-server')
                    if not isinstance(arguments, dict):
                        raise InputError('Tool arguments must be a JSON object')
                    try:
                        inspect.signature(handlers[name]).bind(**arguments)
                    except TypeError as error:
                        raise InputError(str(error)) from error
                    packet = handlers[name](**arguments)
                except InputError as error:
                    errors += 1
                    packet = rejected_input(error, rollout, arguments, errors)
                    success = False
                else:
                    success = True
                    # Successful service work resets provider backoff only.
                    model_consecutive = 0
                    model_error = None
                self.memory.episode(rollout)
                packet = dict(packet, todo_memory=self.memory.capture(step=rollout.tick))
                if success:
                    self.memory.step_note(name, arguments, step_before, rollout.tick, packet,
                                          self.workspace/f'{audit_prefix}call_{call_index:04d}_result.json')
                # Persist every rejection, including the fifth and later, before
                # replying. A failed transport reply must not repeat a handler.
                write_json(self.workspace/f'{audit_prefix}call_{call_index:04d}_result.json', packet)
                reply_items = content_items(packet, images=success and name not in ('pi05_infer', 'policy_rollout'),
                                             max_image_edge=self.image_max_edge)
                if success and name not in ('pi05_infer', 'policy_rollout'):
                    self.latest_image_items = [dict(item) for item in packet.get('images', [])][-3:]
                    self.thread_image_count += sum(item.get('type') == 'inputImage' for item in reply_items)
                self.transport.reply(event['id'], dict(success=success, contentItems=reply_items))
                call_index += 1
                print(json.dumps(dict(event='tool_done', tool=name, step_id=rollout.tick,
                    phase=rollout.phase, counters=rollout.counters)), flush=True)
                deadline = time.monotonic() + self.timeout
            elif method in ('item/started', 'item/completed'):
                if params.get('threadId') != self.thread_id or params.get('turnId') != turn_id:
                    continue
                item = params.get('item', {})
                kind = item.get('type')
                if kind == 'imageView' and method == 'item/started' and item.get('id') not in self.thread_image_ids:
                    self.thread_image_ids.add(item.get('id'))
                    self.thread_image_count += 1
                if kind in NATIVE_WORK_ITEMS or (kind == 'agentMessage' and method == 'item/completed'):
                    # Public outputs only: never copy private reasoning items.
                    with (self.workspace/'agent_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(method=method, **params), ensure_ascii=False)+'\n')
                    event_record = dict(event='agent_activity', phase=method.split('/')[-1],
                        item_type=kind, item_id=item.get('id'), step_id=rollout.tick,
                        status=item.get('status'))
                    if kind == 'agentMessage':
                        event_record['text'] = item.get('text', '')
                    elif kind == 'commandExecution':
                        event_record['command'] = item.get('command')
                        event_record['exit_code'] = item.get('exitCode')
                    print(json.dumps(event_record, ensure_ascii=False), flush=True)
                    if kind in ('commandExecution', 'fileChange') and method == 'item/completed':
                        self.memory.capture(step=rollout.tick)
                    if kind in NATIVE_WORK_ITEMS:
                        # Real analysis is progress, not a missing rollout decision.
                        deadline = time.monotonic() + self.timeout
            elif method == 'turn/completed' and params.get('threadId') == self.thread_id:
                turn = params.get('turn', {})
                if turn.get('id') != turn_id:
                    continue
                write_json(self.workspace/f'{audit_prefix}turn_{completed_turns:02d}_completed.json', turn)
                completed_turns += 1
                failure = turn.get('error') or model_error
                if turn.get('status') == 'failed':
                    if rollout.phase == 'done':
                        return  # Native completion needs no further paid model response.
                    turn_id = self._model_continue(rollout, turn_id, failure,
                        model_consecutive, model_serial)
                    model_serial += 1
                    model_consecutive += 1
                    model_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get('status') != 'completed':
                    raise RuntimeError(f'Codex turn ended: {turn.get("status")}')
                model_error = None
                if rollout.phase == 'done':
                    self.last_run_completed = True
                    return
                if continuations >= 2:
                    raise RuntimeError('Codex ended repeatedly before completing the episode')
                # Context-only continuation, never a human action judgement.
                if self._maybe_rotate_thread('image_budget'):
                    continuations = 0
                else:
                    continuations += 1
                turn_id = self._turn('The same episode is unfinished. Continue the skill; next call: ' +
                                     json.dumps(rollout.next_call()))
                deadline = time.monotonic() + self.timeout
            elif method in ('error', 'turn/failed'):
                if (params.get('threadId') not in (None, self.thread_id)
                        or params.get('turnId') not in (None, turn_id)):
                    continue
                model_error = params.get('error', params)
                if rollout.phase == 'done':
                    return  # Physical task completion needs no further model response.
                if method == 'error' and params.get('willRetry') is True:
                    # Let app-server finish its own retry before supplying feedback.
                    deadline = time.monotonic() + self.timeout
                    print(json.dumps(dict(event='codex_transport_retry',
                        error=model_error, step_id=rollout.tick)), flush=True)
                    continue
                # Wait for terminal confirmation; never overlap active turns.
                deadline = min(deadline, time.monotonic() + 30)
                print(json.dumps(dict(event='codex_model_error_wait_terminal',
                    error=model_error, step_id=rollout.tick)), flush=True)
                continue
            elif 'id' in event and 'method' in event:
                raise RuntimeError(f'Unexpected Codex capability request: {method}')

    def finalize_task_experience(self, rollout):
        if self.task_experience is None:
            return
        # Persist before asking the model, so interruption cannot lose episode evidence.
        self.task_experience.save(rollout)
        try:
            if rollout.phase == 'done' and self.last_run_completed:
                self.run(rollout, summary_only=True)
        except Exception:
            write_json(self.workspace/'task_experience_failure.json', dict(error=traceback.format_exc()))
        finally:
            self.task_experience.save(rollout)

    def close(self):
        self.transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--evaluation-method', default='gpt_only', choices=('gpt_only', 'pi05_plus_gpt', 'rsi_policy_doctor'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--intervention-experience', type=Path)
    parser.add_argument('--task-experience', type=Path)
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--harness-dir', type=Path)
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--student-backend', choices=('prismwam', 'pi05'), default='pi05')
    parser.add_argument('--action-chunk-length', type=int)
    parser.add_argument('--codex', required=True)
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--max-decisions', type=int, default=180, help='0 disables the decision-count limit')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--effort', choices=('low', 'medium', 'high', 'xhigh', 'max'), default=EFFORT)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke-approach', action='store_true')
    mode.add_argument('--tower-test', action='store_true')
    add_arguments(parser)
    args = parser.parse_args()
    try:
        validate(args.control_mode, args.task_details, args.demo_video, args.demo_frames, args.evaluation_method)
        validate_task_experience(args.task_experience)
        if args.harness_dir:
            validate_harness(args.harness_dir, args.evaluation_method)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.evaluation_method != 'gpt_only' and (args.checkpoint is None or args.smoke_approach or args.tower_test):
        parser.error('Hybrid requires --checkpoint and cannot use the direct-only smoke/tower verifier')
    if args.evaluation_method != 'rsi_policy_doctor' and args.intervention_experience:
        parser.error('--intervention-experience requires rsi_policy_doctor')
    if args.evaluation_method != 'gpt_only' and args.task_experience:
        parser.error('--task-experience requires direct control (gpt_only)')
    if args.evaluation_method == 'gpt_only' and (args.checkpoint or args.action_chunk_length):
        parser.error('Policy options require hybrid/RSI control')
    if args.evaluation_method == 'pi05_plus_gpt' and args.student_backend != 'pi05':
        parser.error('pi05_plus_gpt requires the pi05 backend')
    if args.tower_test and args.task != 'build_tower':
        parser.error('Full tower test requires --task build_tower')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    student = worker = rollout = None
    def terminate(signum, frame):
        raise KeyboardInterrupt('Stopping only this owned rollout and Codex process')
    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.evaluation_method != 'gpt_only':
            if args.student_backend == 'prismwam':
                from .prismwam_client import PrismWAMClient
                student = PrismWAMClient(args.student_port, args.checkpoint)
            else:
                from hybrid_rollout.robodojo.pi05_server.client import Pi05Client
                student = Pi05Client(args.student_port, args.checkpoint)
        worker = CodexPolicy(args.output/'codex_workspace', args.codex, method=args.evaluation_method,
                             smoke_approach=args.smoke_approach, tower_test=args.tower_test, effort=args.effort,
                             control_mode=args.control_mode, task_details=args.task_details,
                             demo_video=args.demo_video, demo_frames=args.demo_frames, model=args.model,
                             experience_enabled=args.evaluation_method == 'rsi_policy_doctor',
                             task_experience=args.task_experience, iteration=args.iteration,
                             harness_dir=args.harness_dir)
        from .client import GroundedTools
        if args.evaluation_method == 'rsi_policy_doctor':
            from .rsi import RSITools
            cls, extras = RSITools, dict(student=student, experience_path=args.intervention_experience,
                                        iteration=args.iteration, action_chunk_length=args.action_chunk_length)
        elif args.evaluation_method == 'pi05_plus_gpt':
            from .hybrid import HybridTools
            cls, extras = HybridTools, dict(student=student)
        else:
            cls, extras = GroundedTools, {}
        rollout = cls(args.output, args.task, **extras, sim_port=args.sim_port,
            seed=args.seed, max_decisions=args.max_decisions, prompt_sha256=worker.prompt_sha256,
            smoke_approach=args.smoke_approach, reasoning_effort=args.effort)
        worker.run(rollout)
        worker.finalize_task_experience(rollout)
    except BaseException:
        write_json(args.output/'failure.json', dict(error=traceback.format_exc(),
            controller_version=worker.controller_version if worker else CONTROLLER_VERSION,
            episode_id=rollout.episode if rollout else None, step_id=rollout.tick if rollout else 0,
            counters=rollout.counters if rollout else {}, completed=False))
        if rollout is not None and rollout.episode is not None and rollout.phase != 'done':
            try:
                rollout.finish('controller_error')
            except Exception:
                pass  # Never reconnect or retry an action with an uncertain ACK.
        raise
    finally:
        try:
            if rollout is not None:
                rollout.close()
        finally:
            try:
                if student is not None:
                    student.close()
            finally:
                if worker is not None:
                    try:
                        if worker.task_experience is not None and rollout is not None:
                            worker.task_experience.save(rollout)
                    finally:
                        worker.close()


if __name__ == '__main__':
    main()
