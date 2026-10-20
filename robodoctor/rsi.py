"""Iterative policy doctor: policy rollout, optional fine intervention, experience."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import uuid

import numpy as np

from hybrid_rollout.robodojo.io import InputError, write_json
from hybrid_rollout.robodojo.robodojo_server.client import RoboDojoTools
from hybrid_rollout.robodojo.robodojo_server.decision_log import decision_event, emit
from hybrid_rollout.robodojo.robodojo_server.proposal_diagnostics import action_diagnostics
from .client import GroundedTools
from .experience import ExperienceStore


class RSITools(GroundedTools):
    method = "rsi_policy_doctor"
    control_version = "rsi_policy_doctor_v1"
    fk_preview_horizon = 50

    def __init__(self, output, task, student, *, experience_path=None, iteration=0,
                 action_chunk_length=None, **kwargs):
        Path(output).mkdir(parents=True, exist_ok=True)
        super().__init__(output, task, **kwargs)
        self.student = student
        self.input_experience_path = Path(experience_path).expanduser().resolve() if experience_path else None
        effective_experience_path = self.input_experience_path or (Path(output).resolve() / "intervention_experience.md")
        self.experience = ExperienceStore(effective_experience_path, iteration=iteration, task=task)
        self.iteration = iteration
        self.proposal = None
        metadata_horizon = int(student.metadata.get("action_horizon", 0))
        self.action_chunk_length = int(action_chunk_length or metadata_horizon)
        if self.action_chunk_length < 1:
            raise ValueError("action_chunk_length must be positive")
        self.execution_horizon = int(student.metadata.get("execution_horizon", 16))
        if not 1 <= self.execution_horizon <= 32:
            raise ValueError("student execution_horizon must be 1..32")
        # RoboDojo accepts at most 16 physical actions per chunk_step RPC.
        self.execution_horizon = min(self.execution_horizon, 16)

    def next_call(self):
        if self.phase == "start":
            return RoboDojoTools.next_call(self)
        if self.phase == "act":
            if self.proposal is None:
                return dict(tool="policy_rollout", request_id=getattr(self, "request", {}).get("request_id"),
                            observation_path=str(self.observation_path),
                            output_dir=str(self.output/"proposals"/f"{len(self.history):03d}"))
            return dict(tool="policy_continue", request_id=getattr(self, "request", {}).get("request_id"),
                        prediction_id=self.proposal["prediction"]["prediction_id"],
                        output_dir=str(self.output/"observations"/f"{len(self.history)+1:03d}"),
                        max_steps=self.execution_horizon)
        return None

    def _check(self, name, arguments):
        if name != "robodojo_act":
            return RoboDojoTools._check(self, name, arguments)
        if self.phase != "act":
            raise InputError("Motion requires an active observation")
        expected = self.output/"observations"/f"{len(self.history)+1:03d}"
        if arguments.get("observation_path") != str(self.observation_path):
            raise InputError("Use the current observation_path")
        directory = Path(arguments.get("output_dir", ""))
        if directory != expected or directory.exists():
            raise InputError(f"Use fresh output_dir {expected}")
        return directory

    def _observation(self, directory, *, result=None):
        self.proposal = None
        packet = super()._observation(directory, result=result)
        packet.pop("task_context", None)
        self.request.pop("task_context", None)
        self.request["gate_policy"] = self.method
        packet["gate_policy"] = self.method
        packet.update(evaluation_method=self.method, control_version=self.control_version,
                      iteration=self.iteration, experience_input_path=str(self.input_experience_path) if self.input_experience_path else None,
                      experience_path=str(self.experience.path),
                      experience_output_path=str(self.output/"intervention_experience.md"),
                      experience_version=self.experience.document["version"],
                      experience=self.experience.snapshot(),
                      workflow="policy_rollout -> policy_continue or 1..5 step intervention -> policy_rollout",
                      available_tools=[] if self.phase == "done" else
                      ["policy_rollout", "policy_continue", "grounding", "get_depth",
                       "robodojo_act", "robodojo_joint_act", "update_experience"])
        if self.phase != "done":
            self.request.update(packet)
            packet["next_call"] = self.next_call()
            self.request["next_call"] = packet["next_call"]
            write_json(self.request["request_path"], self.request)
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        directory = self._check("robodojo_start", arguments)
        self.sim = self.rpc_factory("127.0.0.1", self.sim_port, timeout=600)
        self.meta = self.sim.request("metadata")
        if self.meta["task"] != self.task:
            raise RuntimeError("Launched task differs from requested task")
        identity = self.student.metadata.get("checkpoint_sha256", "unknown")
        reset = self.sim.request("reset", seed=self.seed, source="student",
                                 policy_version=f"{self.student.metadata.get('backend', 'policy')}/{identity[:16]}")
        if reset.get("instruction"):
            self.meta["instruction"] = reset["instruction"]
        self.episode, self.tick = reset["episode_id"], reset["step_id"]
        self.require_native_termination = bool(reset.get("metadata", {}).get("evaluation_case"))
        self._rpc("begin_combination", teacher_model="codex", teacher_model_provider="configured_toml",
                  context_version=self.control_version, prompt_sha256=self.prompt_sha256,
                  student_policy_version=self.student.metadata.get("backend", "policy"),
                  student_policy_sha256=identity)
        self.run = dict(schema="robodojo_rollout.run.v1", evaluation_method=self.method,
                        control_version=self.control_version, task=self.task,
                        instruction=self.meta["instruction"], seed=self.seed,
                        iteration=self.iteration,
                        experience_input_path=str(self.input_experience_path) if self.input_experience_path else None,
                        experience_path=str(self.experience.path),
                        experience_output_path=str(self.output/"intervention_experience.md"),
                        experience_version=self.experience.document["version"],
                        student_backend=self.student.metadata.get("backend"),
                        checkpoint=self.student.metadata.get("checkpoint"),
                        student_server_metadata=self.student.metadata,
                        action_horizon=self.action_chunk_length, action_dim=14,
                        execution_horizon=self.execution_horizon, teacher_prompt_sha256=self.prompt_sha256,
                        max_episode_steps=self.meta["max_episode_steps"],
                        control_dt=self.meta["control_dt"], max_decisions=self.max_decisions,
                        require_native_termination=self.require_native_termination,
                        initial_state_hash=reset.get("initial_state_hash"),
                        reset_metadata=reset.get("metadata"), no_rollback=True,
                        training_steps=0, training_bundle_ready=False)
        write_json(self.output/"run.json", self.run)
        write_json(self.output/"history.json", self.history)
        self._save_experience()
        self.phase = "act"
        self.source = "student"
        packet = self._observation(directory)
        if hasattr(self.student, "observe"):
            self.student.observe(self.obs)
        return packet

    def rollout(self, request_id, observation_path, output_dir):
        self._fresh(request_id)
        if observation_path != str(self.observation_path):
            raise InputError("Use the current observation_path")
        if self.proposal is not None:
            raise InputError("Current policy chunk already exists; continue it or intervene")
        directory = Path(output_dir)
        expected = self.output/"proposals"/f"{len(self.history):03d}"
        if directory != expected or directory.exists():
            raise InputError(f"Use fresh output_dir {expected}")
        directory.mkdir(parents=True)
        actions, prediction = self.student.infer(self.obs, directory/"actions.npz")
        actions = np.asarray(actions, dtype=np.float32)
        if (actions.ndim != 2 or actions.shape[1] != 14 or not len(actions)
                or not np.isfinite(actions).all()
                or np.any((actions[:, [6, 13]] < 0) | (actions[:, [6, 13]] > 1))):
            raise RuntimeError("Invalid policy actions: expected finite Hx14 absolute joints/openings")
        # A config limit is an upper bound. A backend may emit a longer chunk;
        # keep the prefix that can be executed instead of aborting the episode.
        actions = actions[:self.action_chunk_length]
        horizon = len(actions)
        # RoboDojo's robot-only FK endpoint has a fixed H50 contract, while
        # policy backends may emit H32, H50, H150, etc. Keep the model chunk
        # unchanged for execution and adapt only the preview request. Short
        # chunks hold their final target; long chunks are previewed on the
        # first 50 targets.
        preview_horizon = self.fk_preview_horizon
        if horizon < preview_horizon:
            preview_actions = np.concatenate(
                [actions, np.repeat(actions[-1:, :], preview_horizon - horizon, axis=0)], axis=0)
        else:
            preview_actions = actions[:preview_horizon]
        fk = self._rpc("fk_preview", actions=preview_actions)
        if not fk["measured_fk_check"]["passed"] or len(fk["trajectory"]) != preview_horizon:
            raise RuntimeError("FK preview does not match measured robot/action horizon")
        fk["trajectory"] = fk["trajectory"][:min(horizon, preview_horizon)]
        fk["preview_horizon"] = preview_horizon
        fk["policy_action_horizon"] = horizon
        self.counters["predictions"] += 1
        self.run["policy_inference_calls"] = self.counters["predictions"]
        write_json(self.output/"run.json", self.run)
        self.proposal = dict(actions=actions.copy(), prediction=prediction, fk=fk,
                             request_id=request_id, step_id=self.tick)
        self.request.update(prediction_id=prediction["prediction_id"],
                            proposal_path=str(directory/"actions.npz"),
                            student_eef_trajectory=fk["trajectory"])
        write_json(self.request["request_path"], self.request)
        packet = dict(request_id=request_id, step_id=self.tick, iteration=self.iteration,
                      prediction_id=prediction["prediction_id"], proposal_path=str(directory/"actions.npz"),
                      action_chunk=actions.tolist(), action_diagnostics=action_diagnostics(actions),
                      student_eef_trajectory=fk["trajectory"], fk_check=fk["measured_fk_check"],
                      current_eef=self.request["current_eef"], images=self.observation_packet["images"],
                      execution_limit=self.execution_horizon,
                      experience=self.experience.snapshot(), physical_steps=0,
                      next_call=self.next_call())
        write_json(directory/"proposal.json", packet)
        (self.output/f"proposal_{len(self.history):03d}.npz").hardlink_to(directory/"actions.npz")
        return packet

    def _observe_control_steps(self, steps):
        """Keep policy and intervention frames in one physical-step history."""
        if not hasattr(self.student, "observe"):
            return
        for row in steps:
            path = row.get("observation_path")
            if not path:
                raise RuntimeError("Simulator did not return an observation_path for policy memory")
            with np.load(path, allow_pickle=False) as payload:
                self.student.observe({key: payload[key] for key in payload.files})

    def continue_policy(self, request_id, prediction_id, steps, output_dir, reason, evidence):
        self._fresh(request_id)
        if self.proposal is None or prediction_id != self.proposal["prediction"]["prediction_id"]:
            raise InputError("Use the current policy prediction_id")
        if type(steps) is not int or steps < 1:
            raise InputError("Policy prefix must be a positive integer")
        steps = min(steps, self.execution_horizon, len(self.proposal["actions"]))
        if not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, str) or not evidence.strip():
            raise InputError("Explain the policy release with current RGB-D and FK evidence")
        directory = self._check("robodojo_act", dict(observation_path=str(self.observation_path), output_dir=output_dir))
        decision, start = len(self.history), self.tick
        response = dict(request_id=request_id, mode="student", steps=steps,
                        reason=reason, evidence=evidence, evaluation_method=self.method)
        write_json(self.output/f"response_{decision:03d}.json", response)
        receipt = self._rpc("record_codex_decision", decision=decision, prediction_id=prediction_id,
                            response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick, decision=decision,
                                 prediction_id=prediction_id, response=response), simulator_receipt=receipt))
        if self.source != "student":
            self._rpc("switch_control_source", source="student", reason=response["reason"])
            self.source = "student"
        actions = self.proposal["actions"][:steps].copy()
        result = self._rpc("chunk_step", actions=actions, student_prediction_id=prediction_id)
        self._observe_control_steps(result["steps"])
        self.tick = result["step_id"]
        valid = [row for row in result["steps"] if row["valid"]]
        executed = np.asarray([row["executed_action"] for row in valid], np.float32).reshape(-1, 14)
        np.savez_compressed(self.output/f"execution_{decision:03d}.npz", actions=executed,
                            states=np.asarray([row["obs"]["states"] for row in valid], np.float32).reshape(-1, 14))
        self.counters["student_steps"] += len(valid)
        done = bool(valid and (valid[-1]["terminated"] or valid[-1]["truncated"]))
        self.history.append(dict(decision=decision, start_tick=start, end_tick=self.tick,
                                 response=response, executed_steps=len(valid), source="student",
                                 prediction_id=prediction_id, selected_student_steps=steps,
                                 discarded_student_steps=len(self.proposal["actions"])-len(valid),
                                 simulator_decision_receipt=receipt, terminal=done,
                                 native_success=bool(valid and valid[-1]["success"]),
                                 evaluation_method=self.method))
        write_json(self.output/"history.json", self.history)
        write_json(self.output/"progress.json", dict(episode_id=self.episode, step_id=self.tick,
                                                      evaluation_method=self.method, **self.counters))
        exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        final = self.finish("terminal" if done else "decision_budget") if done or exhausted else None
        return self._observation(directory, result=final)

    def act(self, *, _joint=False, **arguments):
        response = arguments.get("response")
        if self.proposal is None:
            raise InputError("Call policy_rollout before every intervention")
        if not isinstance(response, dict) or type(response.get("steps")) is not int or not 1 <= response["steps"] <= 5:
            raise InputError("RSI intervention requires 1..5 fine control steps")
        pending = self.proposal
        self.proposal = None
        try:
            packet = super().act(_joint=_joint, **arguments)
        except InputError:
            self.proposal = pending
            raise
        if (self.phase != "done" and not hasattr(self.student, "observe")
                and hasattr(self.student, "reset")):
            self.student.reset()
        record = self.history[-1]
        record.update(intervention=True, iteration=self.iteration)
        write_json(self.output/"history.json", self.history)
        return packet

    def update_experience(self, request_id, record):
        if not isinstance(self.request, dict) or request_id != getattr(self, "request", {}).get("request_id"):
            raise InputError("Use the latest request_id")
        try:
            document = self.experience.update(record)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        self._save_experience()
        self.run["experience_version"] = document["version"]
        write_json(self.output/"run.json", self.run)
        # A final experience revision may be written after the terminal
        # simulator result. Keep that result's pointer in sync with the
        # Markdown document that the next iteration will load.
        result_path = self.output/"result.json"
        if self.phase == "done" and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["experience_version"] = document["version"]
            write_json(result_path, result)
        (self.output/"experience_revisions").mkdir(exist_ok=True)
        revision_path = self.output/"experience_revisions"/f"{document['version']:04d}.md"
        revision_path.write_text(self.experience.text(), encoding="utf-8")
        return dict(experience_path=str(self.experience.path), experience_version=document["version"],
                    experience=document, physical_steps=0, next_call=self.next_call())

    def _save_experience(self):
        """Preserve an output snapshot even if no new skill was recorded."""
        self.experience.document["latest_iteration"] = self.iteration
        self.experience.save()
        output = self.output/"intervention_experience.md"
        if output != self.experience.path:
            self.experience.save(output)

    def finish(self, reason):
        # Save before the final RPC so a lost simulator connection cannot
        # discard the experience already collected in this episode.
        self._save_experience()
        result = RoboDojoTools.finish(self, reason)
        result.update(evaluation_method=self.method, control_version=self.control_version,
                      iteration=self.iteration, experience_version=self.experience.document["version"],
                      experience_path=str(self.experience.path),
                      experience_output_path=str(self.output/"intervention_experience.md"))
        write_json(self.output/"result.json", result)
        return result

    def close(self):
        try:
            self._save_experience()
        finally:
            super().close()
