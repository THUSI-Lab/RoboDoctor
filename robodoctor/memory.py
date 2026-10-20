"""Small current-state memory, with historical notes loaded only on request."""
import json
from pathlib import Path
import re

SECTIONS = ('Episode information', 'TODO list', 'Geometry and orientation', 'Task Progress')


def initial_todo():
    values = ('Episode not started; read the first observation for task and reset details.',
              'Not planned yet. Replace with the complete current subtask list before motion.',
              'No current action geometry verified yet.',
              'No physical progress yet. Inspect observations and plan before motion.')
    return '# Current episode memory\n\n' + ''.join(
        f'## {section}\n\n{value}\n\n' for section, value in zip(SECTIONS, values))


def replace_section(text, section, value):
    pattern = rf'^## {re.escape(section)}\n.*?(?=^## |\Z)'
    replacement = f'## {section}\n\n{value.strip()}\n\n'
    if re.search(pattern, text, flags=re.M | re.S):
        return re.sub(pattern, lambda _: replacement, text, flags=re.M | re.S)
    return text.rstrip() + '\n\n' + replacement


def write_plan_todo(path, subtasks, revision, step):
    """Replace the full plan, preserving the other current-memory sections."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else initial_todo()
    body = f'Current plan revision {revision}, observed at step {step}.\n'
    for task in subtasks:
        # All validated fields are current values, never accumulated revisions.
        body += '\n- ' + json.dumps(task, ensure_ascii=False, separators=(',', ':'))
    path.write_text(replace_section(text, 'TODO list', body))


class EpisodeMemory:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.todo = self.workspace/'TODO.md'
        self.notes = self.workspace/'NOTES.md'
        self.previous = None
        self.revision = 0

    def capture(self, *, step=None):
        """Archive changed snapshots without ever reading long-term notes."""
        text = self.todo.read_text() if self.todo.exists() else initial_todo()
        if text != self.previous:
            self.revision += 1
            with self.notes.open('a') as stream:
                stream.write(f'\n## TODO revision {self.revision} (step {step})\n\n'
                             + text.rstrip() + '\n')
            self.previous = text
        return dict(path=str(self.todo), revision=self.revision, content=text,
                    notes_path=str(self.notes),
                    instruction='Current compressed memory; supersedes older TODO snapshots. '
                    'Reconcile with the latest observation before action. '
                    'NOTES.md is long-term memory; query only relevant excerpts when needed.')

    def episode(self, rollout):
        """Refresh measured episode metadata; preserve agent-owned planning sections."""
        obs = getattr(rollout, 'obs', {}) or {}
        meta = getattr(rollout, 'meta', {}) or {}
        run = getattr(rollout, 'run', {}) or {}
        reset = run.get('reset_metadata') or {}
        data = dict(task=rollout.task, episode_id=getattr(rollout, 'episode', None),
                    step_id=rollout.tick, phase=rollout.phase,
                    instruction=obs.get('instruction', meta.get('instruction')),
                    remaining_steps=obs.get('remaining_steps'),
                    max_episode_steps=meta.get('max_episode_steps'),
                    seed=getattr(rollout, 'seed', None), eval_seed=reset.get('eval_seed'),
                    layout_id=reset.get('layout_id'), variant=reset.get('variant'))
        text = self.todo.read_text() if self.todo.exists() else initial_todo()
        self.todo.write_text(replace_section(text, 'Episode information',
                                             json.dumps(data, ensure_ascii=False, indent=2)))

    def step_note(self, tool, arguments, before, after, packet, result_path):
        if before == after:
            return
        response = arguments.get('response', {})
        note = dict(tool=tool, step_before=before, step_after=after,
                    reason=response.get('reason'), result_path=str(result_path),
                    movement=packet.get('movement'), rollout_finished=packet.get('rollout_finished'))
        with self.notes.open('a') as stream:
            stream.write(f'\n## Step note {before} -> {after}\n\n'
                         + json.dumps(note, ensure_ascii=False, default=str) + '\n')
