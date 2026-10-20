"""Durable direct-control task lessons and optional experience from the prior iter."""
import json
from pathlib import Path


def validate_task_experience(path):
    if path is None:
        return
    path = Path(path)
    if path.suffix.lower() != '.md' or not path.is_file() or not path.read_text(encoding='utf-8').strip():
        raise ValueError('--task-experience requires a nonempty Markdown file')


class TaskExperience:
    def __init__(self, controller, agent, *, source=None, iteration=0):
        validate_task_experience(source)
        if iteration < 0:
            raise ValueError('iteration must be non-negative')
        self.controller, self.agent = Path(controller), Path(agent)
        self.iteration = iteration
        self.output = self.controller/'task_experience.md'
        self.draft = self.agent/'TASK_EXPERIENCE.md'
        self.prior = None
        if source is not None:
            self.prior = self.agent/'PRIOR_TASK_EXPERIENCE.md'
            self.prior.write_text(Path(source).read_text(encoding='utf-8'), encoding='utf-8')
        path = self.agent/'workspace.json'
        metadata = json.loads(path.read_text())
        metadata.update(iteration=iteration, task_experience_input=str(self.prior) if self.prior else None,
                        task_experience_draft=str(self.draft), task_experience_output=str(self.output))
        path.write_text(json.dumps(metadata, indent=2)+'\n')
        with (self.agent/'AGENTS.md').open('a', encoding='utf-8') as stream:
            stream.write(self.instructions())

    def instructions(self):
        text = ('\nDIRECT TASK EXPERIENCE: After this episode finishes, summarize reusable task lessons '
                f'in {self.draft}. Use English Markdown. Include verified useful methods, failures and '
                'their evidence, improvements for the next iteration, and unresolved uncertainties. '
                'Consult current TODO.md, relevant NOTES.md and recorded service results. Distinguish '
                'verified outcomes from hypotheses; do not infer success from a completed movement. '
                'Use object-relative descriptions, not an absolute-coordinate action replay. Preserve '
                'applicable prior lessons, revise contradicted ones, and add new evidence.\n')
        if self.prior:
            text += (f'Prior task/intervention experience is available in {self.prior}. Read it before '
                     'planning motion and use it as reference for this fresh episode. Its progress and '
                     'coordinates are historical, not current observations: reobserve and re-ground.\n')
        return text

    @staticmethod
    def _read_json(path):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def save(self, rollout):
        """Keep a model summary when available, with host-owned native outcome evidence."""
        result = self._read_json(self.controller/'result.json')
        outcome = self._read_json(self.controller.parent/'sim/evaluation_outcome.json')
        draft = self.draft.read_text(encoding='utf-8').strip() if self.draft.exists() else ''
        valid = outcome.get('valid_for_success_rate') is True and outcome.get('complete') is True
        text = ('# Direct Control Task Experience\n\n'
                f'- Task: {rollout.task}\n- Iteration: {self.iteration}\n'
                f'- Seed: {getattr(rollout, "seed", None)}\n'
                f'- Episode: {getattr(rollout, "episode", None)}\n- Control steps: {rollout.tick}\n'
                f'- Finish reason: {result.get("reason", "not finalized")}\n'
                f'- Native evaluation valid: {str(valid).lower()}\n'
                f'- Native success: {outcome.get("native_success") if valid else "unknown"}\n'
                f'- Native score (0–1): {outcome.get("native_score") if valid else "unknown"}\n'
                f'- Summary source: {"model" if draft else "recorded evidence fallback"}\n\n'
                'Past experience is reference material. Reobserve and re-ground in the next episode; '
                'do not copy historical progress or absolute coordinates as current facts.\n\n')
        if draft:
            text += '## Reusable task lessons\n\n'+draft+'\n'
        else:
            text += ('## Recorded evidence (summary unavailable)\n\n'
                     'No completed model summary is available. The following material preserves '
                     'prior experience and this episode’s agent notes; hypotheses and visual claims '
                     'are not verified native success.\n')
            for title, path in [('Prior experience', self.prior),
                                ('Final episode memory', self.agent/'TODO.md'),
                                ('Episode notes and step evidence', self.agent/'NOTES.md')]:
                if path and path.exists():
                    text += f'\n### {title}\n\n'+path.read_text(encoding='utf-8').rstrip()+'\n'
        temporary = self.output.with_suffix('.md.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(self.output)
        return self.output
