"""Only same-episode scratch state; external experience lives in ExperienceStore."""
import json
from pathlib import Path


class RSIMemory:
    def __init__(self, workspace):
        self.path = Path(workspace)/"CURRENT.md"
        self.path.write_text("# Current episode\n\nNo observation yet.\n")

    def capture(self, *, step=None):
        return dict(path=str(self.path), content=self.path.read_text(),
                    instruction="Current episode state only; inspect the latest observation and policy FK.")

    def episode(self, rollout):
        obs = getattr(rollout, "obs", {}) or {}
        value = dict(task=rollout.task, iteration=getattr(rollout, "iteration", None),
                     episode_id=getattr(rollout, "episode", None), step_id=rollout.tick,
                     phase=rollout.phase, instruction=obs.get("instruction"),
                     remaining_steps=obs.get("remaining_steps"),
                     experience_path=str(rollout.experience.path) if rollout.experience.path else None,
                     experience_version=rollout.experience.document["version"])
        self.path.write_text("# Current episode\n\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def step_note(self, tool, arguments, before, after, packet, result_path):
        pass
