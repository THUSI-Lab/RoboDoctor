"""Markdown skill memory for the RSI policy doctor.

The on-disk experience is deliberately human-readable Markdown.  Runtime tool
packets may still use structured Python values, but no JSON is embedded in the
experience document.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any

_SCHEMA = "rsi_intervention_experience.v2"
_INTERACTION_FIELDS = ("target_object", "arm", "interaction_type", "contact_region",
                       "approach_direction", "eef_orientation", "gripper_sequence",
                       "motion_sequence", "success_evidence", "applicability")


def _clean(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "<depth-limit>"
    if isinstance(value, dict):
        return {str(k): _clean(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, depth=depth + 1) for v in list(value)[:128]]
    if hasattr(value, "tolist"):
        return _clean(value.tolist(), depth=depth + 1)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _lines(value: Any, indent: int = 0) -> list[str]:
    """Render nested values as Markdown bullets, never as serialized JSON."""
    pad = " " * indent
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            label = str(key).replace("_", " ")
            label = {"fk": "FK", "rgbd": "RGB-D"}.get(label.lower(), label[:1].upper() + label[1:])
            if isinstance(item, (dict, list, tuple)):
                result.append(f"{pad}- {label}:")
                result.extend(_lines(item, indent + 2))
            else:
                result.append(f"{pad}- {label}: {item}")
        return result or [f"{pad}- none recorded"]
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{pad}- none recorded"]
        result = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                result.append(f"{pad}-")
                result.extend(_lines(item, indent + 2))
            else:
                result.append(f"{pad}- {item}")
        return result
    return [f"{pad}- {value}"]


def _block(title: str, value: Any) -> list[str]:
    return [f"- {title}:", *_lines(value, 2)]


def _field(block: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}:\s*(.*)$", block, flags=re.M)
    return match.group(1).strip() if match else None


def _headings(block: str, prefix: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2).strip()) for match in re.finditer(
        rf"^###\s+({re.escape(prefix)}\d+)\s+—\s+(.+)$", block, flags=re.M)]


class ExperienceStore:
    def __init__(self, path: str | os.PathLike | None, *, iteration: int = 0, task: str = ""):
        self.path = Path(path).expanduser().resolve() if path else None
        if self.path is not None and self.path.suffix.lower() != ".md":
            raise ValueError("Intervention experience must be a .md Markdown path")
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        self.iteration = iteration
        self.task = task
        self.document = self._read()

    def _empty(self) -> dict:
        return {"schema": _SCHEMA, "version": 0, "task": self.task,
                "latest_iteration": self.iteration, "skills": []}

    def _read(self) -> dict:
        if self.path is None or not self.path.exists():
            return self._empty()
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return self._empty()
        if "<!-- RSI_JSON" in raw or raw.lstrip().startswith("{"):
            raise ValueError("Experience must be Markdown; convert the old JSON experience to .md")
        schema = _field(raw, "Schema")
        if schema and schema != _SCHEMA:
            raise ValueError(f"Experience must use schema {_SCHEMA}")
        value = self._empty()
        value["version"] = int(_field(raw, "Version") or 0)
        value["latest_iteration"] = int(_field(raw, "Latest iteration") or self.iteration)
        value["task"] = _field(raw, "Task") or self.task
        diagnosis = raw.split("## Policy diagnosis skills", 1)[-1]
        diagnosis, operations = (diagnosis.split("## Intervention operation information", 1)
                                  if "## Intervention operation information" in diagnosis else (diagnosis, ""))
        dheads = _headings(diagnosis, "D")
        oheads = dict(_headings(operations, "I"))
        for index, (skill_id, title) in enumerate(dheads):
            start = diagnosis.find(f"### {skill_id} — {title}")
            end = next((diagnosis.find(f"### {sid} —", start + 1) for sid, _ in dheads[index + 1:]), len(diagnosis))
            dblock = diagnosis[start:end].strip()
            operation_id = f"I{skill_id[1:]}"
            oblock = ""
            if operation_id in oheads:
                otitle = oheads[operation_id]
                ostart = operations.find(f"### {operation_id} — {otitle}")
                next_ops = [operations.find(f"### {sid} —", ostart + 1) for sid in oheads if sid != operation_id and operations.find(f"### {sid} —", ostart + 1) >= 0]
                oend = min(next_ops) if next_ops else len(operations)
                oblock = operations[ostart:oend].strip()
            value["skills"].append(dict(skill_id=skill_id, skill=title,
                                        diagnosis_markdown=dblock, operation_markdown=oblock,
                                        record_type=_field(dblock, "Record type") or "intervention",
                                        event=_field(dblock, "Event") or "", trigger=_field(dblock, "Trigger / intervention timing") or ""))
        return value

    def snapshot(self) -> dict:
        return _clean(self.document)

    def text(self) -> str:
        def saved_lines(markdown: str) -> list[str]:
            # Stored fragments may come from an older revision that included
            # its own heading, or from the current renderer which starts at a
            # field. Preserve every field in either case.
            fragment = markdown.splitlines()
            return fragment[1:] if fragment and fragment[0].startswith("### ") else fragment

        lines = ["# RSI Intervention Experience", "", "This document is the evolving policy-doctor memory. It contains reusable skills, not a fixed action script.", "", f"- Schema: {_SCHEMA}", f"- Version: {self.document['version']}", f"- Task: {self.document.get('task', self.task) or 'not specified'}", f"- Latest iteration: {self.document.get('latest_iteration', self.iteration)}", "", "## Policy diagnosis skills", "", "The first block records policy failure signatures and verified successful object interaction methods, supported by RGB-D, robot state, executed action and FK evidence.", ""]
        if not self.document["skills"]:
            lines.append("_No diagnosis skills recorded yet. The policy doctor starts from zero experience._")
        for item in self.document["skills"]:
            sid = item.get("skill_id", "D0000")
            title = item.get("skill", "unnamed diagnosis")
            lines.extend(["", f"### {sid} — {title}"])
            if item.get("diagnosis_markdown"):
                lines.extend(saved_lines(item["diagnosis_markdown"]))
            else:
                lines.extend(_block("Event", item.get("event", "")))
                lines.extend(_block("Trigger / intervention timing", item.get("trigger", "")))
                lines.extend(_block("Policy diagnosis", item.get("policy_state", {})))
                if item.get("notes"):
                    lines.extend(_block("Diagnosis notes", item["notes"]))
        lines.extend(["", "## Intervention operation information", "", "The second block records the successful policy baseline, grounded adaptations, failed attempts, verified outcomes and stop conditions. Successful interaction methods and adaptations are used only during fine intervention; normal policy prefixes remain unchanged.", ""])
        if not self.document["skills"]:
            lines.append("_No intervention operation information recorded yet._")
        for item in self.document["skills"]:
            sid = "I" + str(item.get("skill_id", "D0000"))[1:]
            title = item.get("skill", "unnamed intervention")
            lines.extend(["", f"### {sid} — {title}"])
            if item.get("operation_markdown"):
                lines.extend(saved_lines(item["operation_markdown"]))
            else:
                lines.extend(_block("Linked diagnosis", item.get("skill_id", "")))
                lines.extend(_block("Intervention operation", item.get("intervention", {})))
                lines.extend(_block("Stop intervention when", item.get("stop_condition", "")))
                lines.extend(_block("Outcome", item.get("outcome", "")))
        return "\n".join(lines).rstrip() + "\n"

    def save(self, path: str | os.PathLike | None = None) -> None:
        """Atomically save current skills without adding a revision or a skill."""
        destination = Path(path).expanduser().resolve() if path is not None else self.path
        if destination is None:
            return
        if destination.suffix.lower() != ".md":
            raise ValueError("Intervention experience must be a .md Markdown path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(self.text())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def update(self, record: dict, *, iteration: int | None = None) -> dict:
        if self.path is None:
            raise ValueError("No intervention experience path was supplied")
        if not isinstance(record, dict):
            raise ValueError("experience update must be an object")
        record_type = record.get("record_type", "intervention")
        if record_type not in ("policy_success", "policy_failure", "intervention"):
            raise ValueError("record_type must be policy_success, policy_failure or intervention")
        required = ("event", "skill", "trigger", "policy_state", "outcome")
        if record_type != "policy_success":
            required += ("intervention", "stop_condition")
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise ValueError("experience update missing: " + ", ".join(missing))
        policy_state = record["policy_state"]
        if not isinstance(policy_state, dict) or not all(policy_state.get(k) for k in ("observation", "action", "fk")):
            raise ValueError("policy_state must include observation, action and fk")
        interaction = record.get("policy_interaction")
        if record_type == "policy_success" or interaction is not None:
            if (not isinstance(interaction, dict)
                    or not all(isinstance(interaction.get(key), str) and interaction[key].strip()
                               for key in _INTERACTION_FIELDS)
                    or interaction.get("arm") not in ("left", "right", "both")):
                raise ValueError("policy_interaction must include " + ", ".join(_INTERACTION_FIELDS)
                                 + "; arm must be left, right or both")
        if record_type == "policy_success":
            if record.get("intervention"):
                raise ValueError("policy_success records autonomous policy execution, not an intervention")
        else:
            operation = record["intervention"]
            if not isinstance(operation, dict) or not all(operation.get(k) for k in ("timing", "tools", "technique", "object_interaction")):
                raise ValueError("intervention must include timing, tools, technique and object_interaction")
        iteration = self.iteration if iteration is None else int(iteration)
        skill = _clean(record)
        skill["record_type"] = record_type
        skill_id = str(skill.get("skill_id") or f"D{len(self.document['skills']) + 1:04d}")
        skill["skill_id"] = skill_id
        skill["iteration"] = iteration
        skill["task"] = self.task
        skill["diagnosis_markdown"] = "\n".join([f"- Record type: {record_type}"]
            + _block("Event", skill["event"]) + _block("Trigger / intervention timing", skill["trigger"])
            + _block("Policy diagnosis", skill["policy_state"])
            + (_block("Successful policy interaction", skill["policy_interaction"]) if interaction else [])
            + (_block("Diagnosis notes", skill["notes"]) if skill.get("notes") else []))
        operation_lines = _block("Linked diagnosis", skill_id)
        if record_type == "policy_success":
            operation_lines += _block("Intervention use", "No intervention performed. Use this verified policy method only as a re-grounded baseline during fine intervention; keep normal policy execution unchanged.")
        else:
            operation_lines += (_block("Intervention operation", skill["intervention"])
                                + _block("Stop intervention when", skill["stop_condition"]))
        skill["operation_markdown"] = "\n".join(operation_lines + _block("Outcome", skill["outcome"]))
        skills = [item for item in self.document["skills"] if item.get("skill_id") != skill_id]
        skills.append(skill)
        self.document.update(version=int(self.document.get("version", 0)) + 1,
                             task=self.task or self.document.get("task", ""), latest_iteration=iteration, skills=skills)
        self.save()
        return self.snapshot()
