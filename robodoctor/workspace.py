from pathlib import Path
import json
import shutil
from .memory import initial_todo
from hybrid_rollout.robodojo.skill.workspace import prepare_workspace as original_workspace


def hybrid_skill_root(audit):
    """Snapshot current shared control rules without maintaining a second copy."""
    source = Path(__file__).parent
    root = Path(audit)/'skill_source'
    shutil.copytree(source/'hybrid_prompts', root)
    direct = (source/'prompts/SKILL.md').read_text()
    (root/'context/direct_control.md').write_text(direct[direct.index('## One-shot learning:'):])
    eef = (source/'prompts/context/eef_control.md').read_text().replace(
        'robodojo_act combines', 'robodojo_act is the direct correction tool combining')
    (root/'context/eef_control.md').write_text(eef)
    return root


def prepare_workspace(audit, skill_root=None, *, method='gpt_only'):
    if method == 'rsi_policy_doctor':
        return prepare_rsi_workspace(audit, skill_root)
    skill_root = skill_root or Path(__file__).parent/'prompts'
    workspace = original_workspace(audit, skill_root, method=method)
    path = workspace/'AGENTS.md'
    instructions = path.read_text().replace('Two additional tools', 'Five additional tools')
    if method == 'pi05_plus_gpt':
        instructions = instructions.replace('Three additional tools', 'Nine additional tools').replace(
            'injected rollout skill and unchanged gate', 'injected hybrid skill and subtask-evidence review')
        instructions += ('\nPlan first with update_plan. Review pi05 FK against all three cameras and grounding; '
                         'adapt valid independent ordering, intervene on errors, and infer afresh after correction.\n')
        metadata_path = workspace/'workspace.json'
        metadata = json.loads(metadata_path.read_text())
        controller = Path(audit).parent
        metadata.update(evaluation_method=method, source_root=str(Path(__file__).resolve().parent.parent),
                        plan_path=str(controller/'plan.json'), reviews_path=str(controller/'reviews'))
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    instructions += ('\nUse grounding in environment_origin and unified dual-arm robodojo_act for coarse and fine motion. '
                     'Use get_depth on either wrist camera to refine grasp approach. '
                     'Use robodojo_joint_act for previewed absolute/delta qpos recovery when IK fails or motion stalls. '
                     'Read the coordinate-frame and pixel-resolution contract carefully.\n')
    instructions = instructions.replace(
        'Keep working memory in NOTES.md and code, crops, ',
        'Keep current compressed memory in TODO.md, long-term step notes in NOTES.md, and code, crops, ')
    instructions += ('\nTODO.md is automatically injected with service results and at each new turn. '
                     'Fully rewrite its four current-state sections: Episode information, TODO list, '
                     'Geometry and orientation, Task Progress. Do not append incremental updates or old plans. '
                     'Before each action reconcile the latest observation and retain only geometry relevant '
                     'to the selected action. The host archives TODO revisions and executed-step records in '
                     'NOTES.md; append additional step notes and demonstration details there. '
                     'NOTES.md is never automatically injected: query relevant excerpts only when needed.\n')
    (workspace/'TODO.md').write_text(initial_todo())
    (workspace/'NOTES.md').write_text('# Long-term episode memory\n\n'
        'TODO revision history, step notes, demonstration details and past corrections. '
        'Query relevant excerpts when needed; current planning belongs in TODO.md.\n')
    metadata_path = workspace/'workspace.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(todo_path=str(workspace/'TODO.md'), notes_path=str(workspace/'NOTES.md'),
                    todo_auto_injected=True, notes_auto_injected=False)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    path.write_text(instructions)
    return workspace


def prepare_rsi_workspace(audit, skill_root=None):
    audit = Path(audit).resolve()
    method = "rsi_policy_doctor"
    source = Path(skill_root or Path(__file__).parent / "rsi_prompts")
    agent = audit / "agent"
    agent.mkdir(parents=True, exist_ok=False)
    (agent / "scratch").mkdir()
    (agent / "context").mkdir()
    for path in (source / "context").glob("*.md"):
        (agent / "context" / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    (agent / "SKILL.md").write_text((source / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")
    (agent / "AGENTS.md").write_text(
        "Use only the RSI policy doctor skill and current rollout service tools. "
        "Inspect current RGB-D, policy actions, measured FK and same-episode artifacts. "
        "Do not load examples, old plans or legacy control prompts; do not reset or bypass the simulator.\n",
        encoding="utf-8")
    (agent / "workspace.json").write_text(json.dumps({
        "controller_output": str(audit.parent), "evaluation_method": method,
        "observations_path": str(audit.parent / "observations"),
        "proposals_path": str(audit.parent / "proposals"),
        "scratch": str(agent / "scratch"),
    }, indent=2) + "\n", encoding="utf-8")
    return agent
