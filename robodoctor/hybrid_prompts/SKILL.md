---
name: robodojo-hybrid-rollout
description: Plan RoboDojo subtasks, supervise pi05 with camera/grounding/FK evidence, and intervene using absolute dual-arm control.
---

You own the plan and intervention decisions; pi05 supplies short low-level action
proposals. Use the nine recorded tools to control this one episode. Read
context/eef_control.md for coordinate conventions and context/direct_control.md
for the live TODO, one-shot demonstration, reachability, grasp and obstacle rules.
This hybrid workflow replaces the direct-only scheduling workflow; all geometry,
demonstration prerequisites and physical control rules still apply.

## Plan before execution

Call robodojo_start once. Inspect all three cameras and the task instruction.
Create concrete subtasks with src, dst, cond, per-arm reachability, assigned arms,
status, causal dependencies, next action and observed evidence. Call update_plan
before requesting any policy action or making a direct move. The tool persists
the full plan and REPLACES the TODO list in TODO.md, retaining plan versions in
plans/. TODO.md is automatically injected with every service-tool result and
new turn. Its four sections are Episode information, TODO list, Geometry and
orientation, Task Progress (see context/direct_control.md). Before every policy
prefix or direct move, fully refresh action-relevant geometry/orientation and
progress with native file tools; update_plan alone does not update these sections.
Use the complete current state, not appended deltas. After identifying pi05 intent,
retain only the geometry for the selected prefix/correction and remove obsolete
candidates. Keep the TODO list consistent with update_plan; revise that tool's
full plan if assignments/status/dependencies change.

The host archives changed TODO snapshots and executed-step records in NOTES.md.
Append detailed observations, demonstrations, uncertainties, corrections and
plan-change explanations there. NOTES.md is long-term memory, never automatically
injected; query relevant excerpts only when needed. Do not load all revisions.

Use stable subtask IDs. An independent object's task may execute before another
without violating the goal; a placement-order or support dependency may not.
Use separate single-arm HOLD/OBSERVE subtasks when an arm must wait. Use a
coordinated_dual OBS_DEMO subtask assigned to both arms for watching a demonstration;
all manipulation depends on confirmed demonstrator return, not just final release.
For handoffs, split source staging and receiver regrasp into dependent segments.
Do not mark completion from a predicted EEF path or mere pose arrival.

## Identify what pi05 is actually attempting

1. Call pi05_infer using the current request_id, observation_path, and
   controller/proposals/NNN (NNN is the next decision number, starting at 000).
   It predicts 50 absolute joint/opening actions but executes nothing. Inspect
   student_eef_trajectory for both arms: link6 positions, wxyz orientations and
   gripper openings, all in environment_origin. The FK check must pass.
2. Compare the path and gripper sequence with head + left wrist + right wrist
   RGB, current measured poses and recent actual execution. Ground visible
   candidate source objects/destinations to compare metric locations with FK.
   Account for link6-to-finger offset, object extent and intended approach;
   nearest surface distance alone is not a reliable intent classifier. Grounding
   provides coarse localization, not a required grasp/contact coordinate. Judge
   final alignment using fresh head/own-wrist RGB and wrist get_depth; do not
   reject a valid grasp merely because its EEF differs from the grounded point.
3. For each arm, identify the likely subtask/phase (approach, grasp, lift,
   transfer, release, retreat, hold), confidence and concise visible evidence.
   Infer actual progress from fresh images: predicted closure is not a grasp,
   and a predicted transfer is not evidence that an object moved. FK predicts
   robot-only targets, not future object motion, contact or guaranteed tracking.
4. If pi05 is safely advancing a different independent subtask, update_plan to
   reflect that execution order and current arm assignments. Do not intervene
   solely to force your original ordering. Preserve required dependencies and
   placement order. Reopen/split subtasks after slips, bad placements or a handoff
   change. Record uncertainty rather than inventing intent or completion.
5. Call review_pi05 for the current prediction_id and plan. Include both arms'
   subtask IDs, confidence, evidence and actual same-observation grounding_ids
   where useful. Report failure_detected, path_unsafe and dependency_violation.
   Unknown or low-confidence intent requires further inspection or intervention;
   it is not permission to execute a speculative long chunk.

## Execute a prefix or intervene

ALLOW: choose safe_steps=1..15 and call pi05_execute with no more than that
prefix. Near contact, release, uncertain tracking or a subtask boundary use
1..5 steps. Inspect all available FK points for intent, but authorize only the
prefix whose full arm/finger/carried-object route is safe. The tools return fresh
three-camera observations after each prefix. There is no automatic continuation
or model observation between steps within a prefix; choose its length accordingly.
For precision operations such as pouring into a small opening or inserting into
a small hole, require a single manipulating arm. Assign the other arm a
single-arm HOLD/OBSERVE subtask, park it at the side outside the active arm's
swept path, and aim its wrist camera at the operation area. Keep that arm fixed
and use its wrist view to assess alignment or flow between short prefixes; do
not authorize concurrent manipulation by the observation arm. Reposition it
only after the precision operation is complete or the area is clear.
For pouring liquid, first move the observation wrist camera to a stable
side/front view that shows the bottle mouth, the container's inner rim and the
stream together. Review fresh observations after each short tilt increment.
During downward rotation, the bottle mouth's full path must remain inside the
container's inner boundary with visible margin from the rim. If the mouth nears
or crosses that boundary, intervene by stopping, reducing tilt and recentering
before authorizing more motion; do not infer safety from the bottle body pose.

For every grasp, require a broad, continuous, relatively flat contact region
with both fingers opposing the object. Avoid thick-to-thin boundaries,
shoulders/necks, tapered transitions, sloped faces and edges, which can make the
object slide or rotate out. If the proposed contact is not stable, intervene,
reposition and reobserve before authorizing closure; verify a short test lift
and reopen/regrasp after any slip.
For a bottle whose lower body and upper neck have different widths, grip
a broad, uniform-width section of the lower bottle body with full opposing
finger-pad contact. Do not grasp the narrow neck, mouth, or middle
shoulder where the width changes; those contacts can slip during lifting
or pouring.

INTERVENE: choose safe_steps=0 if the next prefix targets a wrong object/location,
violates a task dependency, enters an unsafe route, or follows an actual failed
grasp, slip, drop, misplaced object or stalled motion. Describe the evidence and
correction in NOTES.md and update the relevant subtasks. Use robodojo_act with
grounding/get_depth and the absolute dual-arm numerical IK control to hold,
withdraw, reposition, regrasp or finish the necessary correction. Keep corrections
short near contact; do not execute a known unsafe prefix just to gather evidence.
If IK fails or measured motion stalls, use robodojo_joint_act to preview and
execute a small joint-space retreat/reconfiguration, then retry EEF IK. Read
context/eef_control.md for joint ordering, delta semantics and collision limits.
Supply reason exactly as `left: <ID> <reason>, right: <ID> <reason>`, starting
each arm description with its plan subtask ID and then its purpose/evidence.
Use the same ID for a coordinated subtask; use HOLD for an idle arm with no subtask.

After EVERY policy prefix or direct correction, re-evaluate the returned RGB and
measured poses and call update_plan before further inference/motion. The remaining
pi05 trajectory is discarded, including after a correction; never resume it.
Continue direct correction only while needed, then request a new pi05 prediction
from the corrected state. A plan update at the same physical step preserves the
proposal but invalidates its review, so review again before releasing it.

## Examples

The plan lists A (left object to tray) then B (right object to box), with no
dependency. FK and wrist RGB show pi05 approaching B safely while the left arm
holds. Mark B in_progress, keep A pending, assign a left HOLD subtask, review and
execute a short safe prefix. Do not force pi05 back to A just because A was first.

If B must be placed after A, the same proposal violates that prerequisite.
Intervene before B's conflicting operation, use direct control to establish the
required state, and infer again. Never delete a dependency to justify the proposal.

If FK predicts a lift with closed fingers but fresh wrist RGB shows the object
still on the table, record a failed grasp and reopen grasp/lift. Hold the other
arm, ground/reinspect the source, refine alignment and grasp directly, verify a
small test lift, then let pi05 infer from the new state.

During a one-shot demonstration, use short direct hold-pose actions with
stop_on_reach=false to obtain fresh observations and record the sequence in
NOTES.md. Keep both arms stationary until the demonstrator visibly returns home.
No policy prefix may bypass this prerequisite.

Stop at native rollout_finished. Do not reset, replay uncertain ACKs, open a
second simulator connection, inspect hidden scene/object truth or alter task
rules. All model-authored notes, reasons and reports must be English.
