---
name: robodojo-grounded-direct
description: Ground visible surfaces in environment_origin and control both arms with unified absolute EEF IK.
---

Control this single RoboDojo episode with five services: robodojo_start,
grounding, get_depth, robodojo_act and robodojo_joint_act. Read context/eef_control.md. Use task
text, RGB, measured robot state and same-episode history. Maintain a live subtask
TODO.md as compressed execution memory; revise it throughout execution. Keep
historical details in NOTES.md and query them only when needed.

The episode has a finite total control-step budget. Prefer simultaneous useful
work by both arms whenever their routes and task dependencies allow it. Before
every robodojo_act, check whether the other arm can safely advance another
grasp, transfer, placement, handoff staging, retreat or return-to-origin in the
same call instead of waiting. Do not default to finishing one arm's entire
pick-and-place sequence before starting the other. Both arms move in each
shared control step, so safe parallel progress can save episode steps; fewer
tool calls alone do not save steps. If no useful independent action is safe,
hold that arm explicitly. Preserve fine control near contact and obey all
collision checks; budget pressure never justifies an unsafe shortcut. Reserve
enough remaining steps for placement, release, retraction and required arm return.

1. Call robodojo_start once with its exact paths and inspect all three views.
   Create/update the TODO list, evaluate reachability and assign arms before
   the first motion. Use grounding/depth as needed to resolve uncertain entries.
   For one-shot learning tasks, complete the demonstration observation and
   demonstrator-return checks below before starting any manipulation motion.
2. Select a visible target surface pixel in the original head image. Call
   grounding(request_id, x, y). x is column, y is row, top-left origin; use
   images[].width/height. Never use normalized, cropped or resized coordinates.
   Invalid depth is an error: select another visible pixel.
3. The returned position is in environment_origin, shared by BOTH arms. It is
   a surface point, not an object center or a gripper grasp pose. Grounding is
   for coarse localization only: you do NOT have to grasp at the clicked point
   or make the final EEF position match its coordinates. Construct
   pre-operation targets above/beside it with offsets chosen from visible
   geometry, finger reach, orientation and held objects. No fixed numeric
   margin is required. Use current_eef as the starting pose reference and
   world_axes_in_env to interpret world-up.
4. Before emitting any action, select ready TODO subtasks with their causal
   dependencies satisfied and compatible arm assignments. Assess each arm's proposed trajectory and the
   combined dual-arm collision risk as described below. Then call robodojo_act
   with explicit left AND right absolute positions, wxyz
   quaternions and gripper_closed values in one response.target. Both arms can
   approach simultaneously. Hold unused arms at their current measured poses
   and gripper states. Use a larger steps budget up to 150 with
   stop_on_reach=true ONLY when confident the entire grasp-approach or movement
   path is free of obstacles. If obstacles may be present, visibility is partial
   or clearance is uncertain, use small budgets (typically 1..5) and short
   waypoints, inspecting after each call. Check movement.arms and
   movement.reached before continuing; do not assume an issued target arrived.
5. Use the SAME robodojo_act for orientation refinement, contact, grasp, lift,
   transfer, placement and release. Use small steps and stop_on_reach=false
   for fine control or gripper settling, inspecting fresh images after each
   call. Ground new objects/destinations to guide coarse localization.

There is no separate move_to_abs service or per-arm pre-positioning lock.
grounding and get_depth are read-only; motion returns a new observation and
invalidates the old request_id and grounding samples. Re-ground when needed.

## Joint recovery after failed IK or stalled movement

Use robodojo_joint_act to adjust measured arm qpos directly, in absolute radians
or one-time deltas, when EEF IK fails or the arm cannot move along its chosen
path. Read context/eef_control.md's joint recovery contract and example. Inspect
RGB/current_joints, preview the target FK without motion, then make a small
clearance-increasing change and reobserve before trying EEF IK again. Preserve
gripper states and hold the other arm explicitly when routes conflict. Neither
numerical IK nor direct joints checks collisions; do not push through contact.

## One-shot learning: observe, record, then act

When the task provides a demonstration, learn from fresh observations throughout
the demonstration. Keep BOTH controlled arms at their current measured poses
and preserve their gripper states while watching. Advance simulation with short
hold-pose robodojo_act calls using stop_on_reach=false, then inspect the returned
head/wrist images with the new request_id. Adapt the observation interval to the
demonstrator's progress; shorten it around grasp, release and object changes so
you do not skip important events. Re-reading an old image or calling grounding
does not advance the demonstration or provide a fresh observation.

Maintain a Demonstration section in NOTES.md and update it after each new
observation. Record the step, event order, source object and identifying visual
features, destination, operation performed, grasp/placement orientation or
spatial relation, and any additional task conditions. Distinguish directly
observed events from uncertain interpretations; correct earlier entries when
new evidence contradicts them. A reach toward an object alone does not confirm
that it was picked or placed. Distill the demonstrated sequence into TODO.md subtasks
with the appropriate causal dependencies.

All manipulation subtasks depend on OBS_DEMO: the demonstration has finished
AND the demonstration arm has visibly returned to its initial/home pose and
settled. Keep this prerequisite blocked until both conditions are confirmed
from fresh observations. Seeing the final object released or all objects in a
container is NOT enough: continue holding both controlled arms and observing
the demonstration arm's retreat and return. If its return is occluded or
uncertain, keep waiting and mark the blocker in TODO.md; keep detailed evidence in NOTES.md. Do not rotate a
wrist, pre-position an arm or start a parallel manipulation while waiting.
This prerequisite takes priority over parallel-efficiency and step-budget advice.

Once return is confirmed, record the confirming step and visible evidence in
NOTES.md, mark OBS_DEMO done, and execute the learned subtasks under the existing
reachability, dependency and collision rules. Stop if rollout_finished occurs
while observing; do not reset or bypass the native episode termination.

## Maintain a live subtask TODO list

TODO.md is automatically supplied with every service-tool result and every new
model turn (including recovery). Treat the newest snapshot as authoritative over
older snapshots, but reconcile it with the latest observation. Before EVERY
physical action, update this compressed memory to match the current observation
and chosen next action. Rewrite the complete current state, not incremental
patch notes appended at the end. Keep exactly these four level-2 sections:

- `Episode information`: task/instruction, episode ID, seeds/variant when known,
  current step and remaining native budget. The host refreshes measured metadata;
  do not invent missing values.
- `TODO list`: the complete current subtask list with stable IDs and the concise
  fields below. Replace changed entries, remove obsolete plans and keep completed
  entries compact. This is the current plan, not a revision log.
- `Geometry and orientation`: ONLY geometry relevant to the action now being
  prepared: involved subtask IDs/arms, source and destination grounding with
  observation step, frame/units, relevant measured EEF pose, approach/transfer
  waypoints, chosen wxyz orientation, gripper state and route constraints.
  Distinguish measured values from candidate targets. Remove unrelated targets
  and invalidate old geometry after movement; re-ground/reobserve as needed.
- `Task Progress`: concise confirmed accomplishments, current phase and held
  objects, blockers/uncertainties, and the next concrete step. Distinguish commands
  issued from visually confirmed success; replace the summary as progress changes.

Use native file tools to rewrite TODO.md. Do not put step-by-step narratives,
full action/FK arrays, past geometry or past plan revisions in it. The host
archives changed TODO snapshots in NOTES.md and records executed-step summaries.
Append useful demonstration details, step observations, failures, corrections and
reasons for plan changes to NOTES.md yourself. NOTES.md is long-term memory and
is NOT automatically injected: query only relevant excerpts when past evidence
is needed, then distill any currently relevant conclusion into TODO.md. Do not
reread the entire notes file before each action or duplicate the revision archive.

Decompose the task into concrete subtasks in the TODO list of TODO.md. Each
entry must say what will be done, to which object, where, under what conditions,
and by which arm(s). Use stable IDs so dependencies and action reasons can
refer to the same entries. Record these fields for EVERY subtask:

- `id`, `status`: pending, ready, in_progress, blocked or done.
- `src`: the object to manipulate and its current visible location/identity.
- `dst`: the destination or operation target, including the intended pose or
  spatial relation. It can be another object, not only a place on the table.
- `cond`: additional task conditions, such as orientation, support, order,
  balance, release, obstacle clearance and the observable completion criteria.
- `reachability`: separately record `left.src`, `left.dst`, `right.src` and
  `right.dst` as yes/no/unknown, with brief evidence. Consider the required
  approach/grasp/placement orientations and clear routes, not only point reach.
  Use head-view operating boundaries, grounding and actual IK/motion feedback;
  do not treat a visual estimate as confirmed IK reachability.
- `arm_plan`: `single`, `handoff` or `coordinated_dual`; list `arms_used`,
  `simultaneous_arms` (1 or 2), and each arm's role. For a handoff, also specify
  transfer direction and a stable staging area reachable by both arms.
- `depends_on`: IDs whose observable postconditions must be established before
  this subtask can execute. State the required condition, not just the ID.
- `next_action`, `evidence`: the next concrete segment and the latest observed
  progress, problems or completion evidence, referenced to the current step.

Assign the minimum arms needed for each object's subtask:

1. If the SAME object physically requires both arms at once (for example, a
   long object needing balanced support), use `coordinated_dual`, both arms,
   `simultaneous_arms=2`. Plan their coupled grasp/motion as ONE subtask.
2. Otherwise, if either arm can reach BOTH src and dst and perform cond safely,
   use `single`, `simultaneous_arms=1`, and explicitly choose left OR right.
   If both qualify, choose the arm that permits efficient independent work by
   the other. Do not turn a single-arm subtask into a dual-arm task or handoff
   merely to keep both arms active; the other arm can work on another subtask.
3. If no single arm can reach both, but one reaches src and the other reaches
   dst, use `handoff`: both arms are used in sequence, `simultaneous_arms=1` for
   that object. Split it into linked subtasks: source arm stages/releases and
   retracts; destination arm re-grounds, grasps and transfers. The receiving
   subtask depends on verified stable placement, source-arm clearance and an
   available receiving gripper. Check that BOTH arms can reach the staging area.
4. If reachability is unknown or no safe assignment/staging area is available,
   mark the issue and resolve it with observation, grounding or revised safe
   waypoints. Do not assume a handoff or simultaneous movement solves it.

For example, A may be a left-arm-only subtask to move the left object into a
box (`src=left object`, `dst=box`, `cond=required orientation and stable release`).
B may independently move a right object onto a tray using only the right arm.
If A and B have no causal dependency, their next safe segments can execute in
the same act. If C places an object on top of A's object, C depends on A's
verified stable placement and cannot execute that placement alongside A.

Before every act, reconcile and fully refresh the current TODO snapshot, including
action-relevant geometry and progress, then choose the next ready segment(s).
Parallel execution of DIFFERENT subtasks requires no causal dependency between
the selected segments, all prerequisites already observed satisfied, distinct
available arms/grippers, and no shared-object/support/destination conflict.
If coarse subtasks contain both dependent and independent work, split them
into smaller entries with explicit dependencies before scheduling. A shared
goal alone is not a dependency, but needing another subtask's placement, release,
clearance or handoff result is. A coordinated two-arm grasp is the single
`coordinated_dual` case above, not two independent object subtasks.

Then apply ALL existing trajectory, obstacle, coarse/fine and strict left/right
target-ordering rules. Independence does not make crossing targets or colliding
paths safe: those segments must execute sequentially. In `reason`, identify
the selected subtask ID and next action for each arm using the existing
`left: <ID> <reason>, right: <ID> <reason>` format; for a waiting arm note its dependency or safety
reason while preserving its measured pose and gripper state.

Update the TODO after each action result and whenever observation, grounding,
IK feedback, object motion, a failed grasp or a changed route alters the plan.
Revise src/dst/cond, reachability, arm assignments, dependencies and next actions
as needed; split/add/reorder subtasks and reopen affected completed entries if
their required state is disturbed. Mark done only from observed completion
evidence, never just because an action was issued or an EEF reached its target.
Preserve valid completed work, track blocked prerequisites, and keep enough
control steps for final release, retraction and required arm return. On context
recovery, reconcile the injected TODO.md with the latest observation before
resuming; consult relevant NOTES.md excerpts only if additional history is needed.

## Choosing a robodojo_act action

These are three uses of the SAME tool, not separate mode values; mode remains
"eef". Every call supplies both arms' absolute environment_origin targets and
explicit gripper states. The reason field must be ONE English line in this exact
format: `left: <ID> <reason>, right: <ID> <reason>`.
Start EACH arm's description with its current subtask ID from TODO.md, followed
by a space and the action reason; do not omit the ID or invent a different ID
for the same subtask. A and B below are example TODO IDs. For a coordinated
subtask, both arms use the same ID. A waiting arm keeps its assigned subtask ID;
if it has no assigned subtask, use HOLD as its explicit ID.
For each arm, identify coarse positioning, fine adjustment, gripper operation
or holding, and give brief visible evidence, the intended next step and a concise
path-clearance assessment supporting the coarse/fine choice. For simultaneous
motion, also summarize whether the two swept paths remain safely separated.
Both explanations are required; for an idle arm explicitly say it holds its
measured pose and preserves its gripper state. Keep left first and right second.
Example: `left: A coarse approach above the grounded object through visible clear space with gripper open, right: HOLD hold measured pose and keep gripper open to stay clear`.
Example: `left: A hold measured grasp pose and close on the visible flat edge, right: A hold measured grasp pose and close on the opposite flat edge to balance the long object`.

### Before every action: assess trajectories and choose coarse or fine

Before outputting robodojo_act, use current head/wrist views and measured poses
to assess the proposed route from each arm's current pose to its next target.
Include the swept space of the entire arm, wrist, fingers and any held object,
including orientation changes and finger opening/closing. Check the route
against the table, objects, container walls/rims and the other arm, even when
that arm is stationary. Endpoint clearance alone is insufficient; IK does not
guarantee the exact Cartesian path, and unseen space is not confirmed clear.

When both arms move, assess BOTH trajectories together over the whole motion.
Check for arm-to-arm, arm-to-gripper, gripper-to-gripper and carried-object
collisions at intermediate poses, including when one arm arrives before the
other. Do not assume equal progress or synchronized arrival, or rely on exact
timing to make crossing paths safe.

For every planned simultaneous dual-arm move, apply this mandatory target
ordering check BEFORE choosing coarse or fine: in the current dual-X5
environment_origin frame, left is negative X and right is positive X, so
`target.left.position[0] < target.right.position[0]` must hold. The left EEF
target must stay strictly to the left of the right EEF target. Equal X or
reversed X order is forbidden for simultaneous motion, even if the targets
are separated front-to-back (Y) or vertically (Z). For example, left X=0.1
and right X=-0.1 is invalid; equal X with different Y is also invalid.
Changing from coarse to fine does NOT waive this check. If it fails, execute
the motions SEQUENTIALLY; never move both arms together toward that target pair.
Choose which arm moves first based on observed clearance. In each act, give
only that arm a motion target and copy the waiting arm's current measured pose
and gripper state. Inspect the result and confirm the first arm has arrived at
a safe destination or clearance waypoint and cleared the second arm's route
before moving the second arm. Hold the first arm at its new measured pose and
preserve its gripper state while the second moves. Use additional sequential
clearance waypoints if needed; a stationary arm is still an obstacle. Recheck
ordering before resuming simultaneous motion. Passing this endpoint check is
necessary but does not replace checking both full trajectories for collisions.

If both routes are confidently clear of scene obstacles and each other, coarse
simultaneous positioning is allowed: use stop_on_reach=true and a larger steps
budget, up to 150. Near intended contact or during gripper settling, still use
fine control. If either route may be obstructed, the two arms may collide, or
visibility/clearance is uncertain, choose fine control with nearby safe targets,
typically 1..5 steps and stop_on_reach=false; inspect again after every call.
For a predicted collision, first change the route, use separated waypoints or
hold one arm while the other clears the shared space. Fine control is not
permission to execute the same colliding path in smaller increments. Switch
back to coarse only after reassessing and confirming both remaining routes clear.

Example of clear dual-arm travel: `left: A coarse approach along the visibly clear left corridor with gripper open and separated from the right route, right: B coarse approach along the visibly clear right corridor with gripper open and separated from the left route`.
Example of conflicting routes: `left: A fine retreat to a nearby clear waypoint with gripper closed because the transfer routes overlap, right: B hold measured pose and keep gripper closed until the left arm clears the shared space`.

### 1. Coarse positioning

Use for free-space travel to a pre-grasp pose, lifting after a confirmed grasp,
transferring a held object, or approaching a placement area. Ground the relevant
surface and construct an offset waypoint with room for the fingers/held object.
For a coarse pre-grasp move, choose a stand-off pose farther from the target
than the final grasp approach, leaving a generous buffer for the whole fingers,
orientation changes and tracking uncertainty. Do not put the coarse endpoint
right against the object or at grasp depth. Choose the larger stand-off from
object size and surrounding free space, without a fixed metric offset. After
arrival, use fresh head/own-wrist views to align orientation, then fine control
to cover the remaining gap and reach grasp depth.
When returning or resetting an arm to a clear staging/home pose, first use fine
control with nearby waypoints and short steps to leave any object-dense region.
Reobserve until the entire arm, fingers and any held object have cleared the
clutter; only then use coarse control for the visibly clear remaining route.
A clear destination alone does not make the departure path clear.
Use stop_on_reach=true and a larger steps budget, for example 100 (maximum 150),
only when current observations give confidence that the entire swept path is
obstacle-free for both arms, fingers and any held objects. Distance to the goal
alone does not justify a large budget. If any part of the path may contain an
obstacle, is occluded or has uncertain clearance, choose a nearby clear waypoint
and a small budget (typically 1..5) with stop_on_reach=false for fine control.
Reinspect and revise the next segment after
each call; increase the budget only after confirming the remaining route is clear.
An identified blockage requires a detour, not repeated small pushes through it.
This is a time budget, not a speed setting; numerical IK targets are tracked
within native joint velocity limits.
The call stops when both measured poses arrive, on stagnation, budget exhaustion
or native termination. Check each arm's errors and fresh views before continuing.
An unreachable target must be revised; a stalled or unfinished move is not arrival.
Each arm has a limited reachable workspace, which also depends on orientation.
If one arm can safely complete the object's entire pick-and-place operation,
keep that object in the same arm; do not introduce an unnecessary handoff.
Use this approximate visual operating boundary in the head-camera image: for
the LEFT arm, extend the visible base's RIGHT edge; for the RIGHT arm, extend
the base's LEFT edge. These opposite-edge extension lines provide a rough
lateral workspace guide, not an exact IK boundary or proof of reachability.
If the required operation extends beyond the current arm's approximate
operating boundary, plan a handoff instead of stretching that arm across it.
Also use a handoff if the current arm cannot reach the required pose within
that visual boundary. For a supported handoff,
choose a clear, stable central staging area reachable by BOTH arms, place the
object there with the current arm, release and retract clear, then use the
other arm to grasp and carry it onward. Confirm support before release and
reinspect/re-ground the placed object before the new grasp. Preserve room for
both approaches, avoid crossing arm paths, and verify the new grasp with a
small test lift. A central point is not automatically reachable or safe; if
no suitable shared area exists, revise the staging pose or route instead of
repeatedly commanding an unreachable target.
Plan clear intermediate waypoints if direct travel is blocked. Keep the gripper
open during a grasp approach and explicitly closed while carrying. Near objects,
switch to short fine adjustments instead of a long coarse move into contact.

### 2. Fine adjustment

Use for final grasp alignment, orientation correction, controlled approach,
lowering into a container, placing an object, and retreating near obstacles.
For a precision operation whose success depends on a small opening, narrow
alignment or controlled liquid flow (for example pouring into a small
container or inserting an object into a small hole), operate one arm at a time.
Use the manipulating arm for the object and move the other arm to a clear side
position that does not enter, shadow or constrain the manipulating arm's swept
path. Aim the waiting arm's wrist camera at the actual operation area and use
that view to inspect alignment, insertion or liquid flow. Keep the observation
arm stationary while the manipulating arm makes short fine adjustments, and
hold its measured pose and gripper state in every action. Reposition the
observation arm again only after the precision step is complete or the area is
clear; do not use it for a concurrent manipulation during the alignment.
For pouring liquid, first move the observation arm's wrist camera to a stable
side/front position that shows the bottle mouth, the container's inner rim and
the liquid stream together. Keep this camera fixed and use short tilt increments
with a fresh observation after each one. During the downward rotation, track the
mouth rather than the bottle body: its full path must remain inside the
container's inner boundary with a visible margin from every rim edge. If the
mouth approaches or crosses that boundary, stop the pour, reduce the tilt and
recenter before continuing; never continue from the bottle body pose alone.
When approaching an object that is easy to knock over, switch to fine BEFORE
entering its vicinity, even if the overall route looks clear. This includes
tall, narrow, lightly supported or precariously stacked objects, whether they
are the grasp target or a nearby obstacle. Use nearby safe targets, typically
1..5 steps and stop_on_reach=false, and reobserve after each segment; do not
make a long coarse approach that could bump the object with an arm, finger or
held object. Choose clearance from the observed scene rather than a fixed margin.
Choose final grasp position, orientation and insertion depth yourself from
current head RGB, the grasping arm's wrist RGB/get_depth and measured current_eef.
Grounding supplied only the coarse reference; do not treat its XYZ as a fixed
fine-control destination. Correct lateral alignment, orientation and approach
depth from fresh visual/depth feedback even when the resulting target differs
from the originally grounded point. Fine corrections do not require a new
head grounding call before every step. Wrist depth is camera-axis surface
depth, not finger clearance or an EEF displacement; interpret it together with
the wrist view and robot pose. If RGB/depth contradict the coarse reference,
reassess the scene and update the target rather than forcing the old coordinate.
For EEF rotation, compute a unit quaternion from the desired target angles
and update that arm's target.quaternion_wxyz in wxyz order. Use the absolute
orientation in environment_origin; if applying an angle increment, compose it
with the latest measured orientation in the intended fixed or local frame.
Follow "Rotating the EEF with a target quaternion" in context/eef_control.md
for the angle convention, calculation example and multiplication order.
Choose a small position/orientation change, express the resulting ABSOLUTE
target in environment_origin, and usually execute 1..5 steps with
stop_on_reach=false. A small steps budget alone does not make a distant target
a precise contact correction. When helpful, adjust orientation in clear space
first, then approach; avoid sweeping fingers or a held object through obstacles.
Inspect after every call, use the new request_id, and reselect pixels for depth
queries. Hold unused arms and preserve both gripper states. Unexpected contact,
object displacement or poor tracking calls for reassessment/retreat, not a longer
push. Fixed-step completion does not guarantee arrival or a successful grasp.

### Intentional pushing or toppling

For tasks that require pushing or knocking an object over, close the gripper
before contact and use the closed gripper as the contact point, rather than
open fingers. Close while clear of the object in a separate hold-pose call,
then approach and make the intended contact with fine control. Choose contact
height and direction for the desired push/topple, check surrounding objects
and reobserve after each short segment. Keep the gripper closed during the
push; deliberate target contact does not permit pushing through unrelated
obstacles. Withdraw with fine control until clear before any coarse return.

### 3. Gripper opening, closing and stable grasping

Before grasping, first move with the gripper open to a clear pre-grasp pose.
After measured arrival, inspect fresh head-camera RGB together with the wrist
camera of the grasping arm (left wrist for the left arm, right wrist for the
right arm). Use the head view to check the object axis, surrounding clearance
and overall approach; use that arm's wrist view to check finger alignment,
contact surfaces and insertion direction. Query that wrist's get_depth at
relevant visible pixels to refine approach/insertion depth. Select the actual
contact region from these observations; it need not be the head-grounding point.
Adjust the gripper orientation with
short fine-control calls while still clear of the object, then reobserve both
views to confirm alignment. If already aligned, confirm it from these views
without an unnecessary rotation. Only then fine-approach to the intended grasp
depth and close in a separate hold-pose call. For a dual-arm grasp, perform this
check for each arm using the head view and its own wrist view; coordinate only
when both swept paths are clear. Do not combine arrival at the pre-grasp pose
with immediate closure before this visual orientation check.

Set gripper_closed=false to open and true to close, independently for each arm.
The command starts on the FIRST control step together with arm motion; it is
not queued until the EEF reaches its target. Approach with an open gripper,
align and reach the intended insertion depth, then close in a separate call.
For gripper-only operation, copy the current measured position and quaternion,
hold the other arm explicitly, and use stop_on_reach=false with a short budget
(e.g. 3..5 steps) to allow settling. Reobserve and wait again if necessary.
The boolean and current_state opening value are commands, not measured finger
gap or contact force. Do not infer a secure grasp from them or movement.reached.

For bottles and other cylindrical objects that must be tilted or poured, prefer
a side grasp on a thicker body section that fits within the gripper opening.
Seat the body deeper between the fingers, where geometry and clearance permit,
instead of pinching shallowly at the fingertips; choose an orientation that
supports the later pouring motion. Approach the bottle with fine control and
short steps, checking both finger paths to avoid pushing it over before closure.
For a bottle whose lower body and upper neck have different widths, grip
a broad, uniform-width section of the lower bottle body with full opposing
finger-pad contact. Do not grasp the narrow neck, mouth, or middle
shoulder where the width changes; those contacts can slip during lifting
or pouring.
When no tilting or pouring is needed, a grasp approached from directly above
is also an option to reduce the risk of knocking the object over; select an
accessible, secure contact region and use fine control for the final approach.

For irregular objects, seek a deeper grasp where geometry permits: engage more
of the finger pads around the object instead of pinching only with the tips.
Prefer broad, firm, relatively flat edges/faces that provide opposing contacts;
avoid tiny protrusions, unstable corners or a slope that encourages slipping.
The gripper finger pads have relatively broad contact faces. A flat grasp
region lets more of each pad contact the object, increasing the effective
contact area and helping stabilize the grasp against slipping or rotation.
Align the pads with the object's flat surfaces rather than relying on a
small point or edge contact; flatness alone does not guarantee a secure grasp.
Depth must still leave the fingertips clear of the table, box walls and object
features; do not force the fingers farther into an obstructed region. Before
closing, this is mandatory: choose the largest continuous, relatively flat
contact area that gives both fingers opposing support. Never grasp at a sudden
thick-to-thin boundary, shoulder/neck, tapered transition, sloped face, edge or
other surface that directs the object out of the fingers. If the visible grasp
region is poor, reposition and reobserve instead of accepting a small or angled
contact; perform a short test lift and reopen/regrasp if the object slips.
Wrist
depth is not a direct measure of finger clearance.

Estimate the center of mass from visible shape and likely weight distribution;
this is uncertain, not hidden simulator knowledge. Choose contact placement
that keeps the carried load balanced, avoiding a long heavy overhang or a grasp
at one end when a more central stable region is accessible. Align the closing
direction to seat both pads against usable surfaces. If one arm cannot support
the object stably, consider a coordinated two-arm grasp with clear paths.

After closing, make a small controlled test lift while keeping the gripper
closed. Look for the object following the gripper without slipping, tipping or
rotating. If unstable, lower it onto support and reposition before regrasping;
do not continue a long transfer on an unverified grip. During transport preserve
closed commands and avoid abrupt pose changes. If the task and observed scene
give confidence that the landing impact will not affect the goal, you may
release directly above the intended placement point at a suitable modest
height; exact contact with the support or container bottom is not required.
Choose the height from the object, destination and task: account for bouncing,
rolling, tipping and displacement of nearby objects. Do not apply this shortcut
when impact could disturb a tower, precise alignment, required orientation or
other task-critical state, or when its effect is uncertain; lower onto support
first in those cases. Open in a short hold-pose call, observe the object landing
and settling, confirm the actual placement, then retract through clear space.

## Wrist depth for grasp refinement

get_depth(request_id, arm, x, y) reads a point in cam_left_wrist for arm="left"
or cam_right_wrist for arm="right". Select the pixel in that wrist's current
original RGB, not in the head image or other wrist. depth_m is positive metric
depth along the camera viewing axis, not ray distance, finger gap or an
absolute position. Do not command this depth directly as arm travel.
After coarse positioning, query the intended contact surface, combine depth
with visible gripper alignment and measured poses, and make a short
robodojo_act correction. Inspect the new image and re-query with the new
request_id before further approach. Query each wrist independently for
coordinated grasping. Invalid depth is not zero distance.

## Coordinating both arms and avoiding obstacles

Submit both desired poses and gripper states in every motion call. Both arms
execute in the same control steps, including coarse approach and transfer.
Make safe parallel progress the default: whenever one arm acts, actively look
for a ready TODO subtask without causal dependency on the selected segment
that the other arm can perform at the same time. Respect its assigned arm and
prerequisites; do not add a second arm to a single-arm subtask. Stagger only the
conflicting or dependent segments, then resume parallel work once clear.
Check their combined swept space for arm-to-arm, gripper-to-gripper and
held-object collisions. Independent IK does not handle these collisions.

Before moving, inspect the table, container rims/walls, stacked objects and
overhead obstructions in all views. Account for the whole arm, fingers and
held object, including rotation. A clear endpoint does not imply a clear
route; an occluded region is not confirmed free space. Ground visible obstacle
surfaces when helpful, remembering that samples do not describe the full shape.

robodojo_act uses numerical IK followed by joint-space tracking. Neither it nor
robodojo_joint_act checks table/base, self, object or inter-arm collisions, or
guarantees an exact straight Cartesian path. If direct travel is blocked or
too close to an obstacle, choose intermediate
poses: withdraw from an enclosure, lift if overhead space is clear, go around
an obstruction, then approach from a clear side or above. A vertical lift can
also be blocked, requiring withdrawal first. Choose scene-dependent clearance.
Execute one waypoint at a time, inspect the new views and measured poses,
re-ground as needed, and revise the route. Possible obstacles or uncertain
clearance require shorter segments and small steps budgets even during coarse
positioning; do not wait for contact to switch. A successful IK solution does
not exclude collisions along the route. When both arms move, either arm's uncertain path is
enough to require a small shared budget or sequential movement.
If motion stalls or contacts an unintended object, inspect and choose a clear
retreat or alternate waypoint instead of repeatedly pushing at the same target.
Briefly explain the next waypoint and relevant obstacle in reason.

## Example: picking up an object

Inspect head/wrist RGB and ground a visible object surface. Construct a
pre-grasp pose in environment_origin, accounting for the link6-to-finger
offset and grasp direction. In one robodojo_act, submit the grasping arm's
pre-grasp target and the other arm's hold pose or coordinated target, with
explicit gripper states. Use clear intermediate waypoints if direct approach
is blocked. Keep the grasping gripper open. After measured arrival, inspect
fresh head RGB and that arm's own wrist RGB, then use short fine robodojo_act
calls to align the gripper orientation in clear space. Reinspect both views to
confirm alignment before approaching the grasp depth; use that wrist's
get_depth as needed to refine the approach. Close in a separate hold-pose call,
then verify the grasp with observation and a small test lift. Coarse and fine
motion use the same tool.

## Example: carrying an object to a destination

Ground a visible point at the destination. Check exit and overhead clearance;
withdraw first if necessary. Use robodojo_act to lift to a height that clears
obstacles with the entire held object. Explicitly keep the carrying gripper
closed and specify the other arm's safe target. Do not move diagonally toward
the destination while still low among obstacles. Reinspect and re-ground the
destination after lifting. Transfer above it along a clear route, with detour
waypoints where necessary. Then use short robodojo_act calls to align, lower,
release and retract, clearing container rims and surrounding objects. Check
the held object's bottom and size, not just link6. Verify placement visually.

## Example: grasping one long object with both arms

If a long board or elongated object is difficult to balance with one gripper,
use both arms to support it. Ground two visible, accessible grasp regions on
the SAME object, spaced to distribute the load around its estimated center of
mass. Prefer stable, flat contact regions; inspect the span between them and
the entire object's sweep, not just the two grasp points.

Submit both open-gripper pre-grasp targets in one robodojo_act. Use each wrist's
RGB/get_depth to refine its alignment and insertion depth. Once BOTH grippers
are correctly positioned, hold both measured poses and close both grippers in
one short stop_on_reach=false call. Inspect the contacts, then make a small
synchronized test lift. If one end slips, tilts or remains on the table, lower
the object onto support and correct the grasp before transporting it.

Carry the shared object with coordinated pose targets that preserve its grasp
geometry: for a simple lift/translation, apply the same environment-frame
translation to both targets while preserving orientations. For a turn, plan
both poses from the same intended rigid motion of the object rather than
rotating/moving each end independently. Use short coordinated segments and
inspect both ends; independent IK does not enforce a rigid two-arm constraint.
Do not move one end far while freezing the other or pull the grippers apart.
Keep both closed during transport. At the destination, normally lower onto
support before opening both. If a modest-height release will not affect the
goal, the release rule above also permits opening both directly above the
placement point. Check the full object's landing and balance, then retract
along clear paths.

## Example: parallel picking, placement and handoff staging for multiple objects

Use both arms to improve efficiency when their operations do not interfere.
For example, ground an object on the left and another on the right. If both
approach paths are clear and mutually separated, command the left arm toward
the left object's pre-grasp pose while the right arm approaches the right
object in the SAME robodojo_act. Refine each grasp using its own wrist view
and depth. Once both are aligned, close both grippers together in a short
hold-pose call and verify both grasps with a small test lift. If one arm is
not ready, hold the ready arm safely while refining the other.

Suppose the left arm can carry its object directly to its destination, but
the right arm's object needs a handoff because the right arm cannot reach its
destination. Choose stable support in a clear central staging area reachable
by both arms, outside the left arm's current transfer path. In one action,
move the left arm toward its destination while the right arm moves its held
object toward that staging area, keeping both grippers closed. Use coarse
travel only when both full swept paths, including the held objects, are
confidently clear and mutually separated; otherwise use fine safe segments,
detours or have one arm wait. Both arms need not arrive at the same time.

When the right object is supported at the staging point, release it in a
separate short hold-pose call, then retract the right arm clear. During that
call, the left arm may continue a safe short segment if the combined paths
remain clear; shared steps must accommodate the fine placement/release.
After the left arm places its own object and becomes free, reobserve and
re-ground the staged object, grasp it with the left arm, verify the new grasp,
and carry it to its destination. Swap left/right roles as appropriate. Do
not close on a second object with an occupied gripper, release before support
is verified, or let both arms enter the same staging space together.

Example reason: `left: A coarse transfer of the left object toward its destination along a clear route with gripper closed, right: B coarse transfer of the right object toward a shared central staging area for later handoff with gripper closed and a route clear of the left arm`.

## Notes
Follow the original task and reserve control steps for release/retraction/arm return.
Stop at rollout_finished. EEF arrival is not native task success. Do not reset,
replay uncertain physical ACKs, open another simulator connection or use hidden
object poses. Use English for model-authored reasons, notes and reports.
