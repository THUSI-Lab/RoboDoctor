# Coordinate and action contract

Dual ARX X5: six arm joints and a gripper per side. All control poses target
link6, not the finger pads. Positions are metres; orientations are unit
quaternions in wxyz order. The gripper is commanded separately.

grounding(request_id, x, y) selects a native head RGB pixel and returns a surface
position in environment_origin. Both arms use this same frame, matching
current_eef and all robodojo_act targets. No arm or base-frame conversion is
needed. world_axes_in_env maps world-direction vectors into this frame.
Use this surface point only as a coarse localization reference for a clear
pre-grasp/transfer region, never as a mandatory contact point or final link6
position. At pre-grasp, choose/refine the actual grasp position, orientation and
insertion depth from fresh head RGB, the corresponding wrist RGB/get_depth,
and measured current_eef. These fine targets may differ from the grounded XYZ;
they do not require another grounding call for every small correction.

robodojo_act combines coarse absolute numerical IK movement
and fine EEF/gripper control. Submit:
response={request_id, mode:"eef", steps:1..150, stop_on_reach:true/false, reason,
target:{left:{position,quaternion_wxyz,gripper_closed},right:{...}}}.
reason must be one English line: "left: <ID> <reason>, right: <ID> <reason>".
Each description starts with the current TODO subtask ID; use the same ID for a
coordinated subtask and HOLD only when an idle arm has no assigned subtask.
Explain each arm's action type, visible evidence and purpose; explicitly describe
holding the measured pose and preserving gripper state for an idle arm. Both
parts are required, in left-then-right order, including for gripper-only calls.
Before emitting the action, assess both proposed trajectories from the current
measured poses, including the whole arms, finger motion/rotation and held objects.
Check scene obstacles and mutual collision risk throughout the motion, including
unequal arm progress and one arm arriving first. Use coarse simultaneous travel
only when both full swept paths are confidently clear and safely separated.
Simultaneous dual-arm moves also require strict EEF target ordering in the
current dual-X5 environment_origin frame: target.left.position[0] must be LESS
THAN target.right.position[0] (left is negative X, right is positive X).
Equal or reversed X order is forbidden even with front-to-back Y separation
or different heights. This applies to both coarse and fine simultaneous moves.
If the ordering fails, execute sequentially, never simultaneously, even in fine
control. Move one arm while copying the other's current measured pose and
gripper state. Confirm the first arm has reached a safe destination/clearance
waypoint and cleared the second route, then hold it at its new measured pose
while the second moves. Choose order and clearance waypoints from the scene;
the held arm still counts as an obstacle. Reducing steps does not make a
crossed simultaneous target pair valid. Correct target ordering alone does
not prove the intermediate arm, finger or held-object trajectories are clear.
Possible obstacles, overlapping routes or uncertain clearance require fine
control: nearby safe targets, typically 1..5 steps and stop_on_reach=false.
For predicted collisions, detour or hold one arm while the other clears the area;
do not execute a colliding route merely with fewer steps. Reassess after each
call. Include a brief clearance/risk conclusion and coarse/fine choice in each
arm's reason text, preserving the required left: <ID> <reason>, right: <ID> <reason> format.
For precision operations that depend on narrow alignment or controlled flow,
such as pouring into a small opening or inserting into a small hole, use one
manipulating arm only. Move the other arm to a clear side position outside the
manipulator's swept path and aim its wrist camera at the operation area. Use
that waiting arm's wrist view as an auxiliary inspection view while the
manipulating arm makes short fine corrections. Keep the inspection arm fixed
with its measured pose and gripper state in every action; it must not perform a
concurrent manipulation or obstruct the active arm. Reobserve after each small
step and move the inspection arm again only when the precision operation has
finished or the area is clear.
For pouring liquid, first move the inspection arm's wrist camera to a stable
side/front position that simultaneously shows the bottle mouth, the container's
inner rim and the stream. Keep that camera fixed and use short tilt increments,
reobserving after each increment. During downward rotation, the bottle mouth's
entire path must stay inside the container's inner boundary with a visible
margin from the rim. Track the mouth offset and rotation arc rather than the
bottle body; if the mouth nears or crosses the boundary, stop, reduce tilt and
recenter before continuing.
Always supply both arms, even for one-arm motion. To hold an arm, copy its current
measured pose and preserve its gripper state (current_state[6] for left, [13]
for right; command 0=closed, 1=open). Specify gripper_closed=true to close.
Gripper commands apply from the first control step, concurrently with arm
motion; they do not wait for pose arrival. Use an open-gripper approach and a
separate hold-pose closing call. For gripper settling use stop_on_reach=false
with a short steps budget; a reached EEF pose can otherwise end the call early.

## Rotating the EEF with a target quaternion

To rotate an EEF, determine the desired orientation angles, compute the unit
quaternion, and replace that arm's `target.quaternion_wxyz` in robodojo_act.
The target is an ABSOLUTE link6 orientation in environment_origin, in
`[w, x, y, z]` order; it is not an angle increment or a gripper-finger pose.
Do not put Euler angles into this field or adjust quaternion components by
adding angle values.

For desired absolute roll/pitch/yaw in degrees about fixed environment_origin
X/Y/Z axes, use the extrinsic XYZ convention, R = Rz(yaw) Ry(pitch) Rx(roll):

```python
from scipy.spatial.transform import Rotation

# Replace these example angles with the desired absolute link6 orientation.
roll_deg, pitch_deg, yaw_deg = 0.0, 0.0, 90.0
rotation = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True)
q_xyzw = rotation.as_quat()  # scipy returns x,y,z,w
q_wxyz = q_xyzw[[3, 0, 1, 2]].tolist()
# Set response["target"][arm]["quaternion_wxyz"] = q_wxyz
```

This example gives approximately `[0.70710678, 0, 0, 0.70710678]` as an
absolute target; it does NOT mean adding 90 degrees to the current orientation.
For an incremental rotation, start from the latest measured quaternion:
construct `current = Rotation.from_quat([x, y, z, w])`, and construct `delta`
from the desired angle(s). Use `delta * current` for a rotation about fixed
environment_origin axes, or `current * delta` for a rotation about the EEF's
local axes. Convert the resulting absolute rotation back to wxyz as above.
Always distinguish the rotation axis/frame and degrees versus radians.

For an orientation-only action, keep the arm's measured position and intended
gripper state, and explicitly hold the other arm unless it has a safe concurrent
subtask. Rotation about link6 moves the offset fingers and any held object:
inspect head/own-wrist views and swept clearance, use fine adjustments near
objects, and reobserve the measured orientation after each action. A requested
quaternion or successful IK solution alone does not prove actual arrival.

There is no per-call target translation or rotation cap. Both targets are
solved by this repository's joint-limit-bounded numerical IK BEFORE either arm
moves, seeded from its current measured joints. The URDF FK is checked against
measured link6 poses. Any IK failure returns without motion. Successful solutions
are tracked in joint space using native joint velocity limits; measured qpos is
read again after each actual control step. Each new act solves from the new
measured state. This is NOT collision planning: it does not check the table,
base, self-collision, task objects or the other arm, and does not guarantee a
straight Cartesian path. Choose clear intermediate poses from observations.
When blocked or stuck in a poor IK configuration, use the joint recovery tool
below rather than repeatedly submitting the unchanged EEF target.
Both arms have limited, orientation-dependent reach. If one arm can safely
complete the object's operation, keep using that arm without a handoff.
Approximate the operating boundary from the head-camera image: the LEFT arm
uses the extension of the visible base's RIGHT edge, and the RIGHT arm uses
the extension of the base's LEFT edge. These are rough visual boundaries,
not exact IK limits or guarantees of reachability. Operations beyond the
current arm's boundary require a handoff; do not stretch across it. If the
required pose is otherwise unreachable, also plan a handoff. Place the object
on stable support in a clear central area reachable by both arms, then switch
hands. Verify support,
release and retract the first arm clear; reobserve/re-ground, grasp with the
other arm and confirm with a small test lift before continuing. Check both
approach paths and avoid arm collisions. Do not assume the center is reachable
or safe, or repeatedly retry an unreachable target unchanged.
When both grippers hold ONE long object, coordinate targets to preserve the
shared grasp geometry and inspect for unequal tracking; independent IK does
not enforce a rigid grasp constraint. When holding TWO separate objects, both
may be approached and grasped together, left arm for a left-side object and
right arm for a right-side object, when the full paths do not interfere.
One arm may transfer its object toward its destination while the other stages
an object needing handoff on stable support in a clear central area reachable
by both arms, outside the first arm's transfer path. Keep carrying grippers
closed, release the staged object only after verifying support, and retract.
After the receiving arm places its own object and becomes free, re-ground and
grasp the staged object. If the paths conflict, use fine safe segments, detours
or sequential movement with the waiting arm's measured pose and gripper state
held explicitly. See the dual-arm examples in SKILL.md.

The episode's total control-step budget is finite. Prefer useful simultaneous
work whenever both routes and task dependencies allow it. Before every act,
look for a safe grasp, transfer, placement, staging, retreat or arm-return
segment for the other arm; do not leave it idle by default. Both arms share
each executed control step, so parallel progress can save steps, whereas simply
combining calls into a longer call does not. Serialize only conflicting or
dependent segments, preserve fine control where needed, and reserve enough
steps for release, retraction and required arm return.

steps is the maximum control-step budget for this call. Use stop_on_reach=true
for coarse travel: stop when BOTH measured link6 errors are <=5 mm and
<=0.03 rad, or on stagnation/budget/native termination. Use small steps (e.g.
1..5) and stop_on_reach=false for fine contact corrections and gripper settling.
This executes the selected budget unless the episode ends. A pose already at
target still executes at least one step to apply gripper commands. movement.arms
reports left/right errors; movement.reached describes EEF poses, NOT grasp or
finger-gap verification. Inspect images and native outcome separately.

Choose a large steps budget only when confident the full grasp-approach or
movement path is obstacle-free, including the swept space of both arms and
held objects. Possible obstacles, occlusion or uncertain clearance require
small budgets (typically 1..5), nearby clear waypoints and observation after
each segment, even for coarse positioning. A known blocked path requires a
detour; small steps are not permission to push through it. steps is shared by
both arms, so uncertainty on either route determines the cautious budget.

Keep coarse pre-grasp endpoints farther from the target with a generous,
scene-dependent buffer; use head/own-wrist views to align, then fine to close
the remaining gap. To reset/return an arm from clutter, fine-retreat until its
whole swept geometry clears the objects before using coarse through free space.
For deliberate pushing/toppling tasks, close the gripper before contact and
use it as the contact point; fine-approach, push and withdraw while checking
nearby objects. Do not use open fingers to push.
Coarse positioning uses grounded free-space waypoints and preserves the grasp
during lifting/transfer. Fine control uses small absolute pose corrections,
current wrist RGB/depth and frequent observation near contact. steps changes
execution duration, not target distance or tracking speed. When approaching
objects that are easy to knock over, use fine even if the route appears clear:
nearby safe targets, typically 1..5 steps and stop_on_reach=false, with fresh
observation after each segment. Apply this to both the target and nearby
unstable objects; account for the whole arm, fingers and carried object.
For gripper-only calls, keep both EEF targets at current measured poses and change only the
intended gripper boolean. Read SKILL.md's action guide for grasp-depth,
contact-surface, load-balance, test-lift and release rules.
Prefer flat grasp regions because the gripper pads have broad contact faces:
aligning them with flat object surfaces creates a larger effective contact
area and helps resist slipping or rotation. Avoid tip-only or point contacts and verify stability with a small test lift.
Before closing, require the largest continuous, relatively flat contact area
available, with both fingers opposing the object. Do not close on a sudden
thick-to-thin boundary, shoulder/neck, tapered transition, sloped face or edge;
these contacts make the object slide or rotate out of the gripper. If no stable
flat region is visible, reposition and reobserve before grasping, then reopen
and regrasp after any slip during the test lift.
For a bottle whose lower body and upper neck have different widths, grip
a broad, uniform-width section of the lower bottle body with full opposing
finger-pad contact. Do not grasp the narrow neck, mouth, or middle
shoulder where the width changes; those contacts can slip during lifting
or pouring.
If landing impact will not affect the task goal, release directly above the
placement point at a suitable modest height; exact support/bottom contact is
not required. Choose height based on the scene and object, verify landing and
settling, and use supported placement when impact matters or is uncertain.

get_depth(request_id, arm, x, y) reads the selected wrist's current native RGB
pixel and returns positive camera-axis depth_m (distance_to_image_plane), not
an environment-frame position or finger-to-object gap. Queries do not advance
simulation or consume movement decisions. After motion use the new request_id
and reselect pixels for grounding/depth as needed.

Numerical IK and direct qpos control do not check collisions or guarantee a
straight Cartesian path or a fixed point-clearance margin.
Inspect the swept space of both
arms, grippers and held objects; use clear intermediate absolute poses and
scene-dependent offsets. A reachable endpoint can still have a blocked path.

## Joint-space recovery with robodojo_joint_act

Use this tool when an EEF IK solve fails, movement.status is stalled, or fresh
RGB and measured qpos show that the arm is blocked despite repeated commands.
Inspect first: an IK solution is only a reachable kinematic endpoint, not proof
that the arm can travel there. Do not increase the command into a table/base
contact. Choose a small retreat or posture adjustment away from the obstruction,
then reobserve and retry robodojo_act so IK starts from the new configuration.
If another attempt makes no measurable progress, change the route or strategy;
do not keep forcing the same blocked joint/EEF target.

Arguments: observation_path and output_dir are the same fresh paths used by
robodojo_act; preview_only is required. response is:
{request_id, mode:"joint", coordinate_mode:"delta" or "absolute", steps:1..150,
 stop_on_reach:true/false, reason,
 target:{left:{qpos:[q1,q2,q3,q4,q5,q6],gripper_closed:boolean},right:{...}}}.
All six arm angles are radians in current_joints[arm].joint_names order; the
gripper is separate. Read current_joints for measured qpos, position limits,
velocity limits and gripper opening. Absolute specifies the joint target;
delta is added ONCE to the measured qpos at the start of the call. It is not
added again on each step. Keep a waiting arm at six zero deltas or its measured
absolute qpos; preserve both gripper states, especially while carrying.

First use preview_only=true: returns resolved joint_targets and target_eef in
environment_origin, without physical steps, new observations or consuming
output_dir. Inspect those endpoints and the whole swept arm/gripper route using
RGB. Preview does NOT collision-check. For simultaneous movements the previewed
left EEF X must be strictly less than right EEF X; if equal/crossed, move one arm
at a time. If clear, repeat the request with preview_only=false. The tool
revalidates against the current measured joints and rejects out-of-limit targets
without clipping. Begin recovery with one/few joints and usually 1..5 steps,
then inspect the new state. Joint control still obeys native velocity limits and
physical contacts; directly specifying qpos does not teleport through obstacles.

Example: after inspection shows a small positive change of left joint 2 retreats
from contact, use coordinate_mode="delta", left.qpos=[0,0.05,0,0,0,0],
right.qpos=[0,0,0,0,0,0], steps=3, stop_on_reach=false, preserving both grippers.
Preview, inspect, then execute. The sign/joint/0.05-rad size are illustrative;
choose them from the actual configuration and clearance, not blindly. Inspect
actual qpos before retrying EEF IK. For an absolute command, copy measured qpos
and modify only the intended joints instead of filling unused joints with zeros.

With stop_on_reach=true, arrival requires BOTH arms' largest measured joint
error <=0.01 rad; tracking stall also stops the call. false spends the chosen
budget unless native termination occurs. movement.arms includes joint_error_rad
and target_qpos as well as EEF errors. Arrival never confirms grasp success.
In hybrid mode, executing joint recovery obeys the same current-plan/intervene
requirements as EEF correction and invalidates the old pi05 prediction. Update
the plan and infer fresh after recovery; a preview alone does not discard it.
