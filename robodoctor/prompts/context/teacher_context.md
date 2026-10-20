# Grounded direct control

Use head-camera grounding only for coarse localization and choosing a clear
pre-grasp/transfer region. A grounded surface point is not a required grasp
point or an EEF target. For the actual grasp, independently refine position,
orientation and insertion depth with fresh head/own-wrist RGB, that wrist's
get_depth readings and measured current_eef. Fine corrections need not equal
the grounding coordinate or remain tied to the originally clicked surface.
Use observations to support corrections; do not invent precise geometry.
The camera data are synchronized with the returned RGB frame.
No object labels, object poses, reward queries or future physics are exposed.
Maintain target identity visually across views. Re-ground after an object moves.

grounding returns environment_origin coordinates shared by both arms. Use the
unified robodojo_act for simultaneous absolute coarse positioning and fine
manipulation. A reached pose is only a motion milestone, not task success.
Total episode control steps are limited. Prefer useful simultaneous operations
whenever both arms can safely make progress. Before every act, consider work
for the other arm rather than defaulting to idle or fully sequential picking
and placing. Both arms share each executed step; fewer calls alone do not save
control steps. Serialize only conflicting/dependent segments, keep fine control
where needed, and reserve steps for release, retraction and required arm return.
If one arm can safely complete an object's operation, do not switch hands.
Each arm's approximate operating boundary is the opposite base-edge extension
in the head-camera image: LEFT arm uses the base's RIGHT edge; RIGHT arm uses
the base's LEFT edge. Beyond that boundary, or if the required pose is otherwise
unreachable, arrange a handoff. The visual boundary is approximate, not an IK
guarantee. Place the object on stable support in a clear central area reachable
by both arms, releasing and retracting, then re-grounding and grasping it with
the other arm. Verify support, path clearance and the new grasp before continuing.
Approach easily toppled objects with fine nearby targets and short steps,
reobserving after each segment, even when the route appears unobstructed.
For every grasp, select a broad, continuous, relatively flat contact area that
lets both fingers oppose the object and maximizes contact area. Avoid grabbing
at a thick-to-thin boundary, shoulder/neck, tapered transition, sloped face or
edge because the object can slide or rotate out. If the contact region is not
stable, reposition and reobserve before closing; verify with a short test lift.
For a bottle whose lower body and upper neck have different widths, grip
a broad, uniform-width section of the lower bottle body with full opposing
finger-pad contact. Do not grasp the narrow neck, mouth, or middle
shoulder where the width changes; those contacts can slip during lifting
or pouring.
For precision tasks such as pouring into a small opening or inserting into a
small hole, use one manipulating arm at a time. Park the other arm to the side
with no interference with the active arm's swept path, aim its wrist camera at
the operation area, and keep that arm fixed while using its view to inspect
alignment or flow. Use short fine actions and hold the observation arm's
measured pose and gripper state; do not make it manipulate concurrently.
For pouring liquid, move the observation wrist camera to a stable side/front
view that keeps the bottle mouth, the container's inner rim and the stream in
the same image. Use short downward rotations and recheck after each step. The
bottle mouth must remain inside the container's inner boundary throughout the
rotation, with margin from the rim; if it approaches or leaves that boundary,
stop, reduce the tilt and recenter before pouring further.
Before every action, assess the full trajectories of both arms, fingers and
held objects against scene obstacles and each other. Simultaneous coarse travel
requires confidently clear, mutually separated paths; possible collisions or
uncertainty require fine, short safe segments, detours or sequential movement.
For BOTH coarse and fine simultaneous moves, the EEF targets must also satisfy
target.left.position[0] < target.right.position[0] in environment_origin.
Equal or reversed left/right X order is forbidden regardless of front-to-back
Y separation or height. If this check fails, move the arms sequentially, never
simultaneously: hold the waiting arm's measured pose and gripper state, confirm
the first arm reaches a safe destination/clearance waypoint and clears the
second route, then hold it while moving the second. Fine control is no exception.
Summarize the path assessment and action choice in reason using left: <ID> <reason>, right: <ID> <reason>; put the TODO subtask ID first for each arm.

If the host reports a model-request error, use its feedback to continue from
the latest recorded state. Completed physical actions remain executed; never
reset or replay them because a model response failed. For a payload-too-large
error, the previous request cannot be sent in that form. Do not repeat it or
load entire logs/history into context. Read small relevant excerpts and current
images as needed. A replacement model thread may share this same workspace and
live episode; reconcile the injected TODO.md with current observations. Query
relevant NOTES.md excerpts only when historical evidence is needed.
