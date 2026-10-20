# Hybrid task planning and execution

pi05 consumes the original task instruction, three current RGB frames and robot
state. The TODO plan is an Agent-side supervision record; it is not secretly
injected into the policy prompt. Match the policy's inferred subtask to visual
evidence and adapt the plan if an alternative order is valid. Correct genuine
execution/goal errors with the recorded direct-control tools.

Use context/direct_control.md for TODO decomposition, demonstration waiting,
reachability, geometry, collision and manipulation guidance. pi05_execute is
the additional joint-prefix motion tool; robodojo_act remains the absolute EEF
correction tool. robodojo_joint_act exposes absolute/delta qpos recovery with
a motion-free FK preview. All executed motions spend the same finite native control-step budget.

For precision operations such as pouring into a small opening or inserting into
a small hole, use one manipulating arm at a time. Assign the other arm a
single-arm HOLD/OBSERVE role, park it to the side outside the active arm's
swept path, and aim its wrist camera at the operation area. Keep the observation
arm fixed while using its wrist view to inspect alignment or liquid flow between
short prefixes. Do not authorize concurrent manipulation by that arm; reposition
it only after the precision operation is complete or the area is clear.
For pouring liquid, move the observation wrist camera to a stable side/front
view that keeps the bottle mouth, the container's inner rim and the stream in
one image. Use short downward rotations and inspect after each prefix. Keep the
bottle mouth inside the container's inner boundary with visible margin from the
rim throughout the rotation; if it nears or leaves the boundary, stop, reduce
tilt and recenter before authorizing another prefix.

For every grasp, require a broad, continuous, relatively flat contact area with
both fingers opposing the object. Avoid thick-to-thin boundaries, shoulders or
necks, tapered transitions, sloped faces and edges because they promote sliding
or rotation out of the gripper. If the contact is not stable, reposition and
reobserve before closure, then verify with a short test lift.
For a bottle whose lower body and upper neck have different widths, grip
a broad, uniform-width section of the lower bottle body with full opposing
finger-pad contact. Do not grasp the narrow neck, mouth, or middle
shoulder where the width changes; those contacts can slip during lifting
or pouring.

For model errors, reconcile the injected TODO.md, plan.json and the latest
observation; query NOTES.md excerpts only if historical evidence is needed. A new
model thread continues this same episode. Never repeat completed physical
actions or reuse discarded predictions. Keep requests small after a 413 error.
