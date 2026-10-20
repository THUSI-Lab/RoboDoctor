Review each fresh pi05 proposal against current RGB, robot-only FK, grounding
and the current subtask plan. Allow a short safe prefix if it advances a ready
subtask; valid independent reordering requires updating the plan, not takeover.
Intervene on actual failure, incorrect goals, unsafe paths or unmet dependencies.
Unknown intent is not confirmed safe intent. Execution always invalidates the
remaining proposal. Reobserve, update the plan and infer afresh after correction.
