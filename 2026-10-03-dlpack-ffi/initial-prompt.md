# Initial prompt — rstsr-ffi dlpack family

- **Date**: 2026-10-03
- **Session**: Claude Code; invoked via skill `mattpocock-skills:grill-me` (a grilling/design session)
- **Target repos**: `rstsr-ffi` (new crate + generation scripts), `rstsr-agents` (new update skill)

## Verbatim prompt

> You are going to add a new rstsr-ffi family: dlpack.
>
> - This is totally different to lapack, though much simpler.
> - Something, especially enums, maybe of non_exhaustive, as my comprehension. You are able and should learn history versions of dlpack, to see in what way is better to not make the breaking while dlpack upgrading in minor versions.
> - Bindgen and auto-script techniques are still required for this task, and you should add a skill for checking and upgrading upon dlpack update like many of blas distributions.

## Follow-up instructions in the same session (verbatim)

> And also, you can alleviate main session pressure to generate subagents for comprehension and some proper tasks.

> Well you are going to re-propose all questions in one output, for me to easier review.

> Ah sorry for the changing of working status. You are going to put my initial prompt, and your round-1 questions into rstsr-improve-trajectory, in a proper directory you can create.
