This is a grill session on developing and establishing rules of lapacke translation to rust.

Background is that, for BLAS devices, we need something similar to lapacke (likewise scipy layer, see source code `/home/a/Git-Others/scipy/scipy/linalg/src/_linalg_*.hh`, but we stick more on similarity of lapacke), to unify row/col-major implementations, and make proper work scratch allocation.

Current codebase have already some important lapacke translation functions.

However, we are short of that
- We do not have a guideline of translation. This guideline should be detailed and covering virtually most situations that we encounters. In another word, given this guideline and current code base, a subagent that without other prompts should translate lapacke to rust code correctly; and different subagents should behave samely that better avoid evident random behavior.
- We are short on several functions that we do not follow lapacke closely, especially current svd function workspace allocation is not strictly following lapacke. Current codebase is also not perfect and you should try edit and figure out the best way to implement lapacke translation.
- Note we may encounter three kind of cases: f-prefer/c-prefer/fully-non-contiguous. Note f-contig/c-contig is special case of f-prefer/c-prefer that their memory is contiguous and leading dimension size is equal to the number of rows/columns (f-prefer/c-prefer does not ensure this equality).

I'm also not sure where to put this guideline is better, the skills, ADRs, dev-docs, or somewhere else.

You are suggested to read some codebase of this project, scipy (~/Git-Others/), and lapack (~/Git-Others/lapack/). You can make some comprehension in subagents.
After subagents converges, you then propose questions. You are also suggested write questions in rstsr-improve-trajectory, follow that directory's CLAUDE.md.
