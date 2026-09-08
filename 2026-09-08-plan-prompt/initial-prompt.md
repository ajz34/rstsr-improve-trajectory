You are going to grill me on the following task, and write initial plan (name as `260908-plan-*.md`) on the current task.

You are going to make a plan on possible efficiency improvement on rstsr (386948be819baa334b8da02232f3a1944e5447d5).

## Scope
- rstsr-core and rstsr-common, serial part (DeviceCpuSerial) and general parallel part (use DeviceFaer, but not touch on Faer's own implementation, specifically gemm/matmul).
- As efficiency improvement, you are always going to test on release mode.

## Details
- You are allowed to use simple fixed-aray types to improve efficiency (using lightweight-simd), but (1) this may need data type dispatch, which increase code complexity / compile time / binary size, so you are going to gate this by cargo feature like dim-dispatch (feature name `dispatch_simd`).  (2) crate lightweight-simd is actually fixed array, not true simd. So you need to test efficiency with both target-cpu=native (with approximately v4 features and AVX-512F) and non target-cpu optimized (without FMA/AVX features). (3) In many cases, we are not going to make full optimize, and many cases some simple fixed-size batches and fixed lane size 8/or/16 works well to common types, and not really need lightweight-simd, even in target-cpu=native. So using `dispatch_simd` may actually be overkill in many cases, but anyway for testing and efficiency check you can try simd implementation but not propose git diff if efficiency does not substentially improve.
- Some non-trivial optimization can occur on data transpose (probably at assign trait implementations, note tensor transpose only changes layouts), vecdot (numpy einsum utilizes simd, and common cases of vecdot can be represented as numpy einsum and therefore have room for optimization), reduce functions, etc.
- Though this is a crate that handles arbitary dimensionalities, in some cases common dimensions (like 2-dim data transpose, 2-dim sum axis 0 or axis 1, or the problems that can are of same structure to them) can be optimized. You can use cargo feature name `dispatch_`
- Do not fully utilize current device's L1/2/3 size. You are assumed to have no more than 32kB L1 and 256kB L2 and not truely optimize L3, if cache-size aware optimization is really really required. Cache-size aware optimization may be overkill in some tasks.
- For efficiency, you are suggested to compare to this device's peak theoretical/real efficiency, and use profiling tools to check pitfulls. Deep dig is better than mindlessly finding. We can be patient on this task, and try to make efficiency substentially improving.

## Requirements
- You are going to allowed to temporarily change everything in `../`, and you can read agent instructions there to gain thoughts. However, you are not going to commit anything there.
- You can generate multiple directories in rstsr-improve-trajectory, write test and bench code/scripts in each directory (make every directory to be independent, not related). Each directory can contain a report, a note on how to reproduce benchmark, a proposed git diff to rstsr for efficiency update (if efficiency really improves, otherwise you can also honestly writes the report and not propose a diff and that kind of report is also valuable).
- If necessary, you can also propose git diff to rstsr-agents and rstsr-book. Also stay these diffs in separated directories, not apply them.
- I will decide whether to put the diffs into rstsr. You just propose your edits.

## Suggestions
- You are going to use up to 3 agents: main agent, code agent, review agent.
- For main agent, you do not do any true work on code. You split and assign different tasks to code agent and review agent. Summarized results (not too much details) are merged back to main agent session by review agent.
- Code agent will do everything that writes: planning on its specific task, code changes, running benchmarks, summarize, ...
- Review agent will critically check code agents' works on some milestones, like check planning does not deviates too much on scope, general review on code changes, see if benchmarks are conducted with true effect and not cheating. Review agent will also perform or check summarize results, and handle summarize to main agent.