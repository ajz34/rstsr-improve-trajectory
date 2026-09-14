You are going to thoroughly check some sort of soundness in rstsr, a numpy-like crate.
The soundness here means 1) theoretically correctness (not record in rstsr repo but need report in this repo); 2) unsafe soundness (should directly write reason in simple 1-3 lines at rstsr repo, if need expanded discussion report in this repo); 3) check column-major correctness (may need to write new integration tests, note follow existing integration test naming rules) ; 4) easy-to-find efficiency improvements (this is not the most emphesis).
Note you are not going to touch blas-devices/tblis/sci in this task. Most focus on rstsr-core/rstsr-common and related crates.
If you are able, you can should check device faer, including its linalg functionalities. 

- You are going to organize yourself. I will not intervene you. But note you are suggested not to spawn more than 4 agents.
- You are going to tackle down this task into subtasks, and make each subtask to be a directory. You can refer to how previous `2026-09-08-plan-prompt` do.
- You are allowed to temporarily change rstsr projects for building and testing. You can spawn workspaces for this. However, you are not going to directly commit to them. For each subtask, you are going to make diff files in each subtask.
- If you feel you need new ADRs, you can also report and propose diff files to rstsr-book. But only add ADRs when very necessary.
- You can also propose fixes on existing codebase. It can probably that some subtasks requires another subtasks dependency and I think you can get resolved of them.
- If possible, you are going to check non-common shapes (0/1) and strides (non contiguous strides, partially contiguous strides, ...). Tests can also be introduced into code changes (diff files).
- You are free to git commit to this repository (rstsr-improve-trajectory) in any time, but not rstsr or rstsr-book or other repos.
