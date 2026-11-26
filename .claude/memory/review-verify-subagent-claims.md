---
name: review-verify-subagent-claims
 description: In multi-agent reviews, re-run one command per hard claim before reporting - sub-agent and summary statements were wrong in both directions (commit trailer list, cudarc gating)
metadata:
  type: feedback
---

From the 2026-11-26 rstsr-cuda matmul review:

- A sub-agent reported 8 commits with misplaced `Co-authored-by` trailers; `git log --format=%B | git
  interpret-trailers --parse` per commit showed 6 (two listed commits were correct).
- My own compacted summary claimed cudarc's cuBLAS bindings are not version-gated; reading the `cfg` block
  above each symbol showed `cublasSetMathMode` from `cuda-12090` and the FP64 emulation constant from
  `cuda-13000`. An older memory note had it right.
- Code-reading findings ("definite", "by inspection") from an auditor all held up under a probe, but the
  probe itself produced false alarms four times (wrong reference shapes, wrong tolerance, flags on the wrong
  device value). Classify probe output: FAIL lines of a forced single driver may be a legal refusal.
- Practice: for every claim that names a commit, a line, a cfg or a number, run the one command that shows it
  and keep the output (`results/`), then write the report from that output.
