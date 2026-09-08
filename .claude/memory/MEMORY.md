# Memory Index

- [rstsr & machine benchmark context](rstsr-bench-context.md) - machine specs (9950X3D, AVX-512/BF16), rstsr has NO bench harness yet, criterion declared but unused, nightly toolchain.
- [rstsr tensor extraction quirks](rstsr-tensor-extraction-quirks.md) - to_vec 1-D only (reshape(-1) first), asarray gives IxD, zeros is calloc-lazy, a.t() shape/broadcast rules at 386948be.
- [ZCode user-scope skills](zcode-user-scope-skills.md) - mattpocock-skills v1.2.3 copied into ~/.zcode/skills (25 skills); source of truth is Claude plugin cache; refresh procedure inside.
- [Env tools & references](env-tools-references.md) - conda env "torch" for numpy/torch refs, ~/Git-Others reference C sources (numpy/OpenBLAS/blis), perf installed / valgrind not.
