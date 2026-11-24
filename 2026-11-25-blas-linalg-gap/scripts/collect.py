#!/usr/bin/env python3
"""Regenerate the inventory tables under `data/` for this directory.

Run from anywhere:

    python3 scripts/collect.py

Inputs:

* the `rstsr` and `rstsr-ffi` checkouts in the same pack root as this
  directory (`RSTSR_PACK` overrides the pack root);
* the installed `numpy` and `scipy` (the versions the committed tables were
  authored against are recorded in `../README.md`).

The two `*_coverage.tsv` tables (per-name status and rstsr equivalent) are
hand-authored; the script only validates that they cover every introspected
name.

The script only reads sources and introspects the two Python packages; it
never builds or runs rstsr.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent
PACK = Path(os.environ.get("RSTSR_PACK", TASK_DIR.parents[1]))
RSTSR = PACK / "rstsr"
FFI = PACK / "rstsr-ffi"
DATA = TASK_DIR / "data"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(name: str, header: list[str], rows: list[list[object]]) -> None:
    lines = ["\t".join(header)]
    lines += ["\t".join(str(cell) for cell in row) for row in rows]
    (DATA / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(rows)} rows")


# ---------------------------------------------------------------- linalg API
def linalg_entries() -> list[tuple[str, str]]:
    """(trait, function) pairs from the `duplicate_item` table in traits_def.rs."""
    text = read(RSTSR / "rstsr-linalg-traits/src/traits_def.rs")
    table = re.search(r"#\[duplicate_item\((.*?)\)\]", text, re.S).group(1)
    pairs = re.findall(r"\[\s*(\w+API)\s*\]\s*\[\s*(\w+)\s*\]\s*\[\s*\w+\s*\]", table)
    return [(trait, func) for trait, func in pairs]


def impl_forms(trait: str, backend: str) -> tuple[list[str], list[str]]:
    """Distinct `impl ... for <form>` heads and the files that declare them."""
    forms: list[str] = []
    files: list[str] = []
    root = RSTSR / "rstsr-linalg-traits/src" / ("blas_impl" if backend == "BLAS" else "faer_impl")
    for path in sorted(root.glob("*.rs")):
        text = re.sub(r"\s+", " ", read(path))
        for m in re.finditer(
            rf"impl\s*(?:<[^{{]*?>)?\s*{re.escape(trait)}<Device{backend}>\s*for\s+(.*?)\s*(?:where|\{{)",
            text,
        ):
            forms.append(m.group(1).strip())
            files.append(str(path.relative_to(RSTSR)))
    return forms, sorted(set(files))


def collect_linalg() -> None:
    rows = []
    for trait, func in linalg_entries():
        faer_forms, faer_files = impl_forms(trait, "Faer")
        blas_forms, blas_files = impl_forms(trait, "BLAS")
        rows.append(
            [
                func,
                trait,
                "yes" if faer_forms else "no",
                "yes" if blas_forms else "no",
                "; ".join(faer_files),
                "; ".join(blas_files),
                "; ".join(sorted(set(faer_forms))),
                "; ".join(sorted(set(blas_forms))),
            ]
        )
    write(
        "rstsr_linalg_entries.tsv",
        ["function", "trait", "faer", "blas", "faer_files", "blas_files", "faer_forms", "blas_forms"],
        rows,
    )


# ------------------------------------------------------------------- drivers
def collect_blas_drivers() -> None:
    rows = []
    root = RSTSR / "rstsr-blas-traits/src"
    for path in sorted(root.rglob("*.rs")):
        if path.parent.name not in ("blas2", "blas3"):
            continue
        text = read(path)
        for name in sorted(set(re.findall(r"unsafe fn (driver_\w+)", text))):
            rows.append([name, str(path.relative_to(RSTSR))])
    write("rstsr_blas_drivers.tsv", ["driver_entry", "file"], rows)


def collect_lapack_drivers() -> None:
    rows = []
    root = RSTSR / "rstsr-blas-traits/src"
    for path in sorted(root.glob("lapack_*/*.rs")):
        text = read(path)
        for name in sorted(set(re.findall(r"pub struct ([A-Z0-9]+)_<", text))):
            rows.append([name, str(path.relative_to(RSTSR))])
    write("rstsr_lapack_drivers.tsv", ["routine", "file"], rows)

    # translation files shipped once and symlinked into every device crate
    impl_root = RSTSR / "rstsr-blas-traits/driver_impl"
    files = sorted(str(p.relative_to(RSTSR)) for p in impl_root.rglob("*.rs") if p.is_file())
    write("rstsr_driver_impl_files.tsv", ["file"], [[f] for f in files])


# ----------------------------------------------------------------------- FFI
def collect_ffi() -> None:
    rows = []
    for crate in sorted(p.name for p in FFI.glob("rstsr-*-ffi")):
        src = FFI / crate / "src"
        for module in sorted(p.name for p in src.iterdir() if p.is_dir()):
            extern = src / module / "ffi_extern.rs"
            if not extern.exists():
                continue
            names = re.findall(r"pub (?:unsafe )?fn (\w+)", read(extern))
            rows.append([crate, module, len(set(names))])
    write("ffi_symbols.tsv", ["crate", "module", "symbols"], rows)


# ------------------------------------------------------------ NumPy / SciPy
def ffi_names(crate: str, module: str) -> set[str]:
    path = FFI / crate / "src" / module / "ffi_extern.rs"
    return set(re.findall(r"pub (?:unsafe )?fn (\w+)", read(path)))


def driver_families() -> tuple[set[str], set[str]]:
    blas_rows = [line.split("\t")[0] for line in read(DATA / "rstsr_blas_drivers.tsv").splitlines()[1:]]
    lapack_rows = [line.split("\t")[0] for line in read(DATA / "rstsr_lapack_drivers.tsv").splitlines()[1:]]
    def family(name: str) -> str:
        return name[len("driver_"):] if name.startswith("driver_") else name

    blas = {family(name) for name in blas_rows}
    # driver_syhemm covers both SYMM and HEMM
    if "syhemm" in blas:
        blas.discard("syhemm")
        blas |= {"symm", "hemm"}
    lapack = {family(name).lower() for name in lapack_rows}
    # One translated file can cover the real routine and its complex counterpart:
    # syev/heev, syevd/heevd, sygv/hegv, sygvd/hegvd, orgqr/ungqr, sysv/hesv.
    aliases = {
        "syev": "heev",
        "syevd": "heevd",
        "sygv": "hegv",
        "sygvd": "hegvd",
        "orgqr": "ungqr",
        "sysv": "hesv",
    }
    lapack |= {alias for base, alias in aliases.items() if base in lapack}
    return blas, lapack


def collect_python() -> None:
    import inspect

    import numpy
    import numpy.linalg as nl
    import scipy
    import scipy.linalg as sl
    import scipy.linalg.blas as blas
    import scipy.linalg.lapack as lapack

    versions = [
        ["numpy", numpy.__version__],
        ["scipy", scipy.__version__],
    ]
    write("python_versions.tsv", ["package", "version"], versions)

    np_names = sorted(n for n in dir(nl) if not n.startswith("_") and n != "test")
    write("numpy_linalg_names.tsv", ["name"], [[n] for n in np_names])

    rows = []
    for name in sorted(n for n in dir(sl) if not n.startswith("_") and n != "test"):
        obj = getattr(sl, name)
        kind = "module" if type(obj).__name__ == "module" else ("function" if callable(obj) else "other")
        rows.append([name, kind])
    write("scipy_linalg_names.tsv", ["name", "kind"], rows)

    funcs = sorted(n for n in dir(sl) if not n.startswith("_") and inspect.isfunction(getattr(sl, n)))
    write("scipy_linalg_functions.tsv", ["name"], [[n] for n in funcs])

    # scipy BLAS: routine names only (drop the dispatch helpers and modules)
    helpers = {"HAS_ILP64", "HAS_LP64", "functools", "np", "find_best_blas_type", "get_blas_funcs", "test"}
    routines = sorted(n for n in dir(blas) if not n.startswith("_") and n not in helpers)
    write("scipy_blas_routines.tsv", ["routine"], [[n] for n in routines])

    blas_ffi = ffi_names("rstsr-lapack-ffi", "blas")
    cblas_ffi = ffi_names("rstsr-lapack-ffi", "cblas")
    blas_drivers, _ = driver_families()
    blas_rows = []
    for name in routines:
        family = name[1:] if len(name) > 2 and name[0] in "sdcz" else name
        blas_rows.append(
            [
                name,
                family,
                "yes" if name + "_" in blas_ffi else "no",
                "yes" if "cblas_" + name in cblas_ffi else "no",
                "yes" if family in blas_drivers else "no",
            ]
        )
    write("scipy_blas_coverage.tsv", ["routine", "family", "ffi_blas", "ffi_cblas", "rstsr_driver"], blas_rows)

    # scipy LAPACK: wrappers grouped by routine family (prefix + _lwork stripped)
    helpers = {"HAS_ILP64", "HAS_LP64", "np", "functools", "get_lapack_funcs", "find_best_lapack_type", "test", "routine"}
    wrappers = sorted(n for n in dir(lapack) if not n.startswith("_") and n not in helpers)
    families: dict[str, list[str]] = {}
    for name in wrappers:
        if len(name) < 3 or name[0] not in "sdcz":
            families.setdefault(name, []).append(name)
            continue
        base = name[1:]
        if base.endswith("_lwork"):
            base = base[: -len("_lwork")] + "*_lwork"
        families.setdefault(base, []).append(name)
    write(
        "scipy_lapack_families.tsv",
        ["family", "n_wrappers", "wrappers"],
        [[fam, len(v), " ".join(v)] for fam, v in sorted(families.items())],
    )
    write("scipy_lapack_wrappers.tsv", ["wrapper"], [[n] for n in wrappers])

    lapack_ffi = ffi_names("rstsr-lapack-ffi", "lapack")
    _, lapack_drivers = driver_families()
    lapack_rows = []
    for fam, names in sorted(families.items()):
        if fam.endswith("*_lwork"):
            lapack_rows.append([fam, len(names), "n/a", "", "no"])
            continue
        missing = [n for n in names if n + "_" not in lapack_ffi]
        status = "yes" if not missing else ("no" if len(missing) == len(names) else "partial")
        lapack_rows.append([fam, len(names), status, " ".join(missing), "yes" if fam in lapack_drivers else "no"])
    write(
        "scipy_lapack_coverage.tsv",
        ["family", "n_wrappers", "ffi_lapack", "missing_in_ffi", "rstsr_driver"],
        lapack_rows,
    )


def validate_annotations() -> None:
    """Check the hand-written coverage tables against the introspected names."""
    for names_file, coverage_file in (
        ("numpy_linalg_names.tsv", "numpy_linalg_coverage.tsv"),
        ("scipy_linalg_functions.tsv", "scipy_linalg_coverage.tsv"),
    ):
        if not (DATA / coverage_file).exists():
            print(f"{coverage_file}: not present, skipped")
            continue
        names = set(read(DATA / names_file).splitlines()[1:])
        rows = [line.split("\t")[0] for line in read(DATA / coverage_file).splitlines()[1:]]
        missing = names - set(rows)
        extra = set(rows) - names
        state = "ok" if not missing and not extra else f"drift: missing={sorted(missing)} extra={sorted(extra)}"
        print(f"{coverage_file}: {len(rows)} rows vs {len(names)} names — {state}")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    collect_linalg()
    collect_blas_drivers()
    collect_lapack_drivers()
    collect_ffi()
    collect_python()
    validate_annotations()


if __name__ == "__main__":
    main()
