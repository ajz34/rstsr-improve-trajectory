#!/usr/bin/env python3
"""Per-test compliance table for a rstsr-faer-py array-api-tests report.

Reads a pytest-json-report JSON and emits one CSV row per collected test with
the failure class and a first-cut fix-layer assignment (shim vs rust).

Usage:
    compliance_table.py reports/<report>.json [-o out.csv] [--census]

The fix-layer column is a heuristic mapped from the crash message plus the
name->capability table below (rstsr surface checked 2026-10-05 against
rstsr-core prelude exports + faer_impl linalg traits). Rows the heuristic
cannot place are marked `unclassified` and counted in the census so they can
be triaged by hand.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys

# ---------------------------------------------------------------- capability --
# rstsr primitive availability for array-api names, checked against the
# rstsr-core prelude export list and rstsr-linalg-traits/faer_impl.
#   bind   : rt:: has the primitive; shim-side binding work only
#   alias  : rt:: has it under another name/semantics; shim maps it
#   absent : no rstsr primitive; needs a rust-side implementation
NAME_LAYER = {}
for n in """acos acosh asin asinh atan atanh ceil conj cos cosh exp expm1 floor
           imag log log1p log2 log10 real reciprocal round sign signbit sin sinh
           sqrt square tan tanh trunc abs""".split():
    NAME_LAYER[n] = "shim-bind"
NAME_LAYER.update({
    "atan2": "shim-bind", "copysign": "shim-bind", "floor_divide": "shim-bind",
    "hypot": "shim-bind", "logaddexp": "shim-alias", "maximum": "shim-bind",
    "minimum": "shim-bind", "nextafter": "shim-bind", "pow": "shim-bind",
    "remainder": "shim-alias",
    "bitwise_and": "shim-alias", "bitwise_or": "shim-alias", "bitwise_xor": "shim-alias",
    "bitwise_invert": "shim-alias", "bitwise_left_shift": "shim-alias",
    "bitwise_right_shift": "shim-alias", "logical_and": "shim-alias",
    "logical_or": "shim-alias", "logical_xor": "shim-alias", "logical_not": "shim-alias",
    "max": "shim-bind", "min": "shim-bind", "mean": "shim-bind", "prod": "shim-bind",
    "std": "shim-bind", "var": "shim-bind", "argmax": "shim-bind", "argmin": "shim-bind",
    "count_nonzero": "shim-bind", "sum": "shim-bind", "all": "shim-bind", "any": "shim-bind",
    "expand_dims": "shim-bind", "squeeze": "shim-bind", "flip": "shim-bind",
    "moveaxis": "shim-bind", "concat": "shim-bind", "stack": "shim-bind",
    "unstack": "shim-bind", "broadcast_arrays": "shim-bind", "broadcast_shapes": "shim-bind",
    "empty_like": "shim-bind", "eye": "shim-bind", "tril": "shim-bind", "triu": "shim-bind",
    "full_like": "shim-bind", "ones_like": "shim-bind", "zeros_like": "shim-bind",
    "linspace": "shim-bind", "meshgrid": "shim-bind", "take": "shim-bind",
    "matmul": "shim-bind", "matrix_transpose": "shim-bind", "vecdot": "shim-bind",
    "cholesky": "shim-bind", "det": "shim-bind", "eigh": "shim-bind",
    "eigvalsh": "shim-bind", "inv": "shim-bind", "pinv": "shim-bind",
    "solve": "shim-bind", "svd": "shim-bind", "svdvals": "shim-bind",
    "where": "rust-impl", "nonzero": "rust-impl", "unique_all": "rust-impl",
    "unique_counts": "rust-impl", "unique_inverse": "rust-impl", "unique_values": "rust-impl",
    "searchsorted": "rust-impl", "isin": "rust-impl", "take_along_axis": "rust-impl",
    "sort": "rust-impl", "argsort": "rust-impl", "clip": "rust-impl",
    "cumulative_sum": "rust-impl", "cumulative_prod": "rust-impl", "roll": "rust-impl",
    "repeat": "rust-impl", "tile": "rust-impl", "diff": "rust-impl",
    "tensordot": "rust-impl", "trace": "rust-impl", "outer": "rust-impl",
    "cross": "rust-impl", "matrix_norm": "rust-impl", "vector_norm": "rust-impl",
    "matrix_rank": "rust-impl", "matrix_power": "rust-impl", "qr": "rust-impl",
    "slogdet": "rust-impl", "solve_symmetric": "rust-impl", "positive": "rust-impl",
    "result_type": "rust-impl", "can_cast": "rust-impl", "isdtype": "rust-impl",
    "fft": "suite-scope", "linalg": "mixed",  # linalg namespace: partially rust
    "dtype": "shim-py", "device": "shim-py", "default_device": "shim-py",
})

# spec dunder -> rt primitive (bind) or absent
DUNDER_LAYER = {
    "__abs__": "shim-bind", "__neg__": "shim-bind", "__invert__": "shim-alias",
    "__and__": "shim-alias", "__or__": "shim-alias", "__xor__": "shim-alias",
    "__lshift__": "shim-alias", "__rshift__": "shim-alias",
    "__floordiv__": "shim-bind", "__mod__": "shim-alias", "__pow__": "shim-bind",
    "__matmul__": "shim-bind", "__pos__": "rust-impl",
    "__iand__": "shim-alias", "__ior__": "shim-alias", "__ixor__": "shim-alias",
    "__ilshift__": "shim-alias", "__irshift__": "shim-alias",
    "__ifloordiv__": "shim-bind", "__imod__": "shim-alias", "__ipow__": "shim-bind",
    "__iadd__": "shim-bind", "__isub__": "shim-bind", "__imul__": "shim-bind",
    "__itruediv__": "shim-bind",
}

OP_TO_DUNDER = {
    "**": "__pow__", "**=": "__ipow__", "//": "__floordiv__", "//=": "__ifloordiv__",
    "%": "__mod__", "%=": "__imod__", "&": "__and__", "&=": "__iand__",
    "|": "__or__", "|=": "__ior__", "^": "__xor__", "^=": "__ixor__",
    "<<": "__lshift__", "<<=": "__ilshift__", ">>": "__rshift__", ">>=": "__irshift__",
    "@": "__matmul__", "@=": "__imatmul__",
}


def crash(x):
    for ph in ("call", "setup", "teardown"):
        c = (x.get(ph) or {}).get("crash")
        if c:
            return c
    return {}


def norm_msg(m):
    h = m.split("\n")[0]
    h = re.sub(r"'[^']{0,80}'", "'X'", h)
    h = re.sub(r'"[^"]{0,80}"', '"X"', h)
    h = re.sub(r"\b\d+(\.\d+)?\b", "N", h)
    return h[:160]


def extract_names(nodeid, msg):
    names = []
    names += re.findall(r"has no attribute '([A-Za-z_][A-Za-z0-9_]*)'", msg)
    names += re.findall(r"([A-Za-z_][A-Za-z0-9_]*) is not defined in rstsr_faer\.api", msg)
    names += re.findall(r"is missing the [\w -]+ function ([A-Za-z_][A-Za-z0-9_]*)\(\)", msg)
    names += re.findall(r"is missing the ([A-Za-z_][A-Za-z0-9_]*)\(\) function", msg)
    names += re.findall(r"is missing the method (__[A-Za-z_]+__)\(\)", msg)
    names += re.findall(r"([A-Za-z_][A-Za-z0-9_]*) not found in array module", msg)
    names += re.findall(r"(__[A-Za-z_]+__) not found in array object", msg)
    names += re.findall(r"Argument '([A-Za-z_][A-Za-z0-9_]*)' missing from signature", msg)
    if "NoneType' object is not callable" in msg:
        m = re.search(r"test_\w+\[([A-Za-z_][A-Za-z0-9_]*)", nodeid)
        if m:
            names.append(m.group(1))
        else:
            names.append("?none-callable")
    return sorted(set(names))


def classify(x):
    """-> (root_cause, names, fix_layer, detail)"""
    nodeid = x["nodeid"]
    cr = crash(x)
    msg = cr.get("message", "")
    path = cr.get("path", "")
    loc = ("suite" if "array_api_tests" in path
           else "shim-py" if path.endswith(".py")
           else "shim-rs" if "rstsr_faer" in path else "rust-native")
    names = extract_names(nodeid, msg)

    # --- explicit shim declines -------------------------------------------
    if msg.startswith("NotImplementedError") and ("indexing" in msg or "boolean-mask" in msg or "integer-array" in msg):
        return "indexing-gap", [], "rust-impl", "G-038/G-039"
    if msg.startswith("NotImplementedError") and "reductions over axes" in msg:
        return "axes-not-bound", [], "shim-bind", "rt has *_axes"
    if msg.startswith("NotImplementedError"):
        return "declined-by-design", [], "undecided", norm_msg(msg)

    # --- missing surface ---------------------------------------------------
    if names:
        layers = {NAME_LAYER.get(n, "undecided") for n in names
                  if not n.startswith("?") and not n.startswith("__")}
        dunder = [n for n in names if n.startswith("__")]
        for d in dunder:
            layers.add(DUNDER_LAYER.get(d, "undecided"))
        if not layers:
            layers = {"undecided"}
        if len(layers) == 1:
            layer = next(iter(layers))
        elif layers <= {"shim-bind", "shim-alias"}:
            layer = "shim-bind"
        elif "rust-impl" in layers and layers <= {"rust-impl", "undecided"}:
            layer = "rust-impl"
        else:
            layer = "mixed"
        # has_names-style assertion vs call-site failure
        cause = "missing-surface" if loc == "suite" else "missing-surface"
        return cause, names, layer, norm_msg(msg)

    # --- operator dunders --------------------------------------------------
    m = re.search(r"unsupported operand type\(s\) for (.+?):", msg)
    if m:
        ops = m.group(1)
        ds = [OP_TO_DUNDER.get(o.strip()) for o in re.split(r" or ", ops)]
        ds = [d for d in ds if d]
        layer = "shim-bind"
        if any(DUNDER_LAYER.get(d) == "rust-impl" for d in ds):
            layer = "rust-impl"
        return "missing-dunder", ds, layer, ops
    m = re.search(r"bad operand type for ([a-z]+)\(\)", msg)
    if m:
        d = {"abs": "__abs__"}.get(m.group(1), "?")
        return "missing-dunder", [d], DUNDER_LAYER.get(d, "undecided"), m.group(1)

    # --- promotion / marshalling ------------------------------------------
    if "cross-dtype array promotion is not provided" in msg:
        return "promotion-decline-dtype", [], "rust-design", "G-009"
    if "cross-kind scalar promotion" in msg:
        return "promotion-decline-scalar", [], "rust-design", "G-009/scalar"
    if "expected an rstsr_faer.api Array, got" in msg:
        return "marshal-reject", [], "shim-py", norm_msg(msg)

    # --- known shim bugs ---------------------------------------------------
    if "finfo" in msg and "only real floating-point dtypes" in msg:
        return "dtype-fn-bug", ["finfo"], "shim-rs", "G-032"
    if "iinfo" in msg and "integral" in msg:
        return "dtype-fn-bug", ["iinfo"], "shim-rs", "G-032"

    # --- assertions / unexpected exceptions --------------------------------
    if msg.startswith("AssertionError"):
        return "wrong-value", [], "mixed-values", norm_msg(msg)
    if msg.startswith("ExceptionGroup"):
        return "wrong-value", [], "mixed-values", norm_msg(msg)
    if loc in ("shim-py", "shim-rs", "rust-native"):
        return "unexpected-exception", [], "rust-fix-or-shim", norm_msg(msg)
    return "unclassified", [], "unclassified", norm_msg(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--census", action="store_true")
    a = ap.parse_args()

    d = json.load(open(a.report))
    rows = []
    for x in d["tests"]:
        cr = crash(x)
        path = cr.get("path", "")
        loc = ("suite" if "array_api_tests" in path
               else "shim-py" if path.endswith(".py")
               else "shim-rs" if "rstsr_faer" in path else "rust-native") if cr else ""
        if x["outcome"] == "failed":
            cause, names, layer, detail = classify(x)
        else:
            cause, names, layer, detail = "", [], "", ""
        md = x.get("metadata") or {}
        mod = md.get("test_module")
        if not mod:
            mod = x["nodeid"].split("::")[0].replace("/", ".").removesuffix(".py")
        rows.append({
            "suite_file": mod.split(".")[-1] + ".py",
            "nodeid": x["nodeid"],
            "test_function": md.get("test_function", ""),
            "spec_function": md.get("array_api_function_name", ""),
            "outcome": x["outcome"],
            "params": md.get("params", ""),
            "root_cause": cause,
            "names": ";".join(names),
            "fix_layer": layer,
            "crash_loc": loc,
            "detail": detail,
        })

    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {a.out}", file=sys.stderr)

    if a.census or not a.out:
        fails = [r for r in rows if r["outcome"] == "failed"]
        print(f"total {len(rows)}  passed {sum(r['outcome']=='passed' for r in rows)}  "
              f"failed {len(fails)}  skipped {sum(r['outcome']=='skipped' for r in rows)}")
        print("\n-- root cause --")
        for k, v in collections.Counter(r["root_cause"] for r in fails).most_common():
            print(f"{v:5d}  {k}")
        print("\n-- fix layer --")
        for k, v in collections.Counter(r["fix_layer"] for r in fails).most_common():
            print(f"{v:5d}  {k}")
        print("\n-- by suite file --")
        for k, v in collections.Counter(r["suite_file"] for r in fails).most_common():
            print(f"{v:5d}  {k}")


if __name__ == "__main__":
    main()
