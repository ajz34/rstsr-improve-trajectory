"""Turn a pytest-json-report file from array-api-tests into a readable summary.

    ./.venv/bin/python summarize_report.py [reports/numpy-*.json]
    # no argument -> newest report in reports/

Reads the per-test metadata that array-api-tests attaches through
reporting.py: every test carries its spec function name
(`array_api_function_name`), so failures can be grouped by *spec function*
rather than by pytest node id.
"""

import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")

SEPARATOR = re.compile(r"^[=\-+.\s]+$")
ERROR_LINE = re.compile(r"E\s+(\w*(?:Error|Exception|Failure)\b.*)")
ASSERT_LINE = re.compile(r"E\s+(.*\bassert\b.*)")


def pick_message(longrepr: str) -> str:
    """Best-effort one-liner out of a pytest longrepr."""
    elines = [ln[2:] for ln in longrepr.splitlines() if ln.startswith("E ")]
    for pattern in (ERROR_LINE, ASSERT_LINE):
        for ln in elines:
            m = pattern.match("E " + ln)
            if m and not SEPARATOR.match(m.group(1)):
                return " ".join(m.group(1).split())
    for ln in elines:
        if not SEPARATOR.match(ln):
            return " ".join(ln.split())
    lines = [ln for ln in longrepr.splitlines() if ln.strip()]
    return " ".join(lines[-1].split()) if lines else "<no message>"


def phase_longrepr(test):
    for phase in ("call", "setup", "teardown"):
        lr = (test.get(phase) or {}).get("longrepr")
        if lr:
            return lr
    return ""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        candidates = [p for p in sorted(glob.glob(os.path.join(REPORTS, "*.json")))
                      if not p.endswith(".meta.json")]
        if not candidates:
            sys.exit(f"no reports in {REPORTS}/ - run ./run.sh first")
        path = candidates[-1]

    report = json.load(open(path))
    tests = report["tests"]
    summary = report["summary"]

    import datetime

    created = report.get("created")
    created = (datetime.datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M")
               if isinstance(created, (int, float)) else str(created))

    print(f"report   : {os.path.relpath(path, HERE)}")
    sidecar = path + ".meta.json"
    if os.path.exists(sidecar):
        meta = json.load(open(sidecar))
        print(f"library  : {meta['module']} {meta.get('version')}  "
              f"(declares Array API {meta.get('api_version')})")
        print(f"suite    : {meta.get('suite_commit')}   max-examples: {meta.get('max_examples')}")
    print(f"created  : {created}   duration: {report.get('duration', 0):.0f}s")
    print(f"result   : {summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed, "
          f"{summary.get('skipped', 0)} skipped   ({summary.get('collected', 0)} collected)")

    failed = [t for t in tests if t["outcome"] == "failed"]
    errors = [t for t in tests if t["outcome"] == "error"]
    skipped = [t for t in tests if t["outcome"] == "skipped"]

    if not failed and not errors:
        print("\nno failures")
    if failed or errors:
        print(f"\n== {len(failed) + len(errors)} failing tests ==\n")

        by_module = collections.Counter(
            (t.get("metadata") or {}).get("test_module", "?").split(".")[-1]
            for t in failed + errors
        )
        for module, n in by_module.most_common():
            print(f"  {n:5d}  {module}")

        print("\n  per spec function (failures / tests run):")
        by_func = collections.Counter(
            (t.get("metadata") or {}).get("array_api_function_name") for t in failed + errors
        )
        ran = collections.Counter(
            (t.get("metadata") or {}).get("array_api_function_name") for t in tests
        )
        for func, n in by_func.most_common():
            print(f"  {n:5d} / {ran[func]:<5d}  {func}")

        # representative message per spec function, first failure wins
        print("\n  first failure per spec function:")
        seen = {}
        for t in failed + errors:
            func = (t.get("metadata") or {}).get("array_api_function_name") or t["nodeid"]
            seen.setdefault(func, (t["nodeid"], pick_message(phase_longrepr(t))))
        for func, (nodeid, msg) in seen.items():
            print(f"  [{func}]")
            print(f"    {nodeid.split('::')[-1][:100]}")
            print(f"    {msg[:160]}")

    if skipped:
        reasons = collections.Counter()
        for t in skipped:
            lr = phase_longrepr(t)
            m = re.search(r"'Skipped: ([^']*)'", lr) or re.search(r"'([^']*)'\s*\)?\s*$", lr.strip())
            reasons[m.group(1) if m else lr.strip()[-60:]] += 1
        print("\n== skipped ==")
        for reason, n in reasons.most_common():
            print(f"  {n:5d}  {reason}")


if __name__ == "__main__":
    main()
