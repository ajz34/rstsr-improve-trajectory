"""pytest plugin: run hypothesis without its explain phase (NO_EXPLAIN=1).

The suite's conftest registers and loads its own "array-api-tests" hypothesis
profile in pytest_configure; this plugin registers a child profile
("no-explain": same max_examples / derandomize / deadline, phases minus
Phase.explain) and loads it from its own pytest_configure hook, which pytest
runs after the suite conftest's (LIFO hook order — loading later, e.g. from
collection_modifyitems, does not take effect).

Only the post-failure explanation blob ("Draw N ... (or any other generated
value)") is dropped; outcomes, counts and falsifying examples are unchanged.
Measured 2026-10-04, rstsr-faer-py red run: explain was 34.2s of 36.8s (93%)
on test_manipulation_functions.py; full suite 87s with, several minutes without.
"""

from hypothesis import Phase, settings
from hypothesis.errors import InvalidArgument


def pytest_configure(config):
    try:
        parent = settings.get_profile("array-api-tests")
    except InvalidArgument:
        parent = settings.default
    settings.register_profile(
        "no-explain",
        parent=parent,
        phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink],
    )
    settings.load_profile("no-explain")
