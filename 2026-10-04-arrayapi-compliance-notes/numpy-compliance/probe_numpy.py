"""What does this NumPy claim about the Array API?

Run it with the same interpreter the suite will use:

    ./.venv/bin/python probe_numpy.py

Everything here is *self-declared* by numpy.  The suite's job is to check the
claims.  In particular `__array_api_version__` is the version the suite will
grade against (it reads this attribute; see array_api_tests/__init__.py:88).
"""

import platform
import sys

import numpy as np

print(f"python          : {sys.version.split()[0]} ({platform.machine()})")
print(f"numpy           : {np.__version__}")
print(f"imported from   : {np.__file__}")
print(f"dev build?      : {'yes' if '+' in np.__version__ or np.__version__.endswith('.dev0') else 'no (release)'}")
print()
print(f"__array_api_version__ : {getattr(np, '__array_api_version__', '<absent>')}")
print(f"ndarray.__array_namespace__ : {hasattr(np.ndarray, '__array_namespace__')}")
x = np.arange(6).reshape(2, 3)
print(f"x.__array_namespace__() is numpy : {x.__array_namespace__() is np}")
print()

info = np.__array_namespace_info__()
print("__array_namespace_info__()")
print(f"  devices        : {info.devices()}")
print(f"  default dtypes : { {k: str(v) for k, v in info.default_dtypes().items()} }")
print(f"  capabilities   : {info.capabilities()}")
print(f"  dtypes         : {list(info.dtypes())}")
