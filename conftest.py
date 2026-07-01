"""Make ``src/`` importable regardless of the editable-install state.

macOS marks the venv (and the files uv writes into it) with the ``UF_HIDDEN``
flag, and CPython 3.12's ``site`` module silently skips hidden ``.pth`` files.
Each time uv rewrites the editable install it re-sets that flag, which
intermittently breaks ``import pairs_teardown`` under pytest. Prepending the
source tree here makes test collection independent of that mechanism — it is
tracked in git and survives a full ``rm -rf .venv``.
"""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
