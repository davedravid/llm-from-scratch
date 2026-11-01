"""Report what this computer has for the project. Uses only Python's standard library."""
import importlib.util
import os
import platform
import sys

print(f"Python version : {platform.python_version()}  (need 3.10+)")
print(f"Running from   : {sys.executable}")
print(f"CPU threads    : {os.cpu_count()}")

for lib, phase in [("numpy", "Phase 2"), ("torch", "Phase 4")]:
    found = importlib.util.find_spec(lib) is not None
    status = "installed" if found else f"not installed yet (needed from {phase})"
    print(f"{lib:<15}: {status}")

if importlib.util.find_spec("torch") is not None:
    import torch
    print(f"torch version  : {torch.__version__}")
    print(f"NVIDIA GPU     : {'yes' if torch.cuda.is_available() else 'no (CPU only, which is fine)'}")

ok = sys.version_info >= (3, 10)
print("\nResult:", "OK, ready for Phase 1" if ok else "Python is too old, please install 3.10+")
