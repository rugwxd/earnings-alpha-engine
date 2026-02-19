"""Verify that all required packages are installed and print their versions."""

import importlib
import sys

PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "yfinance": "yfinance",
    "torch": "torch",
    "transformers": "transformers",
    "xgboost": "xgboost",
}

print(f"Python {sys.version}\n")

missing = []
for display_name, import_name in PACKAGES.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {display_name:15s} {version}")
    except ImportError:
        missing.append(display_name)
        print(f"  {display_name:15s} *** MISSING ***")

print()
if missing:
    print(f"MISSING packages: {', '.join(missing)}")
    print(f"Install with:  pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("All packages installed.")
