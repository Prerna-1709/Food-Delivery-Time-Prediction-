"""verify_setup.py — Import verification for Food-Delivery-Intelligence."""
import sys

LIBRARIES = [
    ("pandas",         "pd"),
    ("numpy",          "np"),
    ("sklearn",        "sklearn"),
    ("matplotlib",     "matplotlib"),
    ("seaborn",        "sns"),
    ("scipy",          "scipy"),
    ("joblib",         "joblib"),
    ("tensorflow",     "tf"),
    ("keras",          "keras"),
]

sep = "=" * 52
print(f"\n{sep}")
print("  IMPORT VERIFICATION — Food-Delivery-Intelligence")
print(sep)

all_ok = True
for lib, alias in LIBRARIES:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "n/a")
        print(f"  [OK]  {lib:<16} version: {ver}")
    except ImportError as e:
        print(f"  [FAIL] {lib:<15} ERROR: {e}")
        all_ok = False

# Config module
print()
try:
    sys.path.insert(0, ".")
    from src.utils.config import (
        PROJECT_ROOT, RAW_CSV, RANDOM_SEED,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOL_FEATURES,
        DL_EPOCHS, DL_BATCH_SIZE,
    )
    print(f"  [OK]  config.py loaded")
    print(f"        PROJECT_ROOT    = {PROJECT_ROOT}")
    print(f"        RANDOM_SEED     = {RANDOM_SEED}")
    print(f"        DELAY_THRESHOLD = 40 mins")
    print(f"        DL_EPOCHS       = {DL_EPOCHS}  |  BATCH = {DL_BATCH_SIZE}")
    print(f"        Features        = {len(NUMERIC_FEATURES)} numeric, "
          f"{len(CATEGORICAL_FEATURES)} categorical, {len(BOOL_FEATURES)} bool")
except Exception as e:
    print(f"  [FAIL] config.py  ERROR: {e}")
    all_ok = False

print(f"\n{sep}")
if all_ok:
    print("  [PASS] All imports succeeded. Environment is ready.")
else:
    print("  [FAIL] Some imports failed — see above.")
print(sep + "\n")
sys.exit(0 if all_ok else 1)
