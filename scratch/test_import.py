import sys
import os

# Add MLRL01 to path to simulate running from there
sys.path.append(os.path.join(os.getcwd(), "MLRL01"))

try:
    from features import feature_engineering
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
