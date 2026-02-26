# batch_run_build_associates.py
import time, importlib, secrets
from pathlib import Path
import sys

# Ensure we can import build_associates_from_parquet.py from the same folder
sys.path.insert(0, str(Path(__file__).parent))

# Import your main script as a module
build = importlib.import_module("build_associates_from_parquet_regionfilter_2")

N_RUNS = 10 # How many runs do you want? That is the number of iterations. 

for i in range(1, N_RUNS + 1):
    seed = secrets.randbits(64)  # unpredictable seed
    print(f"\n=== RUN {i}/{N_RUNS} (seed={seed}) ===")
    t0 = time.time()
    try:
        build.main(seed=seed)
        print(f"[OK] Run {i} finished in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[ERROR] Run {i} failed: {e}")
    time.sleep(1)

print("\nAll batch runs completed.")
