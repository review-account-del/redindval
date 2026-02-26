
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High SD but high stability_pos: “varies, but still reliably positive” (often OK)

High SD and stability_pos ~ 0.5: classic “coin-flip indicator” (flag as unreliable)

stability_ge1 is your “strong indicator repeatability”

stability_pos = fraction of runs with RedIndVal > 0

stability_ge1 = fraction of runs with RedIndVal >= 1

stability_class3plus = fraction of runs with RedIndVal_JenksClass >= 3 (if present)

stability_same_sign = fraction of runs with same sign as the mean
"""
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

# -------------------------
# CONFIG
# -------------------------
INPUT_PATH  = Path("/home/FINAL_results_composite_2.txt")  # change as needed
OUTPUT_PATH = Path("/home/FINAL_results_composite_2_stability_report.txt")                 # change as needed

GROUP_COLS = ["focal_id"]  # You can add "main_habitat" if you want per-habitat stability

VALUE_COL = "RedIndVal"

# Minimum sampling universe used when generating runs (from build_associates_from_parquet.py)
SAMPLE_SITES = 350

# Observation count column used to filter out under-sampled species
OBS_COL = "No_obs"

JENKS_COL = "RedIndVal_JenksClass"  # optional input column (per-run)
JENKS_K = 4
JENKS_MEAN_COL = "RedIndVal_mean_JenksClass"  # computed from RedIndVal_mean in this evaluator

# Thresholds used for stability scores
THRESH_POS = 0.0
THRESH_STRONG = 1.0
THRESH_CLASS3 = 3# https://chatgpt.com/g/g-p-68c2de7204f4819195c090cde725d0a9/c/693b1601-36e8-8327-ac6a-1ae0aa60219d 

# Small epsilon to avoid division by zero in Relative_SD
EPS = 1e-12


# -------------------------
# Filtering: drop species with too few observations
# -------------------------
def _resolve_obs_col(df: "pd.DataFrame", preferred: str):
    """Return the name of the observation-count column, or None if not found."""
    if preferred in df.columns:
        return preferred
    # Try common alternatives
    candidates = [
        "No_obs", "No_Obs", "no_obs", "n_obs",
        "No_observations", "no_observations", "observations",
        "No_localities", "no_localities", "n_localities",
        "n_sites", "No_sites"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# Jenks (computed from RedIndVal_mean)
# -------------------------
def jenks_breaks(values: np.ndarray, k: int) -> List[float]:
    """
    Compute Jenks Natural Breaks for a 1D array of values (no NaNs), returning k breakpoints.
    """
    values = np.asarray(values, dtype=float)
    values = np.sort(values)
    n = len(values)
    if n == 0 or k <= 1:
        return [float(values[0]), float(values[-1])] if n > 0 else []

    mat1 = np.zeros((n+1, k+1), dtype=float)
    mat2 = np.zeros((n+1, k+1), dtype=float)

    for j in range(1, k+1):
        mat1[0, j] = 1.0
        mat2[0, j] = 0.0
        for i in range(1, n+1):
            mat2[i, j] = float("inf")

    v = 0.0
    for i in range(1, n+1):
        s1 = s2 = w = 0.0
        for m in range(1, i+1):
            idx = i - m
            val = values[idx]
            s1 += val
            s2 += val * val
            w += 1.0
            v = s2 - (s1 * s1) / w
            if idx != 0:
                for j in range(2, k+1):
                    if mat2[i, j] >= (v + mat2[idx, j-1]):
                        mat1[i, j] = float(idx + 1)
                        mat2[i, j] = v + mat2[idx, j-1]
        mat1[i, 1] = 1.0
        mat2[i, 1] = v

    breaks = [0.0] * (k+1)
    breaks[k] = float(values[-1])
    count = k
    idx = n
    while count >= 2:
        idxt = int(mat1[idx, count]) - 1
        breaks[count-1] = float(values[idxt])
        idx = int(mat1[idx, count] - 1)
        count -= 1
    breaks[0] = float(values[0])
    return breaks

def jenks_classify_nonnegative(series: pd.Series, k: int) -> pd.Series:
    """
    Classify values in 'series' into k Jenks classes (1..k),
    but only for values >= 0.

    - Jenks breaks are computed on the subset of values >= 0.
    - The first break is forced to 0.0.
    - Any negative or NaN value gets NaN class (ignored).
    """
    vals = pd.to_numeric(series, errors="coerce")

    # Only use non-negative values for Jenks
    nonneg_mask = (vals >= 0) & np.isfinite(vals)
    valid = vals[nonneg_mask]
    if valid.size < k:
        return pd.Series(np.nan, index=series.index)

    breaks = jenks_breaks(valid.to_numpy(), k=k)
    breaks = np.unique(breaks)
    if breaks.size < 2:
        return pd.Series(np.nan, index=series.index)

    # Force first break to 0 and last break to max(valid)
    breaks[0] = 0.0
    breaks[-1] = float(valid.max())

    classes = []
    for v in vals:
        # ignore NaNs and negative values
        if not np.isfinite(v) or v < 0:
            classes.append(np.nan)
            continue
        c = 1
        for i in range(1, len(breaks)):
            if v <= breaks[i]:
                c = i
                break
            c = i
        # Ensure classes 1..k
        c = max(1, min(c, k))
        classes.append(float(c))
    return pd.Series(classes, index=series.index)


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sign_of_mean(x: float) -> int:
    # returns -1, 0, or +1
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # Read (tab-delimited)
    df = pd.read_csv(INPUT_PATH, sep="\t", dtype=str)


    df_raw = df.copy()

    # Filter out under-sampled species (prevents artificially zero SD/range when No_obs < SAMPLE_SITES)
    obs_col = _resolve_obs_col(df, OBS_COL)
    if obs_col is None:
        print("[WARN] OBS_COL not found in input; no under-sampling filter applied.")
    else:
        before_rows = len(df)
        # Drop rows where observation count is missing or below threshold
        df = df[df[obs_col].notna() & (df[obs_col].astype(float) >= float(SAMPLE_SITES))].copy()
        after_rows = len(df)
        dropped = before_rows - after_rows
        if dropped > 0:
            # Also report how many focal groups were removed (if possible)
            if GROUP_COLS and all(c in df.columns for c in GROUP_COLS):
                # compute removed groups using original
                try:
                    kept_groups = set(tuple(x) for x in df[GROUP_COLS].drop_duplicates().itertuples(index=False, name=None))
                    all_groups = set(tuple(x) for x in df_raw[GROUP_COLS].drop_duplicates().itertuples(index=False, name=None))
                    removed_groups = len(all_groups - kept_groups)
                    print(f"[INFO] Under-sampling filter removed {removed_groups} group(s) based on {GROUP_COLS}.")
                except Exception:
                    # Keep going; this is only reporting
                    pass
            print(f"[INFO] Under-sampling filter: dropped {dropped} rows where {obs_col} < {SAMPLE_SITES}.")

    # Ensure numeric columns are numeric
    if VALUE_COL not in df.columns:
        raise ValueError(f"Missing required column: {VALUE_COL}")

    df[VALUE_COL] = safe_float_series(df[VALUE_COL])

    if JENKS_COL in df.columns:
        df[JENKS_COL] = safe_float_series(df[JENKS_COL])

    # Drop rows where RedIndVal is missing
    df = df.dropna(subset=[VALUE_COL]).copy()

    # Aggregate per species (or species x habitat if you add that to GROUP_COLS)
    rows = []
    for key, sub in df.groupby(GROUP_COLS, dropna=False):
        x = sub[VALUE_COL].to_numpy(dtype=float)
        n = int(len(x))

        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if n >= 2 else 0.0
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        rng = xmax - xmin

        rel_sd = float(sd / max(abs(mean), EPS))

        stability_pos = float(np.mean(x > THRESH_POS))
        stability_ge1 = float(np.mean(x >= THRESH_STRONG))

        # sign-consistency: share of runs that match sign(mean)
        sgn = sign_of_mean(mean)
        if sgn == 0:
            stability_same_sign = float(np.mean(x == 0))
        else:
            stability_same_sign = float(np.mean(np.sign(x) == sgn))

        # Jenks stability (optional)
        if JENKS_COL in sub.columns and sub[JENKS_COL].notna().any():
            j = sub[JENKS_COL].dropna().to_numpy(dtype=float)
            stability_class3plus = float(np.mean(j >= THRESH_CLASS3)) if len(j) else np.nan
        else:
            stability_class3plus = np.nan

        # Keep identifying columns
        if isinstance(key, tuple):
            out_key = list(key)
        else:
            out_key = [key]

        rec = {col: out_key[i] for i, col in enumerate(GROUP_COLS)}
        rec.update({
            "n_runs": n,
            "RedIndVal_mean": mean,
            "RedIndVal_sd": sd,
            "RedIndVal_min": xmin,
            "RedIndVal_max": xmax,
            "RedIndVal_range": rng,
            "Relative_SD": rel_sd,
            "stability_pos": stability_pos,
            "stability_ge1": stability_ge1,
            "stability_same_sign": stability_same_sign,
            "stability_class3plus": stability_class3plus,
        })
        rows.append(rec)

    out = pd.DataFrame(rows)

    # Jenks classes computed on the *mean* RedIndVal per group (values >= 0 only; first break forced to 0)
    if 'RedIndVal_mean' in out.columns:
        out[JENKS_MEAN_COL] = jenks_classify_nonnegative(out['RedIndVal_mean'], k=JENKS_K)
        out['mean_class3plus'] = (out[JENKS_MEAN_COL] >= THRESH_CLASS3).astype('float').where(out[JENKS_MEAN_COL].notna(), np.nan)
    else:
        out[JENKS_MEAN_COL] = np.nan
        out['mean_class3plus'] = np.nan

    # Optional: add a simple combined stability score (0..1), tuned for decision-use
    # Emphasize: positive sign consistency + being >0 + (optionally) being >=1.
    out["stability_score"] = (
        0.50 * out["stability_pos"] +
        0.30 * out["stability_same_sign"] +
        0.20 * out["stability_ge1"]
    )

    # Sort: show most unstable (high SD + low stability) at top
    out["instability_score"] = (out["Relative_SD"]) * (1.0 - out["stability_score"])
    out = out.sort_values(["instability_score", "RedIndVal_sd"], ascending=[False, False])

    # Write report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, sep="\t", index=False)

    # Console summary
    print("Wrote:", OUTPUT_PATH)
    print("Rows:", len(out))
    print("")
    print("Top 20 most unstable (by instability_score):")
    cols_show = GROUP_COLS + [
        "n_runs", "RedIndVal_mean", "RedIndVal_sd", "RedIndVal_range",
        "Relative_SD", "stability_pos", "stability_ge1", "stability_score",
        "instability_score"
    ]
    print(out[cols_show].head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
