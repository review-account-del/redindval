from __future__ import annotations
"""
indicator_baseline_composite.py
--------------------------------
Effort-weighted, effort-stratified indicator baselines + composite index.

This version implements:

  A1) Redlist exceedance baseline (mean_loglift_0to4, lifted_PM_tau, frac_enriched)
  A5) Natura 2000 enrichment (mean_loglift on n2000_score)
  HP) Habitat purity (share of associated taxa in main habitat)
  B)  Robust per-habitat z-scores + composite RedIndVal_star2
  C)  Jenks Natural Breaks classification of RedIndVal_star2

Jenks here is:
  - computed only for RedIndVal_star2 >= 0
  - forced to start at 0
  - uses 4 classes (1..4)

Inputs (must exist before running):
  - base_folder/
      ├─ <focal 1>/
      │    ├─ RedSpeciesLocal.txt          (LocalityID Value)
      │    ├─ totalTargetObs.txt           (genus species_n obs species red)
      │    └─ allObservations.txt          (1 line: "<name> <id id id ...>")
      ├─ <focal 2>/ ...
  - LocalityMaster.txt (tab): columns at least
        LocalityID, w_inv, effort_stratum, n2000_score, dominant_habitat

Outputs:
  - FINAL_results_composite_2.txt (tab) in OUTPUT_DIR
  - CONFIG_*.txt with run configuration

Notes:
  - Uses robust pseudo-counting with Jeffreys prior (alpha=beta=0.5) for log-lift ratios.
  - Bootstraps are locality-level; background is sampled to match the focal's effort-stratum mix.
  - Composite index uses only A1, A5 and habitat purity, with per-habitat robust z-scores.
  - Composite is only computed when ALL three z-metrics are present; otherwise NaN.
  - Jenks Natural Breaks (k=4) is computed only on RedIndVal_star2 >= 0,
    with breaks starting at 0, and returns classes 1..4.
"""

import os, re, math, time, traceback
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import hashlib


# ----------------------- CONFIG -----------------------
BASE_FOLDER      = Path(r"/home/OUT_datafolders")                # per-species folders northern
LOCALITY_MASTER  = Path(r"/home/metadata/LocalityMaster.txt")
OUTPUT_DIR       = Path(r"/home/out/"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
META_DIR         = Path(r"/home/metadata/")
HABITAT_FILE     = META_DIR / "habitat_assignments.csv"                # taxonID;scientificName;habitat

# Baseline / lift config
BASE_THRESHOLDS = list(range(5))     # mean over t = 0..4
TAU_IND         = 0.30               # tau for redlist exceedance (A1)
TAU_N2000       = 0.25               # tau for N2000 exceedance (A5)
ALPHA_BETA      = (0.5, 0.5)
MIN_N_LOCALITIES= 30
BOOTSTRAPS      = 50
BG_BOOT_CAP     = 1000
RANDOM_SEED     = 42

# Composite weights (only A1, A5, Habitat purity)
# Note: equal proportions (1/3) give too high influence of W_HAB! W_HAB: R2=0.47, W_EXCEED: R2=0.15, W_N2000: R2=0.01
W_EXCEED = 0.4
W_N2000  = 0.4
W_HAB    = 0.2

# Jenks configuration
# We now use 4 classes and only on RedIndVal_star2 >= 0
JENKS_K = 4  # number of classes for Jenks Natural Breaks

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
STREAM_PATH = OUTPUT_DIR / f"FINAL_results_stream_{RUN_ID}.txt"

# --- Locality subsampling stability ---
LSA_REPEATS = 10

# j = min(MAX_J, max(MIN_J, floor(P_FRAC * m))) and if m < MIN_J then j=m
LSA_P_FRAC = 0.10
LSA_MIN_J = 30
LSA_MAX_J = 80

# Use a separate seed stream from the main bootstrap seed
LSA_SEED = RANDOM_SEED + 123


STREAM_COLUMNS = [
    "timestamp","folder","focal_id","main_habitat",
    "No_obs","No_asc_spp","No_redlisted","Habitat_purity",
    # Locality subsampling stability (alpha/gamma)
    "LSA_debug","LSA_m_localities","LSA_j_subsample","LSA_repeats",
    "gamma_unique_species_mean","gamma_unique_species_sd","gamma_cv","gamma_norm_range",
    "gamma_unique_species_min","gamma_unique_species_max","gamma_unique_species_range",
    "alpha_mean_spp_per_locality_mean","alpha_mean_spp_per_locality_sd","alpha_cv","alpha_norm_range",
    "alpha_mean_spp_per_locality_min","alpha_mean_spp_per_locality_max","alpha_mean_spp_per_locality_range",
    # A1 redlist exceedance
    "mean_loglift_0to4","mean_loglift_0to4_CI_low","mean_loglift_0to4_CI_high",
    "lifted_PM_tau0.30","lifted_PM_tau0.30_CI_low","lifted_PM_tau0.30_CI_high",
    "frac_enriched_tau0.30","frac_enriched_tau0.30_CI_low","frac_enriched_tau0.30_CI_high",
    # A5 N2000 enrichment
    "N2000_mean_loglift_0to4","N2000_CI_low","N2000_CI_high",
    # debug / counts
    "N_focal_localities","N_background_localities","N_LM_hab_localities",
    "A1_debug","A5_debug",
]

# --- robust CSV reader for semicolon files with bad encodings ---
def read_semicolon_csv_robust(path: Path, usecols: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                sep=";",
                encoding=enc,
                engine="python",
                on_bad_lines="skip"
            )
            break
        except UnicodeDecodeError:
            df = None
    if df is None:
        print(f"[WARN] Could not decode {path} as UTF-8 or Latin-1; skipping habitat assignment.")
        return None
    if usecols:
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            print(f"[WARN] {path.name} missing columns {missing}; available: {list(df.columns)}")
    return df

def _norm_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def _two_token_key(s: str) -> str:
    toks = _norm_name(s).split()
    return " ".join(toks[:2]) if len(toks) >= 2 else _norm_name(s)

def build_habitat_map(habitat_file: Path) -> pd.DataFrame | None:
    """
    Returns a frame with columns:
      taxon_id, scientificName, habitat, scientificName_norm, gs_key
    """
    hb = read_semicolon_csv_robust(habitat_file, usecols=["taxonID", "scientificName", "habitat"])
    if hb is None or hb.empty:
        return None
    hb["taxon_id"] = pd.to_numeric(hb.get("taxonID"), errors="coerce")
    hb["scientificName"] = hb.get("scientificName")
    hb["habitat"] = hb.get("habitat")
    hb = hb[["taxon_id","scientificName","habitat"]].dropna(subset=["scientificName","habitat"])
    hb["scientificName_norm"] = hb["scientificName"].map(_norm_name)
    hb["gs_key"] = hb["scientificName"].map(_two_token_key)
    hb = hb.drop_duplicates(subset=["taxon_id"]).drop_duplicates(subset=["scientificName_norm"])
    return hb

def infer_habitat_for(focal_id: str, habitat_df: pd.DataFrame | None) -> str:
    """Try exact normalized match; else genus+species prefix match; else 'Unknown'."""
    if habitat_df is None or habitat_df.empty:
        return "Unknown"
    s_norm = _norm_name(focal_id)
    # exact
    hit = habitat_df.loc[habitat_df["scientificName_norm"] == s_norm, "habitat"]
    if not hit.empty and pd.notna(hit.iloc[0]):
        return str(hit.iloc[0])
    # genus+species key
    key = _two_token_key(focal_id)
    hit = habitat_df.loc[habitat_df["gs_key"] == key, "habitat"]
    if not hit.empty and pd.notna(hit.iloc[0]):
        return str(hit.iloc[0])
    return "Unknown"

def _write_stream_row(path, columns, rowdict):
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8") as g:
        if is_new:
            g.write("\t".join(columns) + "\n")
        vals = []
        for c in columns:
            v = rowdict.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        g.write("\t".join(vals) + "\n"); g.flush(); os.fsync(g.fileno())

# ---------- small utils ----------
def _lsa_choose_j(m: int) -> int:
    if m <= 0:
        return 0
    if m < LSA_MIN_J:
        return m
    j = int(math.floor(LSA_P_FRAC * m))
    j = max(LSA_MIN_J, j)
    j = min(LSA_MAX_J, j)
    j = min(j, m)
    return j

def _lsa_is_header(tokens: list[str]) -> bool:
    if not tokens:
        return True
    joined = " ".join(tokens).lower()
    return ("locality" in joined and "sweref" in joined) or joined.startswith("localityid")

def _lsa_read_localityspecies_all(path: Path) -> list[list[str]]:
    """
    Reads LocalitySpecies_ALL.txt.
    Assumes tokens per row:
      LocalityID SWEREF_N SWEREF_E species_id_1 species_id_2 ...
    Returns list of localities, each locality is a list of species ID strings.
    """
    localities: list[list[str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if _lsa_is_header(toks):
                continue
            if len(toks) <= 3:
                localities.append([])
                continue
            spp = [t for t in toks[3:] if t not in ("<NA>", "NA", "nan")]
            localities.append(spp)
    return localities

def _lsa_summarize(vals: list[float]) -> tuple[float,float,float,float,float]:
    """
    mean, sd(sample), min, max, range
    """
    if not vals:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    vmin = float(min(vals))
    vmax = float(max(vals))
    mean = float(sum(vals) / len(vals))
    if len(vals) >= 2:
        var = sum((x - mean) ** 2 for x in vals) / float(len(vals) - 1)
        sd = float(math.sqrt(var))
    else:
        sd = 0.0
    rng = vmax - vmin
    return (mean, sd, vmin, vmax, rng)

def locality_subsample_alpha_gamma(locality_file: Path, repeats: int, seed: int) -> dict:
    """
    Option A: repeated random subsampling of localities.

    gamma = number of unique species IDs across subsample localities
    alpha = mean number of species IDs per locality within subsample

    Returns dict with mean/sd/min/max/range plus CV and normalized range for both.
    """
    if not locality_file.exists():
        return {"LSA_debug": "MISSING_LOCALITYSPECIES_ALL"}

    localities = _lsa_read_localityspecies_all(locality_file)
    m = len(localities)
    if m == 0:
        return {"LSA_debug": "EMPTY_LOCALITYSPECIES_ALL", "LSA_m_localities": 0}

    j = _lsa_choose_j(m)
    if j <= 0:
        return {"LSA_debug": "INVALID_SUBSAMPLE_SIZE", "LSA_m_localities": m, "LSA_j_subsample": j}

    rng = np.random.default_rng(seed)
    idx_all = np.arange(m)

    gamma_vals: list[float] = []
    alpha_vals: list[float] = []

    for _ in range(int(repeats)):
        # sample without replacement within each subsample
        subs = rng.choice(idx_all, size=j, replace=False)

        union = set()
        counts = []
        for ix in subs:
            spp = localities[int(ix)]
            counts.append(len(spp))
            union.update(spp)

        gamma_vals.append(float(len(union)))
        alpha_vals.append(float(sum(counts) / len(counts)) if counts else 0.0)

    g_mean, g_sd, g_min, g_max, g_rng = _lsa_summarize(gamma_vals)
    a_mean, a_sd, a_min, a_max, a_rng = _lsa_summarize(alpha_vals)

    eps = 1e-12
    out = {
        "LSA_debug": "OK",
        "LSA_m_localities": int(m),
        "LSA_j_subsample": int(j),
        "LSA_repeats": int(repeats),

        "gamma_unique_species_mean": g_mean,
        "gamma_unique_species_sd": g_sd,
        "gamma_unique_species_min": g_min,
        "gamma_unique_species_max": g_max,
        "gamma_unique_species_range": g_rng,
        "gamma_cv": (g_sd / max(abs(g_mean), eps)),
        "gamma_norm_range": (g_rng / max(abs(g_mean), eps)),

        "alpha_mean_spp_per_locality_mean": a_mean,
        "alpha_mean_spp_per_locality_sd": a_sd,
        "alpha_mean_spp_per_locality_min": a_min,
        "alpha_mean_spp_per_locality_max": a_max,
        "alpha_mean_spp_per_locality_range": a_rng,
        "alpha_cv": (a_sd / max(abs(a_mean), eps)),
        "alpha_norm_range": (a_rng / max(abs(a_mean), eps)),
    }
    return out

def parse_total_target_obs(path: Path) -> dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"{path.name} is empty")
    toks = line.split()
    nums, name = [], []
    for t in reversed(toks):
        if len(nums) < 3 and t.replace("-", "").isdigit():
            nums.append(int(t))
        else:
            name.append(t)
    if len(nums) < 3:
        m = re.search(r"(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if not m:
            raise ValueError(f"Could not parse counts from {line}")
        nums = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
        left = line[:m.start()].strip(); name = left.split()[::-1]
    name = list(reversed(name)); obs, species, red = nums[::-1]
    genus = name[0]; species_n = " ".join(name[1:]) if len(name) > 1 else ""
    focal_id = " ".join(name)
    return {"focal_id": focal_id, "genus": genus, "species_n": species_n,
            "No_obs": float(obs), "No_asc_spp": float(species), "No_redlisted": float(red)}

def read_redspecieslocal_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["LocalityID", "Value"], engine="python")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df.dropna(subset=["LocalityID","Value"])

def parse_allobservations_ids(path: Path) -> List[int]:
    with open(path, "r", encoding="Windows-1252", errors="ignore") as f:
        line = f.readline()
    return list(map(int, re.findall(r"\d+", line)))

def robust_z_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if mad and np.isfinite(mad) and mad > 0:
        return 0.6745 * (x - med) / mad
    std = x.std(skipna=True, ddof=1)
    if std and np.isfinite(std) and std > 0:
        return (x - med) / std
    return pd.Series(np.zeros(len(x)), index=x.index)

def eff_weighted_proportion(events: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    M = np.asarray(events, dtype=float)
    num = (M * w.reshape(-1,1)).sum(axis=0)
    den = w.sum()
    return np.divide(num, den, out=np.zeros_like(num), where=(den > 0))

def eff_counts_for_jeffreys(events: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
    w = np.asarray(weights, dtype=float)
    M = np.asarray(events, dtype=float)
    succ = (M * w.reshape(-1,1)).sum(axis=0)
    Nw = w.sum()
    return succ, Nw

def event_matrix_from_values(values: np.ndarray, max_t: int) -> np.ndarray:
    thresholds = np.arange(max_t + 1, dtype=float).reshape(1, -1)
    return (values.reshape(-1,1) > thresholds)

def stratified_bg_indices(strata_f: pd.Series, strata_bg: pd.Series,
                          rng: np.random.Generator, size_bg:int|None=None) -> np.ndarray:
    comp = strata_f.value_counts(normalize=True, dropna=False)
    uniq = strata_bg.dropna().unique()
    Nb = len(strata_bg)
    size_bg = Nb if size_bg is None else size_bg
    alloc = {s: int(round(p * size_bg)) for s,p in comp.items()}
    for s in comp.index:
        if s not in uniq:
            alloc[s] = 0
    out = []
    for s,n in alloc.items():
        if n <= 0:
            continue
        pool = np.where(strata_bg.values == s)[0]
        if pool.size == 0:
            continue
        draw = rng.integers(0, pool.size, size=n)
        out.append(pool[draw])
    if not out:
        return np.array([], dtype=int)
    return np.concatenate(out)

# ---------- A1: exceedance (redlist baseline) ----------
def baseline_exceedance_weighted(focal_df: pd.DataFrame, bg_df: pd.DataFrame,
                                 thresholds: List[int], tau: float,
                                 alpha_beta=(0.5,0.5),
                                 bootstraps=300, seed=42,
                                 max_t=15, bg_cap=None) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    alpha, beta = alpha_beta

    Vf = focal_df["Value"].to_numpy(float)
    Vb = bg_df["Value"].to_numpy(float)
    Ef = event_matrix_from_values(Vf, max_t=max_t)
    Eb = event_matrix_from_values(Vb, max_t=max_t)
    wf = focal_df["w_inv"].to_numpy(float)
    wb = bg_df["w_inv"].to_numpy(float)
    sf = focal_df["effort_stratum"].to_numpy()
    sb = bg_df["effort_stratum"].to_numpy()

    p_raw = eff_weighted_proportion(Ef, wf)
    q_raw = eff_weighted_proportion(Eb, wb)
    cf, Nf = eff_counts_for_jeffreys(Ef, wf)
    cb, Nb = eff_counts_for_jeffreys(Eb, wb)

    p_sm = (cf + alpha) / (Nf + alpha + beta)
    q_sm = (cb + alpha) / (Nb + alpha + beta)

    loglift = np.full_like(p_sm, np.nan)
    mask = q_sm > 0
    loglift[mask] = np.log2(p_sm[mask] / q_sm[mask])

    Tsel = np.array([t for t in thresholds if 0 <= t <= max_t], dtype=int)
    mean_loglift = float(np.nanmean(loglift[Tsel]))

    p_sel = p_raw[Tsel]; q_sel = q_raw[Tsel]
    mt = p_sel >= tau
    if mt.any():
        lifted_PM_tau = float(np.maximum(0.0, p_sel[mt] - q_sel[mt]).sum())
        frac_enriched = float((p_sel[mt] > q_sel[mt]).sum() / mt.sum())
    else:
        lifted_PM_tau = float("nan")
        frac_enriched = float("nan")

    mll_s, lpm_s, fr_s = [], [], []
    Nb_all = len(Vb)
    bg_draw = min(Nb_all, bg_cap) if (bg_cap is not None) else Nb_all

    for _ in range(int(bootstraps)):
        # bootstrap focal
        pf = wf / wf.sum()
        idx_f = rng.choice(len(Vf), size=len(Vf), replace=True, p=pf)
        Ef_b = Ef[idx_f,:]
        wf_b = wf[idx_f]
        sf_b = sf[idx_f]

        # stratified bg
        idx_bg_pool = stratified_bg_indices(pd.Series(sf_b), pd.Series(sb), rng, size_bg=bg_draw)
        if idx_bg_pool.size == 0:
            pb = wb / wb.sum()
            idx_b = rng.choice(len(Vb), size=bg_draw, replace=True, p=pb)
        else:
            wb_pool = wb[idx_bg_pool]
            pb_pool = wb_pool / wb_pool.sum() if wb_pool.sum() > 0 else None
            idx_b = idx_bg_pool if pb_pool is None else rng.choice(idx_bg_pool, size=idx_bg_pool.size, replace=True, p=pb_pool)
        Eb_b = Eb[idx_b,:]
        wb_b = wb[idx_b]

        pf_raw = eff_weighted_proportion(Ef_b, wf_b)
        qb_raw = eff_weighted_proportion(Eb_b, wb_b)
        cf2, Nf2 = eff_counts_for_jeffreys(Ef_b, wf_b)
        cb2, Nb2 = eff_counts_for_jeffreys(Eb_b, wb_b)
        pf_sm = (cf2 + alpha) / (Nf2 + alpha + beta)
        qb_sm = (cb2 + alpha) / (Nb2 + alpha + beta)

        ll = np.full_like(pf_sm, np.nan)
        m = qb_sm > 0
        ll[m] = np.log2(pf_sm[m] / qb_sm[m])

        mll_s.append(np.nanmean(ll[Tsel]))
        mt = pf_raw[Tsel] >= tau
        if mt.any():
            lpm = float(np.maximum(0.0, pf_raw[Tsel][mt] - qb_raw[Tsel][mt]).sum())
            fr  = float((pf_raw[Tsel][mt] > qb_raw[Tsel][mt]).sum() / mt.sum())
        else:
            lpm, fr = float("nan"), float("nan")
        lpm_s.append(lpm)
        fr_s.append(fr)

    def _ci(a):
        x = np.asarray([z for z in a if np.isfinite(z)])
        if x.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))

    return {
        "mean_loglift": mean_loglift,
        "mean_loglift_CI": _ci(mll_s),
        "lifted_PM_tau": lifted_PM_tau,
        "lifted_PM_tau_CI": _ci(lpm_s),
        "frac_enriched": frac_enriched,
        "frac_enriched_CI": _ci(fr_s),
    }

# ---------- A5: N2000 enrichment as exceedance on n2000_score ----------
def n2000_enrichment_exceedance(focal_ids: pd.Series, lm: pd.DataFrame,
                                thresholds=BASE_THRESHOLDS, tau=TAU_N2000,
                                bootstraps=300, seed=42, bg_cap=4000) -> Dict[str, object]:
    """
    Returns dict with keys:
      mean_loglift, CI, debug
    """
    lm_sub = lm[["LocalityID","n2000_score","w_inv","effort_stratum"]].dropna()
    if lm_sub.empty:
        return {
            "mean_loglift": float("nan"),
            "CI": (float("nan"), float("nan")),
            "debug": "NO_LM_ROWS_WITH_N2000_SCORE",
        }

    focal_ids_unique = focal_ids.dropna().unique()
    F = lm_sub[lm_sub["LocalityID"].isin(focal_ids_unique)]
    BG= lm_sub[~lm_sub["LocalityID"].isin(focal_ids_unique)]

    if F.empty:
        return {
            "mean_loglift": float("nan"),
            "CI": (float("nan"), float("nan")),
            "debug": f"NO_FOCAL_LOCALITIES_IN_LM (n_ids={len(focal_ids_unique)})",
        }
    if BG.empty:
        return {
            "mean_loglift": float("nan"),
            "CI": (float("nan"), float("nan")),
            "debug": "NO_BACKGROUND_LOCALITIES_IN_LM",
        }

    res = baseline_exceedance_weighted(
        F.rename(columns={"n2000_score":"Value"}),
        BG.rename(columns={"n2000_score":"Value"}),
        thresholds=thresholds, tau=tau, alpha_beta=ALPHA_BETA,
        bootstraps=bootstraps, seed=seed, max_t=15, bg_cap=bg_cap
    )
    return {
        "mean_loglift": res["mean_loglift"],
        "CI": res["mean_loglift_CI"],
        "debug": f"OK (F={len(F)}, BG={len(BG)})",
    }

# ---------- Jenks Natural Breaks ----------
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

# ---------- Write CONFIG summary ----------
def write_config_summary(output_dir: Path):
    cfg_path = output_dir / f"CONFIG_{RUN_ID}.txt"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("indicator_baseline_composite CONFIGURATION\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"BASE_FOLDER:      {BASE_FOLDER}\n")
        f.write(f"LOCALITY_MASTER:  {LOCALITY_MASTER}\n")
        f.write(f"OUTPUT_DIR:       {OUTPUT_DIR}\n")
        f.write(f"HABITAT_FILE:     {HABITAT_FILE}\n\n")

        f.write("BASE_THRESHOLDS:  " + ", ".join(map(str, BASE_THRESHOLDS)) + "\n")
        f.write(f"TAU_IND:          {TAU_IND}\n")
        f.write(f"TAU_N2000:        {TAU_N2000}\n")
        f.write(f"ALPHA_BETA:       {ALPHA_BETA}\n")
        f.write(f"MIN_N_LOCALITIES: {MIN_N_LOCALITIES}\n")
        f.write(f"BOOTSTRAPS:       {BOOTSTRAPS}\n")
        f.write(f"BG_BOOT_CAP:      {BG_BOOT_CAP}\n")
        f.write(f"RANDOM_SEED:      {RANDOM_SEED}\n\n")

        f.write("Composite weights (A1, A5, Habitat purity):\n")
        f.write(f"  W_EXCEED = {W_EXCEED}\n")
        f.write(f"  W_N2000  = {W_N2000}\n")
        f.write(f"  W_HAB    = {W_HAB}\n\n")

        f.write(f"Jenks classes (k): {JENKS_K} (only RedIndVal_star2 >= 0, first break at 0)\n\n")
        f.write(f"RUN_ID: {RUN_ID}\n")
    print(f"[INFO] Configuration summary written to {cfg_path}")

# ---------- main ----------
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    write_config_summary(OUTPUT_DIR)

    # Load LocalityMaster
    lm = pd.read_csv(LOCALITY_MASTER, sep="\t")
    for c in ["w_inv","effort_stratum","n_records","n_species",
              "n_species_N2K","n2000_score"]:
        if c in lm.columns:
            lm[c] = pd.to_numeric(lm[c], errors="coerce")
    if "dominant_habitat" not in lm.columns:
        lm["dominant_habitat"] = "Unknown"
    lm = lm.dropna(subset=["LocalityID","w_inv","effort_stratum"])

    # Habitat assignments
    habitat_map = build_habitat_map(HABITAT_FILE)
    if habitat_map is None:
        print("[WARN] Habitat assignments unavailable; using 'Unknown' for all species.")

    folder_dfs: Dict[str, pd.DataFrame] = {}
    meta_rows = []

    subfolders = [p for p in BASE_FOLDER.iterdir() if p.is_dir()]
    print(f"Starting indicator baseline composite on {len(subfolders)} focal folders...")

    # FIRST PASS: preload focal locality tables + meta
    for fldr in subfolders:
        try:
            rsl = fldr / "RedSpeciesLocal.txt"
            tto = fldr / "totalTargetObs.txt"
            aob = fldr / "allObservations.txt"
            if not (rsl.exists() and tto.exists() and aob.exists()):
                continue

            df_r = read_redspecieslocal_df(rsl).merge(
                lm[["LocalityID","w_inv","effort_stratum","dominant_habitat"]],
                on="LocalityID", how="left"
            ).dropna(subset=["w_inv","effort_stratum"])
            if df_r.empty:
                continue
            folder_dfs[str(fldr)] = df_r

            meta_rec = parse_total_target_obs(tto)
            meta_rec["N_localities"] = float(len(df_r))
            meta_rows.append({
                "focal_id": meta_rec["focal_id"],
                "No_obs": meta_rec["No_obs"],
                "N_localities": meta_rec["N_localities"],
                "No_redlisted": meta_rec["No_redlisted"],
            })
        except Exception as e:
            print(f"[WARN] Skipping {fldr}: {e}")

    meta_all = pd.DataFrame(meta_rows).dropna()
    if not meta_all.empty:
        meta_all = (meta_all.groupby("focal_id", as_index=False)
                            .agg({"No_obs":"max","N_localities":"max","No_redlisted":"max"}))
        meta_all.index = meta_all["focal_id"]

    # SECOND PASS: compute per-focal components
    rows_out = []
    for fldr in subfolders:
        try:
            key = str(fldr)
            if key not in folder_dfs:
                continue
            dfF = folder_dfs[key].copy()
            tto = fldr / "totalTargetObs.txt"
            aob = fldr / "allObservations.txt"
            meta_f = parse_total_target_obs(tto)
            focal_id = meta_f["focal_id"]

            # choose main habitat from species habitat table
            main_hab = infer_habitat_for(focal_id, habitat_map)

            # restrict LocalityMaster to same habitat (for A5); fall back if empty
            lm_hab = lm.copy()
            if "dominant_habitat" in lm_hab.columns and isinstance(main_hab, str) and main_hab != "Unknown":
                lm_hab = lm_hab[lm_hab["dominant_habitat"] == main_hab]
                if lm_hab.empty:
                    print(f"[WARN] No LM rows for habitat '{main_hab}' – falling back to all habitats.")
                    lm_hab = lm
            else:
                lm_hab = lm

            # background localities for A1 (other focal species, same habitat)
            dfBG = pd.concat([v for k,v in folder_dfs.items() if k != key], ignore_index=True)
            if not dfBG.empty and isinstance(main_hab, str):
                dfBG = dfBG[dfBG["dominant_habitat"] == main_hab]

            N_focal_localities = len(dfF)
            N_bg_localities    = len(dfBG)
            N_LM_hab_localities= len(lm_hab)

            # Habitat purity: share of observed species with habitat == main_hab
            purity = float("nan")
            try:
                ids = parse_allobservations_ids(aob)
                if habitat_map is not None and len(ids) > 0 and isinstance(main_hab, str):
                    df_ids = pd.DataFrame({"taxon_id": sorted(set(ids))})
                    hb = habitat_map[["taxon_id","habitat"]].dropna(subset=["taxon_id"])
                    hit = df_ids.merge(hb, on="taxon_id", how="left")["habitat"]
                    known = hit.dropna()
                    if known.size > 0:
                        purity = float((known == main_hab).mean())
            except Exception:
                pass
            
            # --- Locality subsampling stability from LocalitySpecies_ALL.txt ---
            lsa_file = fldr / "LocalitySpecies_ALL.txt"
            stable = int(hashlib.md5(fldr.name.encode("utf-8")).hexdigest()[:8], 16)
            seed = LSA_SEED + stable
            lsa = locality_subsample_alpha_gamma(
                lsa_file,
                repeats=LSA_REPEATS,
                seed=seed,
            )

            # --- A1: redlist exceedance ---
            A1 = {
                "mean_loglift": float("nan"),
                "mean_loglift_CI": (float("nan"),float("nan")),
                "lifted_PM_tau": float("nan"),
                "lifted_PM_tau_CI": (float("nan"),float("nan")),
                "frac_enriched": float("nan"),
                "frac_enriched_CI": (float("nan"),float("nan")),
            }
            A1_debug = "NOT_ATTEMPTED"
            if N_focal_localities < MIN_N_LOCALITIES:
                A1_debug = f"TOO_FEW_FOCAL_LOCALITIES ({N_focal_localities} < {MIN_N_LOCALITIES})"
            elif N_bg_localities == 0:
                A1_debug = "NO_BACKGROUND_LOCALITIES_SAME_HABITAT"
            else:
                try:
                    A1 = baseline_exceedance_weighted(
                        dfF, dfBG,
                        thresholds=BASE_THRESHOLDS, tau=TAU_IND,
                        alpha_beta=ALPHA_BETA, bootstraps=BOOTSTRAPS,
                        seed=RANDOM_SEED, max_t=15, bg_cap=BG_BOOT_CAP
                    )
                    A1_debug = f"OK (F={N_focal_localities}, BG={N_bg_localities})"
                except Exception as e:
                    A1_debug = f"ERROR: {e.__class__.__name__}"
                    print(f"[DEBUG] A1 failed for {focal_id}: {e}")
                    traceback.print_exc()

            if not A1_debug.startswith("OK") and not A1_debug.startswith("ERROR"):
                print(f"[DEBUG] A1 not computed for {focal_id}: {A1_debug}")


            # --- A5: N2000 enrichment (exceedance on n2000_score) ---
            A5 = {"mean_loglift": float("nan"), "CI": (float("nan"), float("nan")), "debug": "NOT_ATTEMPTED"}
            A5_debug = "NOT_ATTEMPTED"
            if N_focal_localities < MIN_N_LOCALITIES:
                A5_debug = f"TOO_FEW_FOCAL_LOCALITIES ({N_focal_localities} < {MIN_N_LOCALITIES})"
            else:
                try:
                    A5 = n2000_enrichment_exceedance(
                        dfF["LocalityID"], lm_hab,
                        thresholds=BASE_THRESHOLDS, tau=TAU_N2000,
                        bootstraps=BOOTSTRAPS, seed=RANDOM_SEED,
                        bg_cap=BG_BOOT_CAP
                    )
                    A5_debug = A5.get("debug", "OK")
                except Exception as e:
                    A5_debug = f"ERROR: {e.__class__.__name__}"
                    print(f"[DEBUG] A5 failed for {focal_id}: {e}")
                    traceback.print_exc()

            if not A5_debug.startswith("OK"):
                print(f"[DEBUG] A5 status for {focal_id}: {A5_debug}")

            row = {
                "focal_id": focal_id,
                "main_habitat": main_hab,
                "No_obs": float(meta_f["No_obs"]),
                "No_asc_spp": float(meta_f["No_asc_spp"]),
                "No_redlisted": float(meta_f["No_redlisted"]),
                "Habitat_purity": purity,
                **lsa,

                # A1
                "mean_loglift_0to4": A1["mean_loglift"],
                "mean_loglift_0to4_CI_low": A1["mean_loglift_CI"][0],
                "mean_loglift_0to4_CI_high": A1["mean_loglift_CI"][1],
                "lifted_PM_tau0.30": A1["lifted_PM_tau"],
                "lifted_PM_tau0.30_CI_low": A1["lifted_PM_tau_CI"][0],
                "lifted_PM_tau0.30_CI_high": A1["lifted_PM_tau_CI"][1],
                "frac_enriched_tau0.30": A1["frac_enriched"],
                "frac_enriched_tau0.30_CI_low": A1["frac_enriched_CI"][0],
                "frac_enriched_tau0.30_CI_high": A1["frac_enriched_CI"][1],

                # A5 (N2000 enrichment)
                "N2000_mean_loglift_0to4": A5["mean_loglift"],
                "N2000_CI_low": A5["CI"][0],
                "N2000_CI_high": A5["CI"][1],

                # counts + debug
                "N_focal_localities": N_focal_localities,
                "N_background_localities": N_bg_localities,
                "N_LM_hab_localities": N_LM_hab_localities,
                "A1_debug": A1_debug,
                "A5_debug": A5_debug,
            }

            row_stream = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "folder": fldr.name,
            }
            row_stream.update(row)
            _write_stream_row(STREAM_PATH, STREAM_COLUMNS, row_stream)
            rows_out.append(row)

            print(
                f"[OK] {fldr.name}: "
                f"A1={row['mean_loglift_0to4']:.3f}  "
                f"A5(N2K)={row['N2000_mean_loglift_0to4']:.3f}  "
                f"HAB_purity={purity:.3f}"
            )

        except Exception as e:
            print(f"[ERR] {fldr.name}: {e}")
            traceback.print_exc()

    # Build DataFrame
    res = pd.DataFrame(rows_out)
    if res.empty:
        out_path = OUTPUT_DIR / "FINAL_results_composite_batch_3.txt"
        res.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote {out_path} with 0 rows.")
        return

    # ---- Per-habitat grouped robust z-scores (only A1, A5, habitat purity) ----
    def z_by_group_forgiving(df: pd.DataFrame, col: str) -> pd.Series:
        out = pd.Series(index=df.index, dtype=float)
        if "main_habitat" not in df.columns:
            return robust_z_series(df[col])
        for hab, sub in df.groupby("main_habitat"):
            vals = sub[col]
            if vals.notna().sum() < 5:
                out.loc[sub.index] = robust_z_series(df[col]).loc[sub.index]
            else:
                out.loc[sub.index] = robust_z_series(vals).values
        return out

    res["Z_exceed"]     = z_by_group_forgiving(res, "mean_loglift_0to4")
    res["Z_N2000"]      = z_by_group_forgiving(res, "N2000_mean_loglift_0to4")
    res["Z_HAB_purity"] = z_by_group_forgiving(res, "Habitat_purity")

    # Row-wise composite ONLY when all 3 metrics are present
    Z_cols = ["Z_exceed","Z_N2000","Z_HAB_purity"]
    W = np.array([W_EXCEED, W_N2000, W_HAB], dtype=float)

    Z = res[Z_cols].to_numpy(dtype=float)
    mask_all = np.isfinite(Z).all(axis=1)   # all three present
    composite = np.full(len(res), np.nan, dtype=float)
    if mask_all.any():
        composite[mask_all] = (Z[mask_all] * W).sum(axis=1) / W.sum()

    res["RedIndVal_star2"] = composite

    # Jenks Natural Breaks classification on composite score, only for >= 0
    res["RedIndVal_JenksClass"] = jenks_classify_nonnegative(res["RedIndVal_star2"], k=JENKS_K)

    # Write final
    out_path = OUTPUT_DIR / "FINAL_results_composite_2_batch_3.txt"
    res.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path} with {len(res)} rows.")

    # by-habitat sanity summary
    (OUTPUT_DIR / "by_habitat_summary.txt").write_text(
        res.groupby("main_habitat")["RedIndVal_star2"].describe().to_string(),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
