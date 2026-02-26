from pathlib import Path
import polars as pl
import numpy as np
from typing import Optional
import re 

# ---------------- CONFIG: EDIT THESE ----------------
PARQUET_DIR = Path("/home/occurrence_ds/")   # folder with parquet parts
OUT         = Path("/home/")          # where to write the txt files
META_DIR    = Path("/home/metadata/")        # your metadata folder

IAS_FILE = META_DIR / "ias_bank.csv"       # format: taxonID;scientificName;weight
N2K_FILE = META_DIR / "n2000_bank.csv"     # format: taxonID;scientificName;habitatCode;weight

GRID_DECIMALS = 3  # rounding for cell grid (e.g., 61.701_16.162)

# ----------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)

def _hab_norm(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z]+", "_", s)       # non-letters -> underscore
    s = re.sub(r"_+", "_", s).strip("_") # collapse and trim
    return s or "unknown"

def read_bank_csv(path: Path, cols):
    if not path.exists():
        # empty DF with the expected schema (prevents crashes)
        return pl.DataFrame({c: [] for c in cols})
    # your banks use semicolon (;)
    df = pl.read_csv(path, separator=";", has_header=True, ignore_errors=True)
    # keep only known columns, cast softly
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df.select([pl.col(c) for c in cols])

# ---------- read banks (EAGER) then convert to LAZY ----------
ias_bank = read_bank_csv(IAS_FILE, ["taxonID", "scientificName", "weight"]).with_columns([
    pl.col("taxonID").cast(pl.Utf8).str.extract(r"(\d+)$", 1).cast(pl.Int64, strict=False),
    pl.col("weight").cast(pl.Float64, strict=False).fill_null(1.0),
])
n2k_bank = read_bank_csv(N2K_FILE, ["taxonID", "scientificName", "habitatCode", "weight"]).with_columns([
    pl.col("taxonID").cast(pl.Utf8).str.extract(r"(\d+)$", 1).cast(pl.Int64, strict=False),
    pl.col("weight").cast(pl.Float64, strict=False).fill_null(1.0),
    pl.col("habitatCode").cast(pl.Utf8, strict=False),
])

ias_bank_lf = (
    ias_bank.lazy()
    .select([
        pl.col("taxonID").alias("taxon_id").cast(pl.Int64, strict=False),
        pl.col("weight").alias("ias_weight").cast(pl.Float64, strict=False),
    ])
)

n2k_bank_lf = (
    n2k_bank.lazy()
    .select([
        pl.col("taxonID").alias("taxon_id").cast(pl.Int64, strict=False),
        pl.col("weight").alias("n2k_weight").cast(pl.Float64, strict=False),
    ])
)

# ---------- scan parquet (LAZY) ----------
try:
    scan = pl.scan_parquet(str(PARQUET_DIR / "**" / "*.parquet"), recursive=True)
except TypeError:
    # older Polars: no recursive=, do manual list
    files = [str(p) for p in PARQUET_DIR.rglob("*.parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files under {PARQUET_DIR}")
    scan = pl.concat([pl.scan_parquet(p) for p in files])

cols = scan.collect_schema().names()
if not {"decimalLatitude", "decimalLongitude", "eventDate"}.issubset(set(cols)):
    missing = {"decimalLatitude", "decimalLongitude", "eventDate"} - set(cols)
    raise RuntimeError(f"Missing columns in parquet: {missing}")

# Prefer a numeric taxon id if present
if "taxonID_num" in cols:
    taxon_expr = pl.col("taxonID_num").cast(pl.Int64, strict=False).alias("taxon_id")
else:
    taxon_expr = (
        pl.col("taxonID")
        .cast(pl.Utf8, strict=False)
        .str.extract(r"(\d+)$", 1)
        .cast(pl.Int64, strict=False)
        .alias("taxon_id")
    )

lat_r = pl.col("decimalLatitude").cast(pl.Float64).round(GRID_DECIMALS).alias("lat_r")
lon_r = pl.col("decimalLongitude").cast(pl.Float64).round(GRID_DECIMALS).alias("lon_r")
loc_id = pl.concat_str([pl.lit("cell_"), lat_r.cast(pl.Utf8), pl.lit("_"), lon_r.cast(pl.Utf8)]).alias("LocalityID")

# Normalize eventDate (YYYY[-MM[-DD]])
date_tok = pl.col("eventDate").cast(pl.Utf8, strict=False).str.extract(r"(\d{4}(?:-\d{2}){0,2})", 1)
date_s = (
    pl.when(date_tok.str.len_chars() == 4).then(date_tok + "-01-01")
     .when(date_tok.str.len_chars() == 7).then(date_tok + "-01")
     .otherwise(date_tok)
     .alias("date_s")
)

# Base lazy frame: keep only what we need
base = scan.select(pl.struct([loc_id, lat_r, lon_r, date_s, taxon_expr]).alias("r")).unnest("r")

# ---------- LocalityEvents.txt (per day × locality) ----------
events_lazy = (
    base
    .filter(pl.col("date_s").is_not_null())
    .group_by(["LocalityID", "date_s"])
    .agg([
        pl.len().alias("n_records"),
        pl.col("taxon_id").n_unique().alias("n_species"),
    ])
    .with_columns((pl.col("LocalityID") + pl.lit("_") + pl.col("date_s")).alias("checklist_id"))
    .select(["LocalityID", "date_s", "checklist_id", "n_records", "n_species"])
)
events_lazy.sink_csv(OUT / "LocalityEvents.txt", separator="\t")

# ---------- LocalityMaster core (LAZY → EAGER) ----------
loc_lazy = (
    base
    .group_by("LocalityID")
    .agg([
        pl.col("lat_r").median().alias("lat_r"),
        pl.col("lon_r").median().alias("lon_r"),
        pl.col("date_s").filter(pl.col("date_s").is_not_null()).min().alias("first_event"),
        pl.col("date_s").filter(pl.col("date_s").is_not_null()).max().alias("last_event"),
        pl.len().alias("n_records"),
        pl.col("date_s").filter(pl.col("date_s").is_not_null()).n_unique().alias("n_events"),
        pl.col("taxon_id").n_unique().alias("n_species"),
    ])
    .with_columns(
        (
            0.5 * pl.col("n_events") +
            0.5 * pl.when(pl.col("n_events") > 0)
                   .then(pl.col("n_species") / pl.col("n_events"))
                   .otherwise(0.0)
        ).alias("effort_index")
    )
)
loc_df = loc_lazy.collect()

# ---------- IAS / N2000 flags & per-locality aggregates (ALL LAZY) ----------
# Join with IAS and N2K banks lazily; add boolean flags
base_flags = (
    base
    .join(ias_bank_lf, on="taxon_id", how="left")
    .with_columns(pl.col("ias_weight").is_not_null().alias("is_IAS"))
    .join(n2k_bank_lf, on="taxon_id", how="left")
    .with_columns(pl.col("n2k_weight").is_not_null().alias("is_N2K"))
)

# Count unique IAS species per locality
ias_counts_lazy = (
    base_flags
    .filter(pl.col("is_IAS") == True)
    .group_by("LocalityID")
    .agg(pl.col("taxon_id").n_unique().alias("n_species_IAS"))
)

# N2000: for each locality, sum max weight per (LocalityID, taxon_id) to ensure uniqueness
n2k_score_lazy = (
    base_flags
    .filter(pl.col("is_N2K") == True)
    .group_by(["LocalityID", "taxon_id"])
    .agg(pl.col("n2k_weight").max().alias("w"))
    .group_by("LocalityID")
    .agg([
        pl.col("w").sum().alias("n2000_score"),
        pl.len().alias("n_species_N2K"),
    ])
)

# Collect lazy aggregates to eager
ias_counts = ias_counts_lazy.collect()
n2k_score  = n2k_score_lazy.collect()

# Merge IAS/N2K aggregates into loc_df
loc_df = loc_df.join(ias_counts, on="LocalityID", how="left")
loc_df = loc_df.join(n2k_score,  on="LocalityID", how="left")

# Fill / cast types for new columns
loc_df = loc_df.with_columns([
    pl.col("n_species_IAS").fill_null(0).cast(pl.Int64),
    pl.col("n_species_N2K").fill_null(0).cast(pl.Int64),
    pl.col("n2000_score").fill_null(0.0).cast(pl.Float64),
])

# ---------- dominant habitat per locality (via species' habitat labels) ----------
# 1) Load habitat assignments (taxonID; scientificName; habitat), robust to encoding.
def _read_habitat_assignments_csv(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing habitat_assignments.csv at {path}")
        return None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pl.read_csv(path, separator=";", has_header=True, encoding=enc, ignore_errors=True)
            break
        except Exception:
            df = None
    if df is None:
        print("[WARN] Could not decode habitat_assignments.csv; skipping dominant habitat.")
        return None
    # normalize columns
    for c in ("taxonID", "scientificName", "habitat"):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df.select(["taxonID", "habitat"]) \
             .with_columns([
                 pl.col("taxonID").cast(pl.Utf8, strict=False)
                                   .str.extract(r"(\d+)$", 1)
                                   .cast(pl.Int64, strict=False)
                                   .alias("taxon_id"),
                 pl.col("habitat").cast(pl.Utf8, strict=False),
             ]) \
             .drop_nulls(subset=["taxon_id"])
    hab = _read_habitat_assignments_csv(HAB_PATH)
    if hab is not None:
        hab = hab.with_columns(pl.col("habitat").map_elements(_hab_norm, return_dtype=pl.Utf8))


HAB_PATH = Path("/home/eric/indicator/metadata/habitat_assignments.csv")
hab = _read_habitat_assignments_csv(HAB_PATH)

if hab is not None and "taxon_id" in hab.columns:
    # 2) For each LocalityID, count species per habitat, pick the argmax as dominant.
    #    We reuse the already built lazy scan 'scan' (same taxon_id / LocalityID build you use above).
    #    Build a locality×species table and join to habitat.
    # ---------- dominant habitat per locality (via species' habitat labels) ----------
    # base already has LocalityID and taxon_id
    sp_loc = (
        base
        .select(["LocalityID", "taxon_id"])
        .unique()  # unique species per locality
        .join(hab.lazy(), on="taxon_id", how="left")
        .group_by(["LocalityID", "habitat"])
        .agg(pl.len().alias("n_sp_hab"))  # pl.count() -> pl.len()
        .collect()
    )

    dom = (
        sp_loc
        .sort(["LocalityID", "n_sp_hab"], descending=[False, True])
        .group_by("LocalityID")
        .agg([
            pl.col("habitat").first().alias("dominant_habitat"),
            (pl.col("n_sp_hab") / pl.col("n_sp_hab").sum()).max().alias("dominant_habitat_purity"),
        ])
        .with_columns([
            pl.col("dominant_habitat").fill_null("Unknown").cast(pl.Utf8),
            pl.col("dominant_habitat_purity").fill_null(0.0).cast(pl.Float64),
        ])
    )

    loc_df = loc_df.join(dom, on="LocalityID", how="left").with_columns([
        pl.col("dominant_habitat").fill_null("Unknown"),
        pl.col("dominant_habitat_purity").fill_null(0.0),
    ])
if hab is not None and hab.height > 0:
    # build sp_loc and dom (your current code) …
    sp_loc = (
        base
        .select(["LocalityID", "taxon_id"])
        .unique()
        .join(hab.lazy(), on="taxon_id", how="left")
        .group_by(["LocalityID", "habitat"])
        .agg(pl.len().alias("n_sp_hab"))
        .collect()
    )

    dom = (
        sp_loc
        .sort(["LocalityID", "n_sp_hab"], descending=[False, True])
        .group_by("LocalityID")
        .agg([
            pl.col("habitat").first().alias("dominant_habitat"),
            (pl.col("n_sp_hab") / pl.col("n_sp_hab").sum()).max().alias("dominant_habitat_purity"),
        ])
        .with_columns([
            pl.col("dominant_habitat").fill_null("unknown").cast(pl.Utf8),
            pl.col("dominant_habitat_purity").fill_null(0.0).cast(pl.Float64),
        ])
        # normalize the chosen dominant label too
        .with_columns(pl.col("dominant_habitat")
                      .map_elements(_hab_norm, return_dtype=pl.Utf8))
    )

    loc_df = loc_df.join(dom, on="LocalityID", how="left")

    # ensure columns exist even if hab is None or dom is empty
    if "dominant_habitat" not in loc_df.columns:
        loc_df = loc_df.with_columns([
            pl.lit("unknown").alias("dominant_habitat"),
            pl.lit(0.0).alias("dominant_habitat_purity"),
        ])
    else:
        loc_df = loc_df.with_columns([
            pl.col("dominant_habitat").fill_null("unknown"),
            pl.col("dominant_habitat_purity").fill_null(0.0),
        ])

else:
    # keep columns to avoid downstream KeyErrors
    loc_df = loc_df.with_columns([
        pl.lit("Unknown").alias("dominant_habitat"),
        pl.lit(0.0).alias("dominant_habitat_purity"),
    ])

# ---------- derive non-IAS, IAS share ----------
loc_df = loc_df.with_columns([
    # n_species_nonIAS = n_species - n_species_IAS (not below 0)
    (pl.col("n_species") - pl.col("n_species_IAS")).clip(lower_bound=0).alias("n_species_nonIAS"),
    # IAS_share_species = n_species_IAS / n_species
    pl.when(pl.col("n_species") > 0)
      .then(pl.col("n_species_IAS") / pl.col("n_species"))
      .otherwise(0.0)
      .alias("IAS_share_species")
])

# ---------- effort weights & strata ----------
if loc_df.height:
    # 1) Keep inverse weights on the ORIGINAL scale (unchanged)
    med_ei = float(loc_df["effort_index"].median())
    w_inv = (pl.lit(med_ei) / pl.when(pl.col("effort_index") > 0)
                               .then(pl.col("effort_index"))
                               .otherwise(pl.lit(np.nan)))
    loc_df = loc_df.with_columns(
        w_inv.fill_nan(1.0).clip(lower_bound=0.2, upper_bound=5.0).alias("w_inv")
    )

    # 2) Add a LOG-transformed effort index (for diagnostics & transparency)
    #    log1p is monotonic: it compresses the huge tail but preserves order.
    loc_df = loc_df.with_columns(
        pl.col("effort_index").log1p().alias("effort_index_log")
    )

    # 3) Deciles by stable RANK (0..9), computed on the log scale (order is same as original,
    #    but storing 'effort_index_log' helps interpret decile ranges later).
    ei = loc_df["effort_index_log"].to_numpy()

    # Guard against non-finite (shouldn't occur here, but safe)
    if not np.isfinite(ei).any():
        bins = np.full(ei.size, 5, dtype=int)  # middle bin if degenerate
        edges = np.linspace(0.0, float(ei.size), 11)
    else:
        med = float(np.nanmedian(ei[np.isfinite(ei)]))
        ei = np.where(np.isfinite(ei), ei, med)

        # Stable rank 1..N (mergesort preserves original order for ties)
        order = np.argsort(ei, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, ei.size + 1, dtype=np.float64)

        # Deciles
        q = 10
        edges = np.linspace(0.0, float(ei.size), q + 1)
        cuts = edges[1:-1]
        bins = np.clip(np.digitize(ranks, cuts, right=True), 0, q - 1).astype(int)

    # 4) Write back: effort_stratum now 0..9
    loc_df = loc_df.with_columns([
        pl.Series("effort_stratum", bins),
        pl.lit("rank_deciles_log1p").alias("effort_strata_method"),
        pl.lit(",".join(f"{int(x)}" for x in edges)).alias("effort_strata_edges_rank")
    ])

    # 5) Quick sanity print (optional; fine to keep)
    counts = (loc_df.group_by("effort_stratum")
                    .len()
                    .sort("effort_stratum")
                    .to_dict(as_series=False))
    print("[effort strata] decile counts:", counts)
    
# ---------- Margalef proxies (all species vs IAS-free) ----------
#   (S-1)/ln(n) with safe NaN handling
n_rec  = loc_df["n_records"].to_numpy()
S_all  = loc_df["n_species"].to_numpy()
S_ex   = loc_df["n_species_nonIAS"].to_numpy()

with np.errstate(divide="ignore", invalid="ignore"):
    M_all = (S_all - 1) / np.log(n_rec)
    M_ex  = (S_ex  - 1) / np.log(n_rec)
    M_all[~np.isfinite(M_all)] = np.nan
    M_ex[~np.isfinite(M_ex)]   = np.nan

loc_df = loc_df.with_columns([
    pl.Series("Margalef_local",  M_all),
    pl.Series("Margalef_exIAS",  M_ex),
])

# ---------- placeholders for columns you may join later ----------
loc_df = loc_df.with_columns([
    pl.lit(None, dtype=pl.Utf8).alias("SWEREF_N"),
    pl.lit(None, dtype=pl.Utf8).alias("SWEREF_E"),
    pl.lit(None, dtype=pl.Utf8).alias("county_id"),
])

# ---------- write outputs ----------
events_out = OUT / "LocalityEvents.txt"
master_out = OUT / "LocalityMaster.txt"

events_lazy.sink_csv(events_out, separator="\t")
loc_df.select([
    "LocalityID",
    "SWEREF_N","SWEREF_E","lat_r","lon_r","county_id",
    "first_event","last_event",
    "n_records","n_events","n_species",
    "n_species_IAS","n_species_nonIAS","IAS_share_species",
    "n_species_N2K","n2000_score",
    "effort_index","w_inv","effort_stratum",
    "Margalef_local","Margalef_exIAS",
    "dominant_habitat","dominant_habitat_purity",
]).write_csv(master_out, separator="\t")

print(f"Wrote: {events_out}")
print(f"Wrote: {master_out}")
