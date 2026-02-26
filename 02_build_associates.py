"""
Associate extraction for focal species using buffered occurrences.

Inputs
------
- A folder of Parquet "parts" created from DwC-A (raw columns, NOT yet GeoParquet),
  typically: r"C:\...\occurrence_ds\part-00001.parquet, part-00002.parquet" ... # type: ignore # type: ignore # pyright: ignore[reportInvalidStringEscapeSequence] # type: ignore
- A national red-list CSV (semicolon-delimited) with a column "Taxon id" holding
  numeric Dyntaxa IDs (e.g., 227592), and a column "RedListCategory" or "Kategori".

Outputs (per focal species, into its own folder)
------------------------------------------------
- uniqueFinds.txt            (focal name + unique associate taxonIDs, once each)
- totalTargetObs.txt         (focal name + counts: focal_obs, all_assoc_obs, red_assoc_obs)
- redFinds.txt               (focal name + unique red-listed associates)
- allObservations.txt        (focal name + all associate taxonIDs incl. duplicates)
- RedSpeciesLocal.txt        (one per line: <LocalityKey> <count_of_redlisted_associates>)

Notes
-----
- Buffer radius is in meters; CRS conversion is done on the fly (4326 -> 3006).
- "LocalityKey" prefers 'localityID' if present; otherwise falls back to 'locality' text.
- Focal occurrence count is computed across the whole dataset (not just the sampled 350).
- Associates are all *other* species whose points intersect any focal buffer.
"""

from pathlib import Path
import re
import time
from datetime import datetime
import unicodedata
from collections import Counter
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
import csv

# =============== CONFIG =====================
OCCURRENCE_DIR = r"/home/occurrence_ds/"          # Path to folder with parquet parts
REDLIST_CSV    = r"/home/metadata/swedish_red-list.csv"   # Path to red-list CSV
OUTPUT_ROOT    = r"/home/OUT_datafolders/"                 # Path to output folder - do not change!         

# --- Geographic stratification (GeoPackage) ---
# A GeoPackage can contain one layer with multiple polygons. Select the region by attribute value.
GEOREGIONS_GPKG  = r"/home/metadata/georegions.gpkg"  # path to GeoPackage with polygons
GEOREGIONS_LAYER = None       # e.g. "georegions" if your gpkg has multiple layers; None = auto-pick first layer
GEOREGION_FIELD  = "region_ID"   # attribute field holding labels like "alpine", "northern", "southern"
GEOREGION_CHOICE = "southern" # one of: "alpine", "northern", "southern"
# If True, points must intersect the chosen region polygons (overlaps are OK).
APPLY_GEO_FILTER = True
# ------------------------------------------------
# Focal species selectors (set ONE of these; both allowed)
FOCAL_TAXONID_NUMS = []  # e.g., [227592, 6003469]
FOCAL_SCI_NAMES    = []  # scientificName exact match(es)
FOCAL_CSV          = r"/home/grandlist1.csv"     # Path to taxon list.

BUFFER_M       = 50      # buffer radius in meters
SAMPLE_SITES   = 350     # max number of focal occurrences (sites) to buffer; if fewer exist, use all
TARGET_CRS     = 3006    # SWEREF 99 TM
SWEDEN_BBOX    = (10.0, 55.0, 25.0, 69.5)  # lon_min, lat_min, lon_max, lat_max (coarse sanity filter)

# Column names expected in the DwC-A parts (coming from your converter)
COL_LON        = "decimalLongitude"
COL_SPECIES_ID  = "taxonID_num"
COL_EVENTDATE  = "eventDate" 
COL_LAT        = "decimalLatitude"
COL_TAXONID    = "taxonID"            # LSID string
COL_SCI        = "scientificName"
COL_VERN       = "vernacularName"
COL_OCCID      = "occurrenceID"
COL_LOCID      = "localityID"         # optional
COL_LOCALITY   = "locality"           # optional
# ========================================================

def read_focal_species_from_csv(csv_path: str):
    """
    Read focal species from a CSV/TSV with columns like:
      - taxonID (or TaxonID / taxonId / id)
      - name (or scientificName)
    Delimiter is auto-detected among , ; and tab.
    """
    ids, names = [], []
    p = Path(csv_path)
    if not p.is_file():
        print(f"[WARN] No CSV found at {csv_path}")
        return ids, names

    with p.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel  # default to comma
        reader = csv.DictReader(f, dialect=dialect)

        # normalize possible header name variants
        def get_any(row, *keys):
            for k in keys:
                if k in row and row[k] not in (None, ""):
                    return row[k]
            return None

        for row in reader:
            tid = get_any(row, "taxonID", "TaxonID", "taxonId", "id")
            nm  = get_any(row, "name", "scientificName", "scientific_name")
            if tid:
                try:
                    ids.append(int(str(tid).strip()))
                except ValueError:
                    print(f"[WARN] Could not parse taxonID '{tid}'")
            if nm:
                names.append(str(nm).strip())

    # dedupe & sort
    ids = sorted(set(ids))
    names = sorted(set(names))
    print(f"[FOCAL] Loaded {len(ids)} taxonIDs and {len(names)} names from CSV.")
    # optional: show the first few for sanity
    print("  IDs head:", ids[:5])
    print("  Names head:", names[:5])
    return ids, names

# ---------- ENTRY POINT ----------
def get_focal_species():
    ids, names = [], []

    # Add from config lists
    ids.extend(FOCAL_TAXONID_NUMS)
    names.extend(FOCAL_SCI_NAMES)

    # Add from CSV if provided
    if FOCAL_CSV:
        csv_ids, csv_names = read_focal_species_from_csv(FOCAL_CSV)
        ids.extend(csv_ids)
        names.extend(csv_names)

    # Deduplicate
    ids = sorted(set(ids))
    names = sorted(set(names))
    return ids, names


def sanitize_filename(name: str) -> str:
    if not isinstance(name, str):
        name = str(name) if name is not None else "unknown"
    # normalize (NFC keeps å/ä/ö as single characters)
    name = unicodedata.normalize("NFC", name.strip())
    # replace spaces with underscore
    name = name.replace(" ", "_")
    # allow letters, digits, underscore, dash, dot — incl. åäöÅÄÖ
    allowed = r"[^A-Za-z0-9_\-\.åäöÅÄÖ]"
    return re.sub(allowed, "_", name)


def extract_dyntaxa_num(lsid: pd.Series) -> pd.Series:
    # From 'urn:lsid:dyntaxa.se:Taxon:6003469' -> '6003469' (string)
    pat = re.compile(r":Taxon:(\d+)$")
    out = lsid.astype("string").str.extract(pat, expand=False)
    return out  # string dtype; may contain <NA>


def load_redlist_ids(path: str) -> set:
    """Reads national red-list CSV (semicolon-delimited) and returns a set of Dyntaxa IDs as strings."""
    df = pd.read_csv(path, sep=";", dtype="string", low_memory=False)
    # Column name is 'Taxon id' (space), numeric
    if "Taxon id" not in df.columns:
        raise SystemExit("Red-list CSV missing 'Taxon id' column.")
    ids = df["Taxon id"].dropna().astype("Int64").astype("string")
    return set(ids.tolist())


def load_region_union():
    """
    Load the selected geographic region (one or multiple polygons) from a GeoPackage
    and return a single unary_union geometry in TARGET_CRS.

    The GeoPackage is expected to contain a field (GEOREGION_FIELD) with values like:
      "alpine", "northern", "southern"

    Overlapping polygons are fine: the union will include overlaps.
    """
    if not APPLY_GEO_FILTER:
        return None

    gpkg = Path(GEOREGIONS_GPKG)
    if not gpkg.is_file():
        raise SystemExit(f"[GEO] GeoPackage not found: {GEOREGIONS_GPKG}")

    # Pick layer
    layer = GEOREGIONS_LAYER
    if layer is None:
        try:
            layers = gpd.list_layers(gpkg)  # geopandas >=0.14
            if layers is None or len(layers) == 0:
                raise ValueError("empty")
            layer = layers.iloc[0]["name"]
        except Exception:
            import fiona
            names = list(fiona.listlayers(str(gpkg)))
            if not names:
                raise SystemExit(f"[GEO] No layers found in GeoPackage: {GEOREGIONS_GPKG}")
            layer = names[0]

    g = gpd.read_file(gpkg, layer=layer)

    if g.empty:
        raise SystemExit(f"[GEO] Layer '{layer}' is empty in {GEOREGIONS_GPKG}")

    if GEOREGION_FIELD not in g.columns:
        raise SystemExit(f"[GEO] Field '{GEOREGION_FIELD}' not found in layer '{layer}'. Columns: {list(g.columns)}")

    choice = str(GEOREGION_CHOICE).strip().lower()
    vals = g[GEOREGION_FIELD].astype("string").str.strip().str.lower()
    g_sel = g.loc[vals == choice].copy()

    if g_sel.empty:
        avail = sorted(set(vals.dropna().tolist()))
        raise SystemExit(f"[GEO] No polygons matched {GEOREGION_FIELD}='{choice}'. Available values: {avail}")

    # Ensure CRS and dissolve to union
    if g_sel.crs is None:
        raise SystemExit(f"[GEO] Region layer '{layer}' has no CRS defined. Please assign CRS in QGIS.")
    g_sel = g_sel.to_crs(TARGET_CRS)
    region_union = g_sel.geometry.unary_union
    print(f"[GEO] Using region '{choice}' from {GEOREGIONS_GPKG} (layer='{layer}', n_polygons={len(g_sel)})")
    return region_union


def filter_points_to_region(gdf: gpd.GeoDataFrame, region_union):
    """Return only points intersecting the selected region union geometry."""
    if region_union is None or gdf.empty:
        return gdf
    # intersects is safer than within for borderline points (and for multipolygons).
    return gdf[gdf.geometry.intersects(region_union)].copy()



def iter_parts(parquet_dir: str):
    """Yield DataFrames for each parquet part file in sorted order."""
    p = Path(parquet_dir)
    parts = sorted(p.glob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"No parquet parts found in {parquet_dir}")
    for fp in parts:
        yield fp, pd.read_parquet(fp)


def to_points_3006(df: pd.DataFrame, region_union=None) -> gpd.GeoDataFrame:
    """
    Convert dataframe to GeoDataFrame (EPSG:3006).
    Optimization: Uses vectorization (points_from_xy).
    """
    # 1. Basic cleaning
    if COL_LON not in df.columns or COL_LAT not in df.columns:
        return gpd.GeoDataFrame(df.iloc[0:0], geometry=[], crs=TARGET_CRS)
    
    # Drop NAs on coords immediately
    w = df.dropna(subset=[COL_LON, COL_LAT]).copy()
    if w.empty:
        return gpd.GeoDataFrame(w, geometry=[], crs=TARGET_CRS)

    # Coerce to float
    w[COL_LON] = pd.to_numeric(w[COL_LON], errors="coerce")
    w[COL_LAT] = pd.to_numeric(w[COL_LAT], errors="coerce")
    
    # 2. Fast numeric bounding box filter (Sweden BBox)
    lon_min, lat_min, lon_max, lat_max = SWEDEN_BBOX
    w = w[w[COL_LON].between(lon_min, lon_max) & w[COL_LAT].between(lat_min, lat_max)]
    
    if w.empty:
        return gpd.GeoDataFrame(w, geometry=[], crs=TARGET_CRS)

    # 3. Vectorized Geometry Creation (Much faster than list comprehension)
    gdf = gpd.GeoDataFrame(
        w,
        geometry=gpd.points_from_xy(w[COL_LON], w[COL_LAT]),
        crs=4326
    ).to_crs(TARGET_CRS)

    # 4. Only apply expensive region filter if explicitly requested (Pass 1 only)
    if region_union is not None:
        gdf = filter_points_to_region(gdf, region_union)
        
    return gdf

def pick_locality_key(df: pd.DataFrame) -> pd.Series:
    """Row-wise LocalityKey:
       1) localityID
       2) locality (text)
       3) rounded lon/lat -> 'cell_<lat>_<lon>' (3 decimals)
       4) leave NaN (we'll repair later from SWEREF if present)
    """
    out = pd.Series(index=df.index, dtype="string")

    if COL_LOCID in df.columns:
        out = df[COL_LOCID].astype("string")

    if COL_LOCALITY in df.columns:
        m = out.isna() | (out.str.len() == 0)
        out = out.where(~m, df[COL_LOCALITY].astype("string"))

    if (COL_LON in df.columns) and (COL_LAT in df.columns):
        lon = pd.to_numeric(df[COL_LON], errors="coerce")
        lat = pd.to_numeric(df[COL_LAT], errors="coerce")
        # enforce fixed 3 decimals, keep trailing zeros
        lon3 = lon.map(lambda x: f"{x:.3f}")
        lat3 = lat.map(lambda x: f"{x:.3f}")
        coord_key = "cell_" + lat3 + "_" + lon3
        m = out.isna() | (out.str.len() == 0)
        out = out.where(~m, coord_key.astype("string"))

    # DO NOT fill "NA" here; leave NaN so we can still repair from SWEREF later
    return out



def main(seed=None):
    """
    Main entry point.
    If seed is None -> random each run.
    If seed is an int -> reproducible results.
    """
    import numpy as np
    import random
    start = time.time()

    # initialize a random generator
    if seed is None:
        seed = int(time.time() * 1e6) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    random.seed(seed)  # also for Python's random if used anywhere

    print(f"[INIT] Using random seed = {seed}")

    # --- geographic filter (region) ---
    region_union = load_region_union()
    region_folder = sanitize_filename(str(GEOREGION_CHOICE).strip().lower()) if APPLY_GEO_FILTER else "national"

    output_root = Path(OUTPUT_ROOT) / region_folder
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[OUT] Writing outputs under: {output_root}")

    # output_root already created above

    # --- get merged focal species from lists + CSV ---
    focal_ids, focal_names = get_focal_species()
    print("Focal TaxonIDs:", sorted(focal_ids))
    print("Focal Names:", sorted(focal_names))
    focal_id_set   = set(str(x) for x in focal_ids)   # keep as strings
    focal_name_set = set(focal_names)

    # Load red-list IDs as strings (numeric Dyntaxa IDs)
    red_ids = load_redlist_ids(REDLIST_CSV)

    print("[PASS 1] Scanning focal occurrences...")
    focal_frames = []
    focal_totals  = {}
    focal_meta    = {}
    focal_taxonid = {}

    for fp, df in iter_parts(OCCURRENCE_DIR):
        # Normalize essential fields
        if COL_TAXONID in df.columns:
            df["taxonID_num"] = extract_dyntaxa_num(df[COL_TAXONID])
        else:
            df["taxonID_num"] = pd.NA

        # Focal filters (by ID or by scientificName)
        mask = pd.Series([False] * len(df))
        if focal_id_set:
            mask |= df["taxonID_num"].astype("string").isin(focal_id_set)
        if focal_name_set and COL_SCI in df.columns:
            mask |= df[COL_SCI].astype("string").isin(focal_name_set)

        if not mask.any():
            continue

        gdf = to_points_3006(df.loc[mask], region_union)
        if gdf.empty:
            continue

        # Track totals and meta per scientificName (group if multiple)
        if COL_SCI not in gdf.columns:
            # If no scientificName, synthesize from taxonID
            gdf[COL_SCI] = gdf["taxonID_num"].fillna("unknown")

        for sci, sub in gdf.groupby(COL_SCI):
            total_before = focal_totals.get(sci, 0)
            focal_totals[sci] = total_before + len(sub)
            # Keep a vernacular sample
            if sci not in focal_meta:
                vern = sub[COL_VERN].dropna().astype("string").head(1).tolist()
                vern_name = vern[0] if vern else None
                focal_meta[sci] = vern_name
            # Keep a numeric taxonID sample (string)
            if sci not in focal_taxonid:
                dyn = sub["taxonID_num"].dropna().astype("string").head(1).tolist()
                focal_taxonid[sci] = dyn[0] if dyn else None

        focal_frames.append(gdf[[COL_SCI, COL_VERN, "taxonID_num", "geometry"]])

    if not focal_frames:
        raise SystemExit("No focal occurrences found. Check FOCAL_TAXONID_NUMS / FOCAL_SCI_NAMES.")

    focal_all = pd.concat(focal_frames, ignore_index=True)

    # Sample up to SAMPLE_SITES per species, then build buffers
    print("[PASS 1] Sampling and buffering focal sites per species...")
    buffers_per_species = {}
    focal_meta_rows = {}   # sci -> DataFrame with one row per sampled site (buffer)

    for sci, sub in focal_all.groupby(COL_SCI):
        # attach focal locality + SWEREF coords for EACH focal occurrence
        sub = sub.copy()
        # pick focal locality key per row
        sub["FocalLocalityKey"] = pick_locality_key(sub)
        # SWEREF coordinates from geometry (EPSG:3006): x=E, y=N
        sub["FocalE"] = sub.geometry.x.astype("float64")
        sub["FocalN"] = sub.geometry.y.astype("float64")

        # sample up to SAMPLE_SITES
        if len(sub) > SAMPLE_SITES:
            sub = sub.sample(SAMPLE_SITES, random_state=int(rng.integers(0, 2**32 - 1)))

        # assign a stable per-buffer ID
        sub = sub.reset_index(drop=True)
        sub["__FID__"] = sub.index.astype(int)  # 0..k-1 within species
        sub["__SCI__"] = sci

        # 50 m buffer (fixed)
        buf = sub.copy()
        buf["geometry"] = buf.geometry.buffer(BUFFER_M, cap_style=1)
        buffers_per_species[sci] = gpd.GeoDataFrame(
            buf[["__FID__", "__SCI__", "FocalLocalityKey", "FocalE", "FocalN", "geometry"]],
            geometry="geometry", crs=sub.crs # type: ignore
        )

        # keep a skinny table for writing rows later (one per focal site)
        focal_meta_rows[sci] = buf[["__FID__", "FocalLocalityKey", "FocalE", "FocalN"]].copy()


# ==============================================================================
    # 1. SETUP & PRE-CALCULATION (Context: After Pass 1, before Pass 2 loop)
    # ==============================================================================
    
    # Prepare output accumulators (Same as your original code)
    results = {}
    for sci in buffers_per_species:
        results[sci] = dict(
            focal_obs_total = int(focal_totals.get(sci, 0)),
            unique_associate_ids = set(),
            unique_red_associate_ids = set(),
            all_assoc_ids = [],
            loc_species_all = {int(fid): set() for fid in focal_meta_rows[sci]["__FID__"]},
            loc_species_red = {int(fid): set() for fid in focal_meta_rows[sci]["__FID__"]},
            red_species_per_locality = {},
            locality_red_rows = [],
        )

    # Combine all species buffers into one big GeoDataFrame
    # (We need this for the 'sjoin', but now we ALSO use it to calculate bounds)
    buffers_rows = []
    for sci, gdfbuf in buffers_per_species.items():
        buffers_rows.append(gdfbuf[["__FID__", "__SCI__", "FocalLocalityKey", "FocalE", "FocalN", "geometry"]])
    
    # If no buffers exist, we can't do anything
    if not buffers_rows:
        print("[WARN] No focal buffers generated. Exiting.")
        return

    buffers_all = gpd.GeoDataFrame(
        pd.concat(buffers_rows, ignore_index=True),
        geometry="geometry", crs=TARGET_CRS
    )

    # --- [NEW] OPTIMIZATION START: Calculate Global Bounding Box ---
    # We calculate the total extent of ALL buffers to filter raw text files fast.
    print("[INFO] Calculating global bounding box of focal buffers for pre-filtering...")
    
    minx, miny, maxx, maxy = buffers_all.total_bounds
    
    # Convert these projected coordinates (SWEREF 3006) back to Lat/Lon (WGS84)
    # so we can filter the raw 'decimalLongitude'/'decimalLatitude' columns.
    from shapely.geometry import box
    bbox_poly_3006 = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=TARGET_CRS)
    bbox_poly_4326 = bbox_poly_3006.to_crs(4326)
    
    # Get the limits
    total_lon_min, total_lat_min, total_lon_max, total_lat_max = bbox_poly_4326.total_bounds
    
    # Add a small margin (0.05 deg) to be safe against projection rounding
    margin = 0.05
    total_lon_min -= margin
    total_lon_max += margin
    total_lat_min -= margin
    total_lat_max += margin

    print(f"[INFO] Global filter bounds (Lon/Lat): {total_lon_min:.2f}, {total_lat_min:.2f} to {total_lon_max:.2f}, {total_lat_max:.2f}")
    # --- [NEW] OPTIMIZATION END ---
    
    # Pass 2: stream all parts again
    print("[PASS 2] Scanning associates against focal buffers...")
    
    for fp, df in iter_parts(OCCURRENCE_DIR):
        # --- OPTIMIZATION 1: Fast Numeric Filter ---
        # Only keep rows that are geographically roughly near ANY focal buffer
        # This discards 90-99% of data before expensive geometry operations
        if COL_LON in df.columns and COL_LAT in df.columns:
            # Ensure numeric
            df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
            df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
            
            # Apply BBox filter
            df = df[
                df[COL_LON].between(total_lon_min, total_lon_max) & 
                df[COL_LAT].between(total_lat_min, total_lat_max)
            ]
        
        if df.empty:
            continue

        df["taxonID_num"] = extract_dyntaxa_num(df.get(COL_TAXONID, pd.Series(index=df.index)))

        # Exclude focal species themselves using the *merged* sets
        is_focal_row = pd.Series(False, index=df.index)
        if focal_ids:
            is_focal_row |= df["taxonID_num"].astype("string").isin(focal_id_set)
        if focal_names and COL_SCI in df.columns:
            is_focal_row |= df[COL_SCI].astype("string").isin(focal_name_set)
        df_assoc = df.loc[~is_focal_row].copy()

        # --- OPTIMIZATION 2: Skip Region Filter ---
        # We pass region_union=None here. 
        # Why? Because we are about to do an inner join with 'buffers_all'.
        # 'buffers_all' is ALREADY filtered to the region in Pass 1.
        # If a point intersects a buffer, it is by definition in the region (or close enough).
        gdf_assoc = to_points_3006(df_assoc, region_union=None)
        if gdf_assoc.empty:
            continue

        # Locality key and red-flag
        loc_key = pick_locality_key(df_assoc)
        gdf_assoc["LocalityKey"] = loc_key.reindex(gdf_assoc.index)
        gdf_assoc["is_red"] = gdf_assoc["taxonID_num"].astype("string").isin(red_ids)
        gdf_assoc["E"] = gdf_assoc.geometry.x.astype("float64")
        gdf_assoc["N"] = gdf_assoc.geometry.y.astype("float64")

        # Spatial join once per part (bbox-indexed + exact predicate)
        # hits_left: associates; hits_right: buffers with __SCI__
        hits = gpd.sjoin(
            gdf_assoc, buffers_all[["__FID__", "__SCI__", "FocalLocalityKey", "FocalE", "FocalN", "geometry"]],
            how="inner", predicate="intersects"
        )

        if hits.empty:
            continue

        for sci, sub in hits.groupby("__SCI__"):
            r = results[sci]

            # update global associate lists
            assoc_ids = sub["taxonID_num"].astype("string").fillna("NA").tolist()
            r["all_assoc_ids"].extend(assoc_ids)
            r["unique_associate_ids"].update([x for x in assoc_ids if x and x != "NA"])

            # per focal site aggregation
            for fid, g in sub.groupby("__FID__"):
                fid = int(fid)
                sp = g["taxonID_num"].astype("string").dropna().tolist()
                r["loc_species_all"][fid].update(sp)

            # red-listed per site
            red_sub = sub[sub["is_red"]]
            if not red_sub.empty:
                # unique red species (overall)
                r["unique_red_associate_ids"].update(
                    red_sub["taxonID_num"].astype("string").dropna().unique().tolist()
                )
                # per focal site red species
                for fid, g in red_sub.groupby("__FID__"):
                    fid = int(fid)
                    sp = g["taxonID_num"].astype("string").dropna().tolist()
                    r["loc_species_red"][fid].update(sp)

                # >>> Accumulate UNIQUE red-listed species per locality <<<
                for loc_key, gk in red_sub.groupby("LocalityKey"):
                    sp_set = set(gk["taxonID_num"].astype("string").dropna().unique().tolist())
                    bucket = r["red_species_per_locality"].get(loc_key, set())
                    bucket.update(sp_set)
                    r["red_species_per_locality"][loc_key] = bucket
                    r["locality_red_rows"].append(
                        red_sub[["LocalityKey", "taxonID_num", "N", "E"]].copy()
                    )

    # Write outputs per species
    # output_root already created above

    for sci, r in results.items():
        focal_id_str = focal_taxonid.get(sci) or ""  # may be None/empty
        focal_display = sci

        # ---- NEW FOLDER NAMING LOGIC ----
        base_name = sanitize_filename(sci)

        out_dir = output_root / base_name

        # If already exists, create a numbered suffix so nothing is overwritten
        # e.g., Bombus_pascuorum_1000123__001, __002, ...
        if out_dir.exists():
            i = 1
            while True:
                candidate = output_root / f"{base_name}__{i:03d}"
                if not candidate.exists():
                    out_dir = candidate
                    break
                i += 1

        out_dir.mkdir(parents=True, exist_ok=False)
        # ---- END NEW FOLDER NAMING LOGIC ----

        focal_id_str = focal_taxonid.get(sci) or ""  # may be None
        focal_display = sci
        folder_name = sanitize_filename(sci)

        # Global >=15 frequency filter for summary outputs
        freq = Counter(x for x in r["all_assoc_ids"] if x and x != "NA")
        keep_ids = {tid for tid, c in freq.items() if c >= 15}
        # Unique associates AFTER applying >=15 rule
        uniq_ids_filtered = sorted(keep_ids)
        # Red-listed associates AFTER applying >=15 rule
        red_after_filter = sorted({x for x in r["unique_red_associate_ids"] if x and x in keep_ids})
        
        def _repr_int(x):
            try:
                return str(int(round(float(x))))
            except Exception:
                return "0"
            
        # A) uniqueFinds.txt  -> filtered (>=15)
        with open(out_dir / "uniqueFinds.txt", "w", encoding="utf-8") as f:
            f.write(f"{focal_display} {' '.join(uniq_ids_filtered)}\n")

        # B) totalTargetObs.txt  (focal_total, unique_associates>=15, unique_red_associates>=15)
        with open(out_dir / "totalTargetObs.txt", "w", encoding="utf-8") as f:
            f.write(
                f"{focal_display} {r['focal_obs_total']} {len(uniq_ids_filtered)} {len(red_after_filter)}\n"
            )

        # C) redFinds.txt (unique red-listed associates) -> filtered (>=15)
        with open(out_dir / "redFinds.txt", "w", encoding="utf-8") as f:
            f.write(f"{focal_display} {' '.join(red_after_filter)}\n")

        # D) allObservations.txt (all associate observation IDs incl. duplicates) -> filtered (>=15)
        filtered_all_ids = [x for x in r["all_assoc_ids"] if x and x in keep_ids]
        with open(out_dir / "allObservations.txt", "w", encoding="utf-8") as f:
            f.write(f"{focal_display} {' '.join(filtered_all_ids)}\n")

        # E) RedSpeciesLocal.txt (counts of redlisted associate OBS per locality)
        # One "<LocalityKey> <count>" per line, sorted by key
        with open(out_dir / "RedSpeciesLocal.txt", "w", encoding="utf-8") as f:
            for k in sorted(r["red_species_per_locality"].keys()):
                f.write(f"{k} {len(r['red_species_per_locality'][k])}\n")
        
        # Helper to pick representative SWEREF coords per locality:
        def _repr_coords(pairs):
            # pairs: list of (E, N)
            if not pairs:
                return ("0", "0")
            # Use median to reduce outlier influence; cast to ints
            import numpy as np
            arr = np.array(pairs, dtype="float64")
            E = int(np.median(arr[:, 0]))
            N = int(np.median(arr[:, 1]))
            return (str(N), str(E))  # N first, then E (as in your example)

        # F) LocalitySpecies_ALL.txt  (one row per sampled focal site)
        with open(out_dir / "LocalitySpecies_ALL.txt", "w", encoding="utf-8") as f:
            f.write("LocalityID SWEREF_N SWEREF_E Species_IDs\n")
            meta = focal_meta_rows[sci].sort_values("__FID__")
            for _, row in meta.iterrows():
                fid = int(row["__FID__"])
                loc_id = str(row["FocalLocalityKey"]).replace(" ", "_")
                N = _repr_int(row["FocalN"])
                E = _repr_int(row["FocalE"])
                sp_ids = sorted(r["loc_species_all"].get(fid, set()), key=lambda s: (len(s or ""), s or ""))
                f.write(f"{loc_id} {N} {E} {' '.join(sp_ids)}\n")

        # G) LocalitySpecies_RED.txt  (group by LocalityID; never "NA")
        # Build once from the accumulated raw rows
        if r["locality_red_rows"]:
            df_lr = pd.concat(r["locality_red_rows"], ignore_index=True)
            # Start with the per-row LocalityKey you computed earlier
            df_lr["LocalityID"] = df_lr["LocalityKey"].astype("string")

            # If LocalityID is missing/empty/"NA", synthesize from SWEREF -> WGS84-based cell_<lat>_<lon>
            mask_bad = df_lr["LocalityID"].isna() | (df_lr["LocalityID"] == "") | (df_lr["LocalityID"] == "NA")
            if mask_bad.any():
                # Build geometry in EPSG:3006 (E, N), convert to EPSG:4326, then round to fixed 3 decimals
                pts = gpd.GeoSeries(
                    gpd.points_from_xy(
                        pd.to_numeric(df_lr.loc[mask_bad, "E"], errors="coerce"),
                        pd.to_numeric(df_lr.loc[mask_bad, "N"], errors="coerce"),
                        crs=3006
                    )
                ).to_crs(4326)
                lat = pts.y.map(lambda v: f"{float(v):.3f}")
                lon = pts.x.map(lambda v: f"{float(v):.3f}")
                df_lr.loc[mask_bad, "LocalityID"] = ("cell_" + lat + "_" + lon).astype("string")

            # Drop rows that still have no ID (truly hopeless)
            df_lr = df_lr.dropna(subset=["LocalityID"])

            # Aggregate: median SWEREF per LocalityID + unique species list
            agg = (
                df_lr.groupby("LocalityID", as_index=False)
                     .agg(
                         SWEREF_N=("N", "median"),
                         SWEREF_E=("E", "median"),
                         Species_IDs=("taxonID_num", lambda s: " ".join(sorted(set(s.astype("string").dropna()))))
                     )
            )
            # Round SWEREF to integers for readability
            agg["SWEREF_N"] = agg["SWEREF_N"].round(0).astype("Int64").astype("string")
            agg["SWEREF_E"] = agg["SWEREF_E"].round(0).astype("Int64").astype("string")

            # Write
            out_path = out_dir / "LocalitySpecies_RED.txt"
            agg[["LocalityID", "SWEREF_N", "SWEREF_E", "Species_IDs"]].to_csv(out_path, sep="\t", index=False)
        else:
            # Still write an empty header if there were no red associates
            (out_dir / "LocalitySpecies_RED.txt").write_text("LocalityID\tSWEREF_N\tSWEREF_E\tSpecies_IDs\n", encoding="utf-8")

        print(f"[OK] Wrote outputs for {focal_display} → {out_dir}")

        """# G) LocalitySpecies_RED.txt  (one row per sampled focal site; empty list allowed)
        with open(out_dir / "LocalitySpecies_RED.txt", "w", encoding="utf-8") as f:
            f.write("LocalityID SWEREF_N SWEREF_E Species_IDs\n")
            meta = focal_meta_rows[sci].sort_values("__FID__")
            for _, row in meta.iterrows():
                fid = int(row["__FID__"])
                loc_id = str(row["FocalLocalityKey"]).replace(" ", "_")
                N = _repr_int(row["FocalN"])
                E = _repr_int(row["FocalE"])
                sp_ids = sorted(r["loc_species_red"].get(fid, set()), key=lambda s: (len(s or ""), s or ""))
                # if no hits, Species_IDs will be empty — that’s OK and required
                f.write(f"{loc_id} {N} {E} {' '.join(sp_ids)}\n")

        print(f"[OK] Wrote outputs for {focal_display} → {out_dir}")"""

    dt = time.time() - start
    print(f"[DONE] Finished in {dt:.1f}s")

if __name__ == "__main__":
    main()