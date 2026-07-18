"""
build_reference_from_raw_with_fly_id.py -- modified version of
build_reference_from_raw.py that additionally tracks which fly each cell
came from, enabling a leave-one-fly-out robustness check that wasn't
possible with the original extraction (which only kept flat per-group
cell lists, no provenance).

FLY IDENTITY: parsed from the filename as {date}_Fly{N} -- e.g.,
"200728_Fly1_11_Image11_pData_SIMA_only_m.mat" -> fly_id "200728_Fly1".
Date is included because fly numbering resets each recording day: two
files both labeled "Fly1" from different dates are two different
physical flies, not the same one recorded twice.

Everything else -- the R_teta formula (max over time), the
celltype_from_labels mapping (T4_T5 + Layer -> T4A-D/T5A-D), the
disk-management approach (one file at a time, deleted immediately after
extraction) -- is unchanged and already validated against the original
extraction (confirmed matching to <1 degree / <1% magnitude against
processed_Data_SIMA_CS5_sh_Edges.mat's established Z values).

USAGE:
    python build_reference_from_raw_with_fly_id.py
    (requires re-downloading all 117 files -- ~17.6GB, same cost as the
    original extraction, since fly identity cannot be recovered from
    data already saved without it)
"""

import os
import re
import time

import h5py
import numpy as np
from remotezip import RemoteZip
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import pdist, squareform

ZENODO_URL = "https://zenodo.org/records/5562205/files/DATA.zip"
TEMP_DIR = "raw_extraction_temp_v2"
PROGRESS_FILE = "raw_r_teta_progress_with_fly.npz"
TETA_DEG = np.array([90, 45, 0, 315, 270, 225, 180, 135])
GROUP_NAMES = ["T4A", "T4B", "T4C", "T4D", "T5A", "T5B", "T5C", "T5D"]
N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)


def celltype_from_labels(t4_t5_val, layer_val):
    family = "T4" if t4_t5_val == 4 else ("T5" if t4_t5_val == 5 else None)
    if family is None or not (1 <= layer_val <= 4):
        return None
    sublayer = "ABCD"[int(layer_val) - 1]
    return family + sublayer


def fly_id_from_filename(filename):
    """Parses '{date}_Fly{N}_...' -> '{date}_Fly{N}'. Date included
    because fly numbering resets each recording day.
    """
    match = re.match(r"(\d+)_Fly(\d+)_", os.path.basename(filename))
    if match:
        return f"{match.group(1)}_Fly{match.group(2)}"
    return "unknown"


def extract_r_teta_from_file(local_path, fly_id):
    """Returns a dict {group_name: [(r_teta, fly_id), ...]} for one file."""
    results = {g: [] for g in GROUP_NAMES}
    with h5py.File(local_path, "r") as f:
        if "strct" not in f or "ClusterInfo_ManuallySelect" not in f["strct"]:
            return results
        cluster = f["strct/ClusterInfo_ManuallySelect"]
        if "AV_ROIS_resp" not in cluster:
            return results

        resp_refs = cluster["AV_ROIS_resp"]
        t4_t5 = cluster["T4_T5"][()].flatten()
        layer = cluster["Layer"][()].flatten()
        n_rois = resp_refs.shape[0]

        for i in range(n_rois):
            ct = celltype_from_labels(t4_t5[i], layer[i])
            if ct is None:
                continue
            ref = resp_refs[i, 0]
            resp_matrix = f[ref][()]
            if resp_matrix.ndim != 2 or resp_matrix.shape[1] != 8:
                continue
            r_teta = resp_matrix.max(axis=0)
            results[ct].append((r_teta, fly_id))
    return results


def rank_residualize(rdm, against_rdm):
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    r_vals = rankdata(rdm[iu])
    r_against = rankdata(against_rdm[iu])
    slope, intercept = np.polyfit(r_against, r_vals, 1)
    predicted = slope * r_against + intercept
    return r_vals - predicted


def build_population_matrix(all_cells, exclude_fly=None):
    """all_cells: dict {group_name: [(r_teta, fly_id), ...]}.
    If exclude_fly is set, omits all cells from that fly before averaging.
    """
    population_matrix = np.zeros((N_DIRECTIONS, len(GROUP_NAMES)))
    for col, g in enumerate(GROUP_NAMES):
        cells = [r for r, fid in all_cells[g] if fid != exclude_fly]
        if len(cells) == 0:
            continue
        population_matrix[:, col] = np.array(cells).mean(axis=0)
    return population_matrix


def main():
    os.makedirs(TEMP_DIR, exist_ok=True)

    if os.path.exists(PROGRESS_FILE):
        saved = np.load(PROGRESS_FILE, allow_pickle=True)
        all_cells = {g: list(saved[g]) for g in GROUP_NAMES}
        processed_files = set(saved["_processed_files"].tolist())
        print(f"Resuming: {len(processed_files)} files already processed")
    else:
        all_cells = {g: [] for g in GROUP_NAMES}
        processed_files = set()

    print("=== Listing all raw recording files in the archive ===")
    with RemoteZip(ZENODO_URL) as zf:
        all_files = zf.namelist()
        pdata_files = sorted([f for f in all_files if "pData_SIMA_only_m.mat" in f])
    print(f"Total files: {len(pdata_files)}")

    remaining = [f for f in pdata_files if f not in processed_files]
    print(f"Remaining to process: {len(remaining)}\n")

    for idx, target in enumerate(remaining):
        fly_id = fly_id_from_filename(target)
        t0 = time.time()
        try:
            with RemoteZip(ZENODO_URL) as zf:
                zf.extract(target, path=TEMP_DIR)
                extracted_path = os.path.join(TEMP_DIR, target)

            file_results = extract_r_teta_from_file(extracted_path, fly_id)
            n_cells_this_file = sum(len(v) for v in file_results.values())
            for g in GROUP_NAMES:
                all_cells[g].extend(file_results[g])

            os.remove(extracted_path)

            processed_files.add(target)
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(remaining)}] {os.path.basename(target)} "
                  f"(fly={fly_id}): {n_cells_this_file} cells ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{idx+1}/{len(remaining)}] {os.path.basename(target)}: "
                  f"FAILED ({e}) -- skipping, will retry on next run")
            continue

        if (idx + 1) % 10 == 0 or idx == len(remaining) - 1:
            save_dict = {g: np.array(all_cells[g], dtype=object) for g in GROUP_NAMES}
            save_dict["_processed_files"] = np.array(list(processed_files), dtype=object)
            np.savez(PROGRESS_FILE, **save_dict)
            print(f"    -- progress saved: {len(processed_files)} files")

    print("\n=== Extraction complete ===")
    all_fly_ids = sorted(set(
        fid for g in GROUP_NAMES for _, fid in all_cells[g]
    ))
    print(f"  Unique flies: {len(all_fly_ids)}")
    for g in GROUP_NAMES:
        print(f"  {g}: {len(all_cells[g])} cells")

    print("\n=== Building the full (all-flies) reference, for comparison ===")
    full_population_matrix = build_population_matrix(all_cells)
    full_rdm = squareform(pdist(full_population_matrix, metric='cosine'))
    np.save("raw_population_matrix_v2.npy", full_population_matrix)

    circ_ref = np.zeros((N_DIRECTIONS, N_DIRECTIONS))
    for i in range(N_DIRECTIONS):
        for j in range(N_DIRECTIONS):
            d = abs(STIMULUS_DIRECTIONS_DEG[i] - STIMULUS_DIRECTIONS_DEG[j])
            circ_ref[i, j] = min(d, 360 - d)
    full_ref_resid = rank_residualize(full_rdm, circ_ref)

    print("\n=== Loading Flyvis CC/random RDMs for the leave-one-fly-out "
          "comparison ===")
    flyvis = np.load("results_exp1_8dir_50models_full_shiu.npz", allow_pickle=True)
    cc_rdm = flyvis["cc_rdm_cosine"]
    rand_rdm = flyvis["rand_rdm_cosine"]
    cc_resid = rank_residualize(cc_rdm, circ_ref)
    rand_resid = rank_residualize(rand_rdm, circ_ref)

    r_cc_full, _ = spearmanr(cc_resid, full_ref_resid)
    r_rand_full, _ = spearmanr(rand_resid, full_ref_resid)
    print(f"  Full reference (all {len(all_fly_ids)} flies): "
          f"CC r={r_cc_full:.4f}, Random r={r_rand_full:.4f}")

    print(f"\n=== LEAVE-ONE-FLY-OUT: rebuilding the reference with each of "
          f"{len(all_fly_ids)} flies excluded in turn ===\n")
    cc_shifts = []
    rand_shifts = []
    for fly in all_fly_ids:
        reduced_matrix = build_population_matrix(all_cells, exclude_fly=fly)
        reduced_rdm = squareform(pdist(reduced_matrix, metric='cosine'))
        reduced_resid = rank_residualize(reduced_rdm, circ_ref)

        r_cc, _ = spearmanr(cc_resid, reduced_resid)
        r_rand, _ = spearmanr(rand_resid, reduced_resid)
        cc_shift = r_cc - r_cc_full
        rand_shift = r_rand - r_rand_full
        cc_shifts.append((fly, r_cc, cc_shift))
        rand_shifts.append((fly, r_rand, rand_shift))
        print(f"  Excluding {fly}: CC r={r_cc:+.4f} (shift {cc_shift:+.4f})  "
              f"| Random r={r_rand:+.4f} (shift {rand_shift:+.4f})")

    cc_shifts_sorted = sorted(cc_shifts, key=lambda x: abs(x[2]), reverse=True)
    rand_shifts_sorted = sorted(rand_shifts, key=lambda x: abs(x[2]), reverse=True)

    print(f"\n=== Interpretation ===")
    print(f"  Most influential fly for CC: {cc_shifts_sorted[0][0]} "
          f"(shift {cc_shifts_sorted[0][2]:+.4f})")
    print(f"  Most influential fly for Random: {rand_shifts_sorted[0][0]} "
          f"(shift {rand_shifts_sorted[0][2]:+.4f})")

    max_cc_shift = abs(cc_shifts_sorted[0][2])
    max_rand_shift = abs(rand_shifts_sorted[0][2])
    if max_cc_shift > 0.3 or max_rand_shift > 0.3:
        print(f"  CONCERNING: a single fly's exclusion shifts the result by "
              f"more than 0.3 (CC max={max_cc_shift:.3f}, random "
              f"max={max_rand_shift:.3f}). The result may depend heavily on "
              f"one fly, out of only {len(all_fly_ids)} total -- worth real caution.")
    else:
        print(f"  REASSURING: no single fly's exclusion shifts the result "
              f"past 0.3 (CC max={max_cc_shift:.3f}, random "
              f"max={max_rand_shift:.3f}). The result does not appear to "
              f"depend disproportionately on any one of the {len(all_fly_ids)} flies.")


if __name__ == "__main__":
    main()
