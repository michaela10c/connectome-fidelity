"""
build_reference_from_raw.py -- builds the Henning reference RDM directly
from real per-direction response data (R_teta), extracted from all 117
raw recording files, with NO von Mises parametric reconstruction step.

VALIDATED FORMULA (confirmed against processed_Data_SIMA_CS5_sh_Edges.mat's
already-established Z values, exact match to <1 degree / <1% magnitude):
  R_teta = AV_ROIS_resp.max(axis=0)  # peak response per direction, 8 values
  Z = sum(R_teta * exp(1j*teta_rad)) / sum(abs(R_teta))
  teta_deg = [90, 45, 0, 315, 270, 225, 180, 135]

VALIDATED LABELING SCHEME (confirmed via Layer values 1-4 combined with
T4_T5 values 4/5 cleanly separating T5A from T5B by preferred direction):
  T4_T5 == 4 -> "T4" family;  T4_T5 == 5 -> "T5" family
  Layer 1/2/3/4 -> sub-layer A/B/C/D within that family

WHY THIS MATTERS: every previous version of this reference (including the
one already checked five independent ways) was built by reconstructing a
von Mises curve from each cell's compressed Z summary. This version skips
that reconstruction entirely -- R_teta itself, the real 8-point response,
is averaged directly within each of the 8 groups. This is the most direct
test yet of whether the earlier finding (real per-cell heterogeneity,
82.3% circular, 17.7% real non-circular residual) was an artifact of the
parametric reconstruction or genuinely reflects the raw data.

DISK MANAGEMENT: processes one file at a time via remotezip, extracts only
what's needed, deletes the raw file immediately after. Peak disk usage is
one file (~150MB), not the full 17.6GB.

Saves progress incrementally (raw_r_teta_by_group.npz) so a crash partway
through doesn't lose completed work -- re-running skips files already
processed.

USAGE:
    python build_reference_from_raw.py
"""

import os
import time

import h5py
import numpy as np
from remotezip import RemoteZip

ZENODO_URL = "https://zenodo.org/records/5562205/files/DATA.zip"
TEMP_DIR = "raw_extraction_temp"
PROGRESS_FILE = "raw_r_teta_progress.npz"
TETA_DEG = np.array([90, 45, 0, 315, 270, 225, 180, 135])
GROUP_NAMES = ["T4A", "T4B", "T4C", "T4D", "T5A", "T5B", "T5C", "T5D"]


def celltype_from_labels(t4_t5_val, layer_val):
    family = "T4" if t4_t5_val == 4 else ("T5" if t4_t5_val == 5 else None)
    if family is None or not (1 <= layer_val <= 4):
        return None  # unexpected label combination -- skip rather than guess
    sublayer = "ABCD"[int(layer_val) - 1]
    return family + sublayer


def extract_r_teta_from_file(local_path):
    """Returns a dict {group_name: list of R_teta arrays (8,)} for one file."""
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
                continue  # unexpected shape -- skip rather than guess
            r_teta = resp_matrix.max(axis=0)
            results[ct].append(r_teta)
    return results


def main():
    os.makedirs(TEMP_DIR, exist_ok=True)

    # resume from progress file if it exists
    if os.path.exists(PROGRESS_FILE):
        saved = np.load(PROGRESS_FILE, allow_pickle=True)
        all_r_teta = {g: list(saved[g]) for g in GROUP_NAMES}
        processed_files = set(saved["_processed_files"].tolist())
        print(f"Resuming: {len(processed_files)} files already processed")
    else:
        all_r_teta = {g: [] for g in GROUP_NAMES}
        processed_files = set()

    print("=== Listing all raw recording files in the archive ===")
    with RemoteZip(ZENODO_URL) as zf:
        all_files = zf.namelist()
        pdata_files = sorted([f for f in all_files if "pData_SIMA_only_m.mat" in f])
    print(f"Total files: {len(pdata_files)}")

    remaining = [f for f in pdata_files if f not in processed_files]
    print(f"Remaining to process: {len(remaining)}\n")

    for idx, target in enumerate(remaining):
        local_path = os.path.join(TEMP_DIR, os.path.basename(target))
        t0 = time.time()
        try:
            with RemoteZip(ZENODO_URL) as zf:
                zf.extract(target, path=TEMP_DIR)
                # remotezip preserves the DATA/ subfolder; find actual path
                extracted_path = os.path.join(TEMP_DIR, target)

            file_results = extract_r_teta_from_file(extracted_path)
            n_cells_this_file = sum(len(v) for v in file_results.values())
            for g in GROUP_NAMES:
                all_r_teta[g].extend(file_results[g])

            os.remove(extracted_path)  # discard raw file immediately

            processed_files.add(target)
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(remaining)}] {os.path.basename(target)}: "
                  f"{n_cells_this_file} cells extracted ({elapsed:.1f}s)")

        except Exception as e:
            print(f"  [{idx+1}/{len(remaining)}] {os.path.basename(target)}: "
                  f"FAILED ({e}) -- skipping, will retry on next run")
            continue

        # save progress every 10 files
        if (idx + 1) % 10 == 0 or idx == len(remaining) - 1:
            save_dict = {g: np.array(all_r_teta[g], dtype=object) for g in GROUP_NAMES}
            save_dict["_processed_files"] = np.array(list(processed_files), dtype=object)
            np.savez(PROGRESS_FILE, **save_dict)
            total_cells = sum(len(v) for v in all_r_teta.values())
            print(f"    -- progress saved: {len(processed_files)} files, "
                  f"{total_cells} total cells so far")

    print("\n=== Extraction complete ===")
    for g in GROUP_NAMES:
        print(f"  {g}: {len(all_r_teta[g])} cells")
    total = sum(len(v) for v in all_r_teta.values())
    print(f"  TOTAL: {total} (compare against 3537 from the Z-based extraction)")

    print("\n=== Building reference DIRECTLY from real R_teta, no von Mises "
          "reconstruction ===")
    population_matrix = np.zeros((8, len(GROUP_NAMES)))
    for col, g in enumerate(GROUP_NAMES):
        if len(all_r_teta[g]) == 0:
            print(f"  WARNING: no cells for {g}, leaving column as zero")
            continue
        arr = np.array(all_r_teta[g])  # (n_cells, 8)
        population_matrix[:, col] = arr.mean(axis=0)

    np.save("raw_population_matrix.npy", population_matrix)
    print("Saved raw_population_matrix.npy")
    print(population_matrix)


if __name__ == "__main__":
    main()
