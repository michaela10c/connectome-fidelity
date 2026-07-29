"""
Cross-Individual Substitutability Test (Randal Koene's second question)

Question, precisely: can recording from OTHER individuals' corresponding
cells substitute for recording from the SAME individual's own cells?

The Henning et al. (2022) reference used throughout this paper is already,
implicitly, an answer of "yes" to this question: it pools recordings across
14 flies and treats the pooled result as a single reference, used to
validate models built on a DIFFERENT connectome (Flyvis), not derived from
any of these 14 flies specifically. That assumption has never been tested
directly. This script tests it.

METHOD: leave-one-fly-out cross-individual generalization, directly
parallel to the split-half reliability test already used to validate this
same dataset (see build_henning_reference.py), except the split is defined
by INDIVIDUAL ANIMAL IDENTITY rather than random half-sampling of pooled
cells.

For each of the 14 flies:
  1. Build a reference from that fly's OWN cells only ("held-out").
  2. Build a reference from the OTHER 13 flies' pooled cells ("donor pool").
  3. Correlate the two (Spearman, matching the paper's RSA methodology).

This gives 14 leave-one-fly-out correlations. Compared against a matched
random split-half baseline recomputed on this same raw data (for a fair,
apples-to-apples comparison, not just citing the earlier summary numbers),
this directly answers Randal's question:
  - If leave-one-fly-out correlations are comparable to random split-half,
    that supports "yes, donor/pooled recordings are a reasonable stand-in."
  - If leave-one-fly-out correlations are substantially LOWER, that means
    individual identity matters, and pooled data understates true
    cross-individual variability -- an important, honest limitation on
    the existing methodology, not a failed test.

Uses the SAME reconstruction methodology as build_henning_reference.py
(raw per-cell averaging, and von Mises curve fitting) for full
consistency with the paper's existing, validated approach.

USAGE:
    python cross_individual_substitutability.py
    (run from the directory containing henning_data/DATA/, or pass
     --data_path directly)
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import scipy.io as sio
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

LAYER_CELLTYPES = ['T4A', 'T4B', 'T4C', 'T4D', 'T5A', 'T5B', 'T5C', 'T5D']
N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)


def kappa_from_mean_resultant_length(r, max_kappa=50.0):
    """Same as build_henning_reference.py -- inverts r = I1(kappa)/I0(kappa)."""
    r = np.clip(r, 0, 0.999)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(
            r < 0.53,
            2 * r + r**3 + 5 * r**5 / 6,
            np.where(
                r < 0.85,
                -0.4 + 1.39 * r + 0.43 / (1 - r),
                1 / (r**3 - 4 * r**2 + 3 * r),
            ),
        )
    return np.clip(kappa, 0, max_kappa)


def von_mises_curve(theta_deg, pref_deg, kappa, amplitude):
    """Same as build_henning_reference.py."""
    theta = np.radians(theta_deg)
    pref = np.radians(pref_deg)
    return amplitude * np.exp(kappa * (np.cos(theta - pref) - 1))


def extract_animal_id(flyname):
    """'200728_Fly1_11_Image11_pData_SIMA_only_m.mat' -> '200728_Fly1'."""
    m = re.match(r'(\d+_Fly\d+)_', flyname)
    return m.group(1) if m else flyname


def load_cells_by_animal(mat_path):
    """Load all cells' complex tuning vectors (Z), organized by
    animal_id -> cell_type -> list of Z values.
    """
    d = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    T4T5_mb = d['T4T5_mb']

    cells = defaultdict(lambda: defaultdict(list))
    for rec in T4T5_mb:
        animal = extract_animal_id(rec.Flyname)
        for ct in LAYER_CELLTYPES:
            arr = np.atleast_1d(getattr(rec.Z, ct))
            if arr.size > 0 and np.iscomplexobj(arr):
                cells[animal][ct].extend(arr.tolist())
    return cells


def build_profile_raw(z_values):
    """Raw reconstruction: average each cell's peak response directly
    across the 8 native directions, no curve-fitting. Matches
    build_henning_reference.py's 'raw' method in spirit -- since raw Z
    values here are single complex summaries (not full 8-point curves in
    this file), we approximate the raw profile via the per-cell resultant
    projected onto each of the 8 directions (cos of angular difference,
    scaled by |Z|), then average across cells. This is a simplified
    per-cell-averaged raw profile.
    """
    if len(z_values) == 0:
        return None
    z_arr = np.array(z_values)
    amplitudes = np.abs(z_arr)
    angles_deg = np.degrees(np.angle(z_arr))
    profile = np.zeros(N_DIRECTIONS)
    for i, theta in enumerate(STIMULUS_DIRECTIONS_DEG):
        # each cell's raw contribution at this direction: projection of its
        # resultant vector onto this stimulus direction
        contributions = amplitudes * np.cos(
            np.radians(theta - angles_deg)
        )
        profile[i] = np.mean(contributions)
    return profile


def build_profile_von_mises(z_values):
    """Von Mises reconstruction: each cell gets its own fitted preferred
    direction and concentration, curves averaged. Matches
    build_henning_reference.py exactly.
    """
    if len(z_values) == 0:
        return None
    z_arr = np.array(z_values)
    amplitudes = np.abs(z_arr)
    pref_deg = np.degrees(np.angle(z_arr))
    kappa = kappa_from_mean_resultant_length(amplitudes)
    curves = np.array([
        von_mises_curve(STIMULUS_DIRECTIONS_DEG, pref_deg[i], kappa[i], amplitudes[i])
        for i in range(len(z_arr))
    ])
    return curves.mean(axis=0)


def build_reference_rdm(cells_by_celltype, method):
    """Build an 8x8 RDM across the LAYER_CELLTYPES, using either 'raw' or
    'von_mises' profile reconstruction. Returns None if fewer than 2 cell
    types have data (RDM undefined).
    """
    build_fn = build_profile_raw if method == "raw" else build_profile_von_mises
    profiles = []
    valid_types = []
    for ct in LAYER_CELLTYPES:
        z_values = cells_by_celltype.get(ct, [])
        profile = build_fn(z_values)
        if profile is not None:
            profiles.append(profile)
            valid_types.append(ct)
    if len(profiles) < 3:
        return None, valid_types
    profiles = np.array(profiles)  # (n_valid_types, 8)
    eps = 1e-10
    rdm = squareform(pdist(profiles + eps, metric="cosine"))
    return rdm, valid_types


def rdm_upper_triangle(rdm):
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def compare_rdms_matched_types(cells_a, cells_b, method):
    """Build RDMs from two cell groups, restricted to cell types present
    in BOTH groups (so the RDMs are the same shape), then Spearman
    correlate their upper triangles.
    """
    types_a = set(ct for ct in LAYER_CELLTYPES if cells_a.get(ct))
    types_b = set(ct for ct in LAYER_CELLTYPES if cells_b.get(ct))
    common_types = sorted(types_a & types_b)
    if len(common_types) < 3:
        return None, common_types

    build_fn = build_profile_raw if method == "raw" else build_profile_von_mises
    profiles_a = np.array([build_fn(cells_a[ct]) for ct in common_types])
    profiles_b = np.array([build_fn(cells_b[ct]) for ct in common_types])

    eps = 1e-10
    rdm_a = squareform(pdist(profiles_a + eps, metric="cosine"))
    rdm_b = squareform(pdist(profiles_b + eps, metric="cosine"))

    r, p = spearmanr(rdm_upper_triangle(rdm_a), rdm_upper_triangle(rdm_b))
    return r, common_types


def leave_one_fly_out(cells_by_animal, method):
    """For each fly, correlate its own reference against the pooled
    reference built from all OTHER flies. Returns list of (animal, r,
    n_common_types).
    """
    animals = sorted(cells_by_animal.keys())
    results = []
    for held_out in animals:
        held_out_cells = cells_by_animal[held_out]

        pooled_cells = defaultdict(list)
        for other in animals:
            if other == held_out:
                continue
            for ct, zs in cells_by_animal[other].items():
                pooled_cells[ct].extend(zs)

        r, common_types = compare_rdms_matched_types(
            held_out_cells, pooled_cells, method
        )
        results.append((held_out, r, len(common_types)))
    return results


def random_split_half(cells_by_animal, method, n_trials=500, seed=42):
    """Matched baseline: pool ALL cells (ignoring animal identity), then
    repeatedly split randomly into two halves and correlate. This is the
    same test already used to validate this dataset, recomputed here on
    this exact data path for a fair, apples-to-apples comparison against
    the leave-one-fly-out result above.
    """
    rng = np.random.default_rng(seed)
    all_cells = defaultdict(list)
    for animal_cells in cells_by_animal.values():
        for ct, zs in animal_cells.items():
            all_cells[ct].extend(zs)

    build_fn = build_profile_raw if method == "raw" else build_profile_von_mises
    valid_types = [ct for ct in LAYER_CELLTYPES if len(all_cells[ct]) >= 4]

    trial_results = []
    for _ in range(n_trials):
        half_a = {}
        half_b = {}
        for ct in valid_types:
            zs = all_cells[ct]
            idx = rng.permutation(len(zs))
            half = len(zs) // 2
            half_a[ct] = [zs[i] for i in idx[:half]]
            half_b[ct] = [zs[i] for i in idx[half:]]

        profiles_a = np.array([build_fn(half_a[ct]) for ct in valid_types])
        profiles_b = np.array([build_fn(half_b[ct]) for ct in valid_types])
        eps = 1e-10
        rdm_a = squareform(pdist(profiles_a + eps, metric="cosine"))
        rdm_b = squareform(pdist(profiles_b + eps, metric="cosine"))
        r, _ = spearmanr(rdm_upper_triangle(rdm_a), rdm_upper_triangle(rdm_b))
        if not np.isnan(r):
            trial_results.append(r)
    return np.array(trial_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        default="henning_data/DATA/processed_Data_SIMA_CS5_sh_Edges.mat",
        help="Path to the raw Henning .mat file",
    )
    args = parser.parse_args()

    print(f"Loading {args.data_path} ...")
    cells_by_animal = load_cells_by_animal(args.data_path)
    print(f"  {len(cells_by_animal)} individual flies found")

    for method in ["raw", "von_mises"]:
        print(f"\n{'=' * 60}")
        print(f"METHOD: {method}")
        print(f"{'=' * 60}")

        print("\nRunning leave-one-fly-out cross-individual test ...")
        loo_results = leave_one_fly_out(cells_by_animal, method)
        loo_rs = [r for (_, r, _) in loo_results if r is not None]
        for animal, r, n_types in loo_results:
            r_str = f"{r:.3f}" if r is not None else "N/A (too few common types)"
            print(f"  {animal:15s} r = {r_str}  ({n_types} cell types)")

        print(f"\nLeave-one-fly-out: mean r = {np.mean(loo_rs):.3f}, "
              f"std = {np.std(loo_rs):.3f}, n = {len(loo_rs)}/14 flies")

        print("\nRunning matched random split-half baseline (500 trials) ...")
        split_half_rs = random_split_half(cells_by_animal, method)
        print(f"Random split-half: mean r = {np.mean(split_half_rs):.3f}, "
              f"std = {np.std(split_half_rs):.3f}, n = {len(split_half_rs)} trials")

        gap = np.mean(split_half_rs) - np.mean(loo_rs)
        print(f"\nGap (split-half minus leave-one-fly-out): {gap:.3f}")
        if gap > 0.1:
            print("  -> Meaningful gap: cross-individual substitution is "
                  "notably less reliable than within-pool splitting.")
        elif gap > 0.03:
            print("  -> Modest gap: some evidence individual identity matters, "
                  "but not dramatically.")
        else:
            print("  -> Small/no gap: cross-individual data performs "
                  "comparably to within-pool splitting.")


if __name__ == "__main__":
    main()
