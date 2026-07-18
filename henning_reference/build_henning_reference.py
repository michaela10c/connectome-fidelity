"""
build_henning_reference.py -- constructs a candidate non-circular biological
reference RDM from the Henning, Ramos-Traslosheros, Guer & Silies (2022)
T4/T5 population tuning dataset, and immediately checks it for the same
circularity confound that broke the current Maisak-derived reference.

METHODOLOGY, stated explicitly because it's a real modeling choice, not a
re-derivation of raw data:

Each cell's `Z` field is a single complex number (a circular-statistics
vector-sum summary: Z = mean_i[ r_i * exp(1j*theta_i)] across the 8 measured
stimulus directions). It is NOT the full 8-point response curve -- that
information is not preserved in this file. To build a population response
PROFILE across the 8 stimulus directions (needed for an RDM), each cell's
tuning curve is reconstructed as a von Mises function:
    response(theta) = |Z| * exp(kappa * (cos(theta - angle(Z)) - 1))
parameterized by that cell's OWN fitted preferred direction (angle(Z)) and
OWN fitted concentration (kappa, inverted from |Z| via the standard circular
mean-resultant-length relationship). This is the same VON MISES RECONSTRUCTION
STRATEGY already used for the current (confounded) reference -- the
difference here is that every cell gets its OWN fitted direction and
sharpness from real data, instead of all cells in a layer sharing one
idealized, identical-width curve. This is what should break the circularity:
real per-cell heterogeneity in both angle AND concentration, not four
identical curves at 90-degree spacing.

This is still a modeling step, not raw ground truth -- if the per-recording
pData_SIMA_only_m.mat files contain full per-direction response curves
(not yet checked), that would be a strictly better foundation and should be
tried if this pipeline proves promising.

USAGE:
    python build_henning_reference.py
    (run from the directory containing henning_data/DATA/)
"""

import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

DATA_DIR = "henning_data/DATA"
LAYER_CELLTYPES = ['T4A', 'T4B', 'T4C', 'T4D', 'T5A', 'T5B', 'T5C', 'T5D']
N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)  # 0,45,...,315


def kappa_from_mean_resultant_length(r, max_kappa=50.0):
    """Inverts the standard circular-statistics relationship
    r = I1(kappa)/I0(kappa) (mean resultant length -> concentration).
    Uses the standard Fisher (1993) approximation, valid for 0 <= r < 1.
    r close to 1 -> very sharp tuning (high kappa); r close to 0 -> broad/
    untuned (kappa near 0). Clipped at max_kappa to avoid blowup as r -> 1.
    """
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
    """response(theta) = amplitude * exp(kappa * (cos(theta-pref) - 1)),
    normalized so response(pref) = amplitude, response(pref+180) -> ~0 for
    large kappa. Evaluated at the 8 native stimulus directions.
    """
    theta = np.radians(theta_deg)
    pref = np.radians(pref_deg)
    return amplitude * np.exp(kappa * (np.cos(theta - pref) - 1))


def main():
    print("=== Loading per-cell Z (complex tuning vectors) across all 114 "
          "recordings ===")
    d = sio.loadmat(f"{DATA_DIR}/processed_Data_SIMA_CS5_sh_Edges.mat",
                     squeeze_me=True, struct_as_record=False)
    T4T5_mb = d['T4T5_mb']

    all_z = {f: [] for f in LAYER_CELLTYPES}
    for rec in T4T5_mb:
        for f in LAYER_CELLTYPES:
            arr = np.atleast_1d(getattr(rec.Z, f))
            if arr.size > 0 and np.iscomplexobj(arr):
                all_z[f].extend(arr.tolist())

    for f in LAYER_CELLTYPES:
        print(f"  {f}: {len(all_z[f])} cells")
    total = sum(len(v) for v in all_z.values())
    print(f"  TOTAL: {total} (expect 3537, matching the Snob cluster file)")
    if total != 3537:
        print("  WARNING: count mismatch from the confirmed total -- "
              "check extraction logic before trusting anything downstream.")

    print("\n=== Reconstructing per-cell von Mises tuning curves, "
          "averaging within each of the 8 layer/celltype groups ===")
    population_matrix = np.zeros((N_DIRECTIONS, len(LAYER_CELLTYPES)))
    for col, f in enumerate(LAYER_CELLTYPES):
        z_arr = np.array(all_z[f])
        pref_deg = np.degrees(np.angle(z_arr))
        magnitude = np.abs(z_arr)
        kappa = kappa_from_mean_resultant_length(magnitude)

        # each cell's curve at the 8 native directions, then average
        # across all cells in this group -- this is the step where real
        # per-cell heterogeneity (in both pref_deg and kappa) should
        # produce a population profile that is NOT four idealized,
        # identical-width curves
        curves = np.array([
            von_mises_curve(STIMULUS_DIRECTIONS_DEG, pref_deg[i], kappa[i], magnitude[i])
            for i in range(len(z_arr))
        ])
        population_matrix[:, col] = curves.mean(axis=0)
        print(f"  {f}: mean preferred direction={pref_deg.mean():.1f} deg, "
              f"mean kappa={kappa.mean():.2f} (higher=sharper), "
              f"kappa std={kappa.std():.2f} (within-group spread)")

    print("\n=== Building the reference RDM (cosine distance, 8x8, native "
          "stimulus resolution) ===")
    rdm = squareform(pdist(population_matrix, metric='cosine'))
    print(f"  RDM shape: {rdm.shape}")
    print(f"  RDM off-diagonal range: {rdm[rdm > 0].min():.4f} to {rdm.max():.4f}")

    print("\n=== THE NON-NEGOTIABLE CHECK: is this new reference itself "
          "confounded with circular distance, the way the Maisak-derived "
          "one was? ===")
    circ_ref = squareform(pdist(
        STIMULUS_DIRECTIONS_DEG.reshape(-1, 1),
        metric=lambda a, b: min(abs(a[0] - b[0]), 360 - abs(a[0] - b[0]))
    ))
    rdm_upper = rdm[np.triu_indices(N_DIRECTIONS, k=1)]
    circ_upper = circ_ref[np.triu_indices(N_DIRECTIONS, k=1)]
    r_pearson, p_pearson = pearsonr(rdm_upper, circ_upper)
    r_spearman, p_spearman = spearmanr(rdm_upper, circ_upper)
    print(f"  Pearson r vs circular-distance reference:  {r_pearson:.4f}  (p={p_pearson:.2e})")
    print(f"  Spearman rho vs circular-distance reference: {r_spearman:.4f}  (p={p_spearman:.2e})")
    print(f"  (the retracted Maisak-derived reference scored r=0.978 on "
          f"this exact check -- that's the number this needs to beat)")

    print("\n=== Interpretation ===")
    if abs(r_pearson) > 0.9:
        print(f"  STILL SEVERELY CIRCULAR (r={r_pearson:.3f}). Real per-cell "
              f"heterogeneity in angle and kappa was not enough to escape "
              f"near-circular structure at this resolution. This reference "
              f"would likely fail the same way the current one did.")
    elif abs(r_pearson) > 0.7:
        print(f"  IMPROVED BUT STILL SUBSTANTIALLY CIRCULAR (r={r_pearson:.3f} "
              f"vs 0.978 for the retracted reference). Real progress, but "
              f"probably not enough on its own to fully resolve the "
              f"confound -- a partial-correlation correction would still "
              f"be needed, and would need to be checked for how much "
              f"signal survives it.")
    else:
        print(f"  MEANINGFULLY LESS CIRCULAR (r={r_pearson:.3f} vs 0.978 for "
              f"the retracted reference). This is genuinely promising -- "
              f"worth pursuing further, including the 8-vs-12-direction "
              f"stimulus mismatch with Flyvis (needs either interpolation "
              f"or re-simulating Flyvis under this exact 8-direction "
              f"paradigm) before this could actually be used.")

    np.save("henning_population_matrix.npy", population_matrix)
    np.save("henning_reference_rdm.npy", rdm)
    print(f"\nSaved population_matrix and reference_rdm as .npy files for "
          f"further use.")


if __name__ == "__main__":
    main()
