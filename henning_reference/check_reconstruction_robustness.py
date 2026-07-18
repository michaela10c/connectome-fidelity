"""
check_reconstruction_robustness.py -- tests whether the negative
correlation between Flyvis models and the Henning reference is an
artifact of the specific von Mises reconstruction choice, or survives
under reasonable alternative modeling assumptions. Does not require raw
per-direction data -- tests sensitivity to the MODELING CHOICE itself,
using the same per-cell Z (angle, magnitude) already extracted.

Two alternatives tested, each rebuilding the full reference and re-running
the identical comparison pipeline against CC and random:

  1. WRAPPED GAUSSIAN tuning curve instead of von Mises. Different curve
     shape (Gaussian-like tails vs. von Mises' specific exp(cos) form),
     with width set via the standard circular-statistics formula
     circular_sd = sqrt(-2*ln(R)) (R = mean resultant length = |Z|),
     an equally standard alternative to the kappa-inversion used
     originally.

  2. PERTURBED KAPPA: the original kappa-inversion scaled by 0.5x and
     2x, testing whether moderate uncertainty in tuning-sharpness
     estimation changes the sign or robustness of the result.

If the negative correlation survives all three variants (original,
wrapped Gaussian, perturbed kappa), that's real evidence it isn't an
artifact of one specific parametric choice. If it flips sign or
collapses under any of them, that's important to know before treating
this as a genuine finding.

USAGE:
    python check_reconstruction_robustness.py
    (needs the same .mat files as build_henning_reference.py, plus
    results_exp1_8dir_50models_full_shiu.npz)
"""

import numpy as np
import scipy.io as sio
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import pdist, squareform

DATA_DIR = "henning_data/DATA"
LAYER_CELLTYPES = ['T4A', 'T4B', 'T4C', 'T4D', 'T5A', 'T5B', 'T5C', 'T5D']
N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)


def kappa_from_mean_resultant_length(r, max_kappa=50.0):
    """Original method, unchanged -- reused for the 'perturbed kappa'
    variants and as the baseline for comparison.
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


def circular_sd_from_mean_resultant_length(r, max_sd_deg=180.0):
    """Standard circular-statistics formula for circular standard
    deviation: sd = sqrt(-2*ln(R)), R = mean resultant length. An
    equally standard, independently-derived alternative to the
    kappa-inversion -- not a variant of it, a genuinely different
    formula from the same literature.
    """
    r = np.clip(r, 1e-6, 0.999)
    sd_rad = np.sqrt(-2 * np.log(r))
    sd_deg = np.degrees(sd_rad)
    return np.clip(sd_deg, 1.0, max_sd_deg)


def von_mises_curve(theta_deg, pref_deg, kappa, amplitude):
    theta = np.radians(theta_deg)
    pref = np.radians(pref_deg)
    return amplitude * np.exp(kappa * (np.cos(theta - pref) - 1))


def wrapped_gaussian_curve(theta_deg, pref_deg, sd_deg, amplitude):
    """Wrapped Gaussian: response falls off with squared circular
    distance from the preferred direction, width set by sd_deg. Genuinely
    different tail behavior from von Mises, not a reparameterization of it.
    """
    diff = np.abs(theta_deg - pref_deg) % 360
    circ_diff = np.minimum(diff, 360 - diff)
    return amplitude * np.exp(-(circ_diff ** 2) / (2 * sd_deg ** 2))


def rank_residualize(rdm, against_rdm):
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    r_vals = rankdata(rdm[iu])
    r_against = rankdata(against_rdm[iu])
    slope, intercept = np.polyfit(r_against, r_vals, 1)
    predicted = slope * r_against + intercept
    return r_vals - predicted


def build_reference(all_z, curve_fn, param_fn, param_scale=1.0):
    """General reference-builder: curve_fn is either von_mises_curve or
    wrapped_gaussian_curve; param_fn converts |Z| to that curve's width
    parameter (kappa or sd); param_scale multiplies the resulting
    parameter, used for the perturbation variants.
    """
    population_matrix = np.zeros((N_DIRECTIONS, len(LAYER_CELLTYPES)))
    for col, f in enumerate(LAYER_CELLTYPES):
        z_arr = np.array(all_z[f])
        pref_deg = np.degrees(np.angle(z_arr))
        magnitude = np.abs(z_arr)
        param = param_fn(magnitude) * param_scale

        curves = np.array([
            curve_fn(STIMULUS_DIRECTIONS_DEG, pref_deg[i], param[i], magnitude[i])
            for i in range(len(z_arr))
        ])
        population_matrix[:, col] = curves.mean(axis=0)
    return population_matrix


def evaluate_variant(name, population_matrix, cc_resid, rand_resid, circ_ref):
    rdm = squareform(pdist(population_matrix, metric='cosine'))
    ref_resid = rank_residualize(rdm, circ_ref)
    r_cc, _ = spearmanr(cc_resid, ref_resid)
    r_rand, _ = spearmanr(rand_resid, ref_resid)
    iu = np.triu_indices(N_DIRECTIONS, k=1)
    r_circ, _ = spearmanr(rdm[iu], circ_ref[iu])
    print(f"  {name:30s}: CC r = {r_cc:+.4f}  |  Random r = {r_rand:+.4f}  "
          f"|  reference circularity r = {r_circ:.4f}")
    return r_cc, r_rand


def main():
    print("=== Loading per-cell Z data (same source as the original "
          "reference build) ===")
    d = sio.loadmat(f"{DATA_DIR}/processed_Data_SIMA_CS5_sh_Edges.mat",
                     squeeze_me=True, struct_as_record=False)
    T4T5_mb = d['T4T5_mb']

    all_z = {f: [] for f in LAYER_CELLTYPES}
    for rec in T4T5_mb:
        for f in LAYER_CELLTYPES:
            arr = np.atleast_1d(getattr(rec.Z, f))
            if arr.size > 0 and np.iscomplexobj(arr):
                all_z[f].extend(arr.tolist())
    total = sum(len(v) for v in all_z.values())
    print(f"  Total cells: {total} (expect 3537)")

    print("\n=== Loading Flyvis CC/random RDMs (fixed across all variants) ===")
    flyvis = np.load("results_exp1_8dir_50models_full_shiu.npz", allow_pickle=True)
    cc_rdm = flyvis["cc_rdm_cosine"]
    rand_rdm = flyvis["rand_rdm_cosine"]

    circ_ref = np.zeros((N_DIRECTIONS, N_DIRECTIONS))
    for i in range(N_DIRECTIONS):
        for j in range(N_DIRECTIONS):
            dd = abs(STIMULUS_DIRECTIONS_DEG[i] - STIMULUS_DIRECTIONS_DEG[j])
            circ_ref[i, j] = min(dd, 360 - dd)

    cc_resid = rank_residualize(cc_rdm, circ_ref)
    rand_resid = rank_residualize(rand_rdm, circ_ref)

    print("\n=== Testing whether the negative correlation survives under "
          "alternative, equally-defensible reconstruction choices ===\n")

    results = {}

    # Baseline: original von Mises + original kappa-inversion (should
    # reproduce the already-confirmed r=-0.4094 / r=-0.5753 exactly)
    pop_vonmises = build_reference(all_z, von_mises_curve, kappa_from_mean_resultant_length, 1.0)
    results['original (von Mises)'] = evaluate_variant(
        "Original (von Mises, baseline)", pop_vonmises, cc_resid, rand_resid, circ_ref)

    # Alternative 1: wrapped Gaussian curve shape, independently-derived width
    pop_gaussian = build_reference(all_z, wrapped_gaussian_curve,
                                   circular_sd_from_mean_resultant_length, 1.0)
    results['wrapped Gaussian'] = evaluate_variant(
        "Wrapped Gaussian (different shape)", pop_gaussian, cc_resid, rand_resid, circ_ref)

    # Alternative 2 & 3: perturbed kappa (narrower / broader tuning than estimated)
    pop_kappa_half = build_reference(all_z, von_mises_curve, kappa_from_mean_resultant_length, 0.5)
    results['kappa x0.5 (broader tuning)'] = evaluate_variant(
        "Kappa x0.5 (broader tuning)", pop_kappa_half, cc_resid, rand_resid, circ_ref)

    pop_kappa_double = build_reference(all_z, von_mises_curve, kappa_from_mean_resultant_length, 2.0)
    results['kappa x2 (sharper tuning)'] = evaluate_variant(
        "Kappa x2 (sharper tuning)", pop_kappa_double, cc_resid, rand_resid, circ_ref)

    print("\n=== Interpretation ===")
    all_cc_negative = all(r_cc < 0 for r_cc, r_rand in results.values())
    all_rand_negative = all(r_rand < 0 for r_cc, r_rand in results.values())
    all_cc_less_negative = all(
        results[k][0] > results[k][1] for k in results
    )

    print(f"  CC negative in all variants: {all_cc_negative}")
    print(f"  Random negative in all variants: {all_rand_negative}")
    print(f"  CC less negative than random in all variants: {all_cc_less_negative}")

    if all_cc_negative and all_rand_negative and all_cc_less_negative:
        print(f"\n  ROBUST: the qualitative pattern (both negative, CC less "
              f"negative than random) holds across every tested alternative "
              f"-- a different curve shape entirely, and a 4x range of "
              f"assumed tuning sharpness (0.5x to 2x). This is not an "
              f"artifact of the specific von Mises + kappa-inversion choice "
              f"originally used. Real, unexplained, and now checked against "
              f"reconstruction-method sensitivity as well as noise and "
              f"single-pair/single-group artifacts.")
    else:
        print(f"\n  NOT FULLY ROBUST: the pattern changes under at least one "
              f"alternative reconstruction choice. This means the specific "
              f"parametric assumptions in the original build matter for the "
              f"result -- worth identifying exactly which variant changes "
              f"things and why before treating this as a settled, "
              f"reconstruction-independent finding.")


if __name__ == "__main__":
    main()
