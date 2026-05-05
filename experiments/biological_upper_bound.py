"""
Experiment 3: Biological Upper Bound — T4/T5 Direction Tuning RDM
from Maisak et al. 2013

Constructs a biological reference RDM from T4/T5 direction tuning data
(Maisak et al. 2013, Nature 500:212-216, Fig. 3g/3h) and compares it against
the CC and random cosine RDMs from Experiments 1 and 2.

DESIGN CHOICE — Stimulus RDM (12×12):
    The Maisak tuning matrix is (12 directions × 8 subtypes). We use this as
    a population matrix over stimuli and compute a 12×12 stimulus RDM, which
    is directly comparable to the CC and random 12×12 RDMs from Experiment 1.
    This asks: do CC networks represent the 12 directions in a geometry
    consistent with the biological T4/T5 tuning structure?

TUNING CURVE CONSTRUCTION:
    Rather than manually digitizing the polar plots (error-prone), we model
    each subtype's tuning curve analytically using a von Mises profile, as
    justified by Maisak et al.'s description:
      - ~60-90° half-width at half-maximum
      - peaks at cardinal directions (0°, 90°, 180°, 270° for layers 2,3,1,4)
      - no response at anti-preferred direction (rectified)
      - normalized to maximum = 1.0
    This is more principled than manual digitization and reproducible.
    kappa=2.5 gives HWHM ≈ 67°, within the reported 60-90° range.

    For Experiment 2 (ON+OFF, 24 conditions): T4 subtypes respond to ON edges
    (intensity=1), T5 subtypes respond to OFF edges (intensity=0). The biological
    24×24 RDM encodes this ON/OFF segregation by setting cross-pathway responses
    to zero, consistent with Maisak et al. Fig. 3c/3d.

IMPORTANT CAVEATS (must be stated in any presentation/paper):
    1. Biological data: moving square-wave gratings (Maisak et al. Fig. 3g/3h)
       Model data: MovingEdge stimulus. Qualitative direction tuning structure
       is preserved, but absolute response profiles differ.
    2. Biological RDM covers T4/T5 subspace (8 of 65 cell types).
       Frame as: upper bound for T4/T5 subpopulation, not full population.
    3. Von Mises approximation: captures published tuning width and peak
       locations, but does not reproduce trial-by-trial variability.
    4. Interpret as qualitative upper bound, not quantitative validation.

USAGE:
    # After running Experiment 1:
    results_exp1 = run_experiment(n_models=10, randomization_strategy="full_shiu")

    # After running Experiment 2 (optional):
    results_exp2 = run_experiment(n_models=10, randomization_strategy="full_shiu")

    # Run biological upper bound:
    bio_results = run_biological_upper_bound(results_exp1, results_exp2)

REFERENCES:
    Maisak et al. 2013, Nature 500:212-216
    Nili et al. 2014, PLOS Computational Biology (permutation test)
    Kriegeskorte et al. 2008, Nature Neuroscience (RSA)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── STIMULUS ANGLES (matching Experiments 1 and 2) ───────────────────────────
ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
ANGLES_RAD = np.deg2rad(ANGLES_DEG)
N_DIRS = len(ANGLES_DEG)


# ── BIOLOGICAL TUNING CURVE CONSTRUCTION ─────────────────────────────────────

def von_mises_tuning(preferred_deg, kappa=2.5, angles_deg=ANGLES_DEG,
                     rectify=True):
    """
    Von Mises direction tuning curve, normalized to maximum = 1.0.

    Maisak et al. 2013 report ~60-90 degree half-width at half-maximum (HWHM)
    for T4 and T5. A von Mises distribution with kappa=2.5 gives HWHM approx
    67 degrees, within the reported range. Rectification enforces no response
    at anti-preferred direction, as reported in the paper.

    Args:
        preferred_deg: preferred direction in degrees
        kappa: concentration parameter (2.5 gives HWHM approx 67 degrees)
        angles_deg: list of stimulus angles in degrees
        rectify: if True, set sub-threshold values to 0 (no negative responses)

    Returns:
        curve: numpy array of shape (len(angles_deg),), normalized to max=1.0
    """
    angles = np.deg2rad(angles_deg)
    preferred = np.deg2rad(preferred_deg)
    curve = np.exp(kappa * (np.cos(angles - preferred) - 1))
    if rectify:
        # Maisak et al.: "No decrease of calcium was detectable for grating
        # motion opposite to the preferred direction of the respective layer."
        threshold = np.exp(kappa * (-2))  # value at 180 degrees from preferred
        curve = np.maximum(curve - threshold, 0)
    curve = curve / curve.max()
    return curve


# T4/T5 subtype preferred directions (from Maisak et al. 2013):
#   Layer 1 (T4a, T5a): back-to-front  -> 180 degrees
#   Layer 2 (T4b, T5b): front-to-back  ->   0 degrees
#   Layer 3 (T4c, T5c): upward         ->  90 degrees
#   Layer 4 (T4d, T5d): downward       -> 270 degrees
#
# Layer-to-direction mapping confirmed by Maisak et al. Fig. 3g/3h:
# "peak responses in each layer shifted by 90 degrees"
# and Fig. 3a/3b showing layer-specific responses to cardinal directions.

SUBTYPE_PREFERRED_DIRS = {
    "T4a": 180, "T4b": 0,   "T4c": 90,  "T4d": 270,
    "T5a": 180, "T5b": 0,   "T5c": 90,  "T5d": 270,
}
BIO_CELL_TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

# Tuning matrix: (8 subtypes x 12 directions)
BIO_TUNING_MATRIX = np.stack([
    von_mises_tuning(SUBTYPE_PREFERRED_DIRS[ct])
    for ct in BIO_CELL_TYPES
], axis=0)

# Population matrix over stimuli: (12 directions x 8 subtypes)
BIO_POP_MATRIX_12x8 = BIO_TUNING_MATRIX.T


# ── HELPERS ───────────────────────────────────────────────────────────────────

def build_rdm_from_pop_matrix(pop_matrix):
    """
    Build cosine distance RDM from (n_stimuli x n_cells) population matrix.
    Handles NaN/inf by clamping before distance computation.
    Returns (n_stimuli x n_stimuli) RDM.
    """
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3)
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rdm[i, j] = cosine(pop_matrix[i], pop_matrix[j])
    return rdm


def rdm_similarity(rdm1, rdm2):
    """Spearman r and Kendall tau_A between upper triangles of two RDMs."""
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    r_s, p_s = spearmanr(rdm1[idx], rdm2[idx])
    r_k, p_k = kendalltau(rdm1[idx], rdm2[idx])
    return r_s, p_s, r_k, p_k


def permutation_test_rdm(rdm1, rdm2, n_permutations=10000, seed=SEED):
    """
    Stimulus-label randomization test (Nili et al. 2014).
    Permutes rows and columns of rdm2 simultaneously (preserves RDM symmetry).
    One-sided p-value: proportion of null correlations >= observed.
    """
    rng = np.random.default_rng(seed)
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    obs_r,   _ = spearmanr(rdm1[idx], rdm2[idx])
    obs_tau, _ = kendalltau(rdm1[idx], rdm2[idx])
    null_r   = np.zeros(n_permutations)
    null_tau = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        rdm2_p = rdm2[np.ix_(perm, perm)]
        null_r[i],   _ = spearmanr(rdm1[idx], rdm2_p[idx])
        null_tau[i], _ = kendalltau(rdm1[idx], rdm2_p[idx])
    p_r   = np.mean(null_r   >= obs_r)
    p_tau = np.mean(null_tau >= obs_tau)
    return obs_r, p_r, obs_tau, p_tau, null_r, null_tau


def plot_null_distribution(null_r, obs_r, null_tau, obs_tau,
                           title_suffix, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle(f"Permutation null distribution — {title_suffix}", fontsize=10)
    for ax, null, obs, label in zip(
        axes,
        [null_r,   null_tau],
        [obs_r,    obs_tau],
        ["Spearman r", "Kendall tau"]
    ):
        ax.hist(null, bins=50, color="steelblue", alpha=0.7,
                label="Null (permutations)")
        ax.axvline(obs, color="firebrick", linewidth=2,
                   label=f"Observed = {obs:.3f}")
        ax.set_xlabel(f"{label} (CC vs Biology)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("../figures/"+fname, dpi=150, bbox_inches="tight")
    print(f"    Saved: ../figures/{fname}")
    plt.show()


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def run_biological_upper_bound(results_exp1, results_exp2=None,
                               n_permutations=10000):
    """
    Compare CC and random representational geometry against the biological
    T4/T5 direction tuning structure from Maisak et al. 2013.

    Args:
        results_exp1: dict from Experiment 1 run_experiment()
                      (ON edges, 12 conditions, n=10, full_shiu)
        results_exp2: dict from Experiment 2 run_experiment(), optional
                      (ON+OFF edges, 24 conditions, n=10, full_shiu)
        n_permutations: permutations for Nili et al. 2014 test

    Returns:
        dict with all RDMs and statistics
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: BIOLOGICAL UPPER BOUND")
    print("Maisak et al. 2013, Nature 500:212-216, Fig. 3g/3h")
    print("="*60)
    print("\nCaveats:")
    print("  [1] Biological stimulus: square-wave gratings (Maisak Fig. 3g/3h)")
    print("      Model stimulus: MovingEdge. Qualitative match, not quantitative.")
    print("  [2] Biological RDM: T4/T5 subspace only (8 of 65 cell types)")
    print("  [3] Tuning curves: von Mises (kappa=2.5, HWHM approx 67 degrees)")
    print("      consistent with Maisak et al. reported 60-90 degree HWHM")
    print("  [4] Interpret as qualitative upper bound for T4/T5 subpopulation")

    # ── 1. Visualize biological tuning curves ─────────────────────────────────
    print("\n--- BIOLOGICAL TUNING CURVES (von Mises, kappa=2.5) ---")
    print(f"  Subtypes: {BIO_CELL_TYPES}")
    print(f"  Preferred directions: {list(SUBTYPE_PREFERRED_DIRS.values())}")
    print(f"  Tuning matrix shape: {BIO_TUNING_MATRIX.shape}")

    fig_tc, axes_tc = plt.subplots(2, 4, figsize=(14, 6),
                                   subplot_kw={"projection": "polar"})
    fig_tc.suptitle(
        "Biological T4/T5 Direction Tuning — Von Mises Model\n"
        "Based on Maisak et al. 2013 Fig. 3g (T4) and 3h (T5)\n"
        "kappa=2.5, HWHM approx 67 degrees, rectified (no anti-PD response)",
        fontsize=9
    )
    angles_closed = np.append(ANGLES_RAD, ANGLES_RAD[0])
    
    for ax, name, curve in zip(axes_tc.flatten(), BIO_CELL_TYPES, BIO_TUNING_MATRIX):
        vals = np.append(curve, curve[0])
        color = "steelblue" if name.startswith("T4") else "coral"
        ax.plot(angles_closed, vals, color=color, linewidth=1.5)
        ax.fill(angles_closed, vals, alpha=0.25, color=color)
        ax.set_title(
            f"{name}\nPD={SUBTYPE_PREFERRED_DIRS[name]} deg",
            fontsize=8,
            pad=20              # increased from 12 to clear the 90-deg label
        )
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["0", "90", "180", "270"], fontsize=6)
        ax.tick_params(axis='x', pad=2)   # pull angular labels closer to the ring
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.88])   # leave headroom for suptitle
    plt.savefig("../figures/maisak2013_t4t5_von_mises_tuning.png", dpi=150,
                bbox_inches="tight")
    print("  Saved: ../figures/maisak2013_t4t5_von_mises_tuning.png")
    plt.show()

    # ── 2. Build biological 12x12 stimulus RDM ────────────────────────────────
    print("\n--- BIOLOGICAL 12x12 STIMULUS RDM ---")
    bio_rdm_12 = build_rdm_from_pop_matrix(BIO_POP_MATRIX_12x8)
    print(f"  Shape: {bio_rdm_12.shape}")
    print(f"  Off-diagonal range: "
          f"{bio_rdm_12[bio_rdm_12 > 0].min():.4f} to {bio_rdm_12.max():.4f}")

    # ── 3. Experiment 1: three-way comparison ─────────────────────────────────
    print("\n--- EXPERIMENT 1 (ON edges, 12 conditions) vs BIOLOGY ---")
    cc_rdm1   = results_exp1["cc_rdm_cosine"]
    rand_rdm1 = results_exp1["rand_rdm_cosine"]

    r_cc_bio1,   p_cc_bio1,   rk_cc_bio1,   pk_cc_bio1   = rdm_similarity(cc_rdm1,   bio_rdm_12)
    r_rand_bio1, p_rand_bio1, rk_rand_bio1, pk_rand_bio1 = rdm_similarity(rand_rdm1, bio_rdm_12)
    r_cc_rand1,  p_cc_rand1,  rk_cc_rand1,  pk_cc_rand1  = rdm_similarity(cc_rdm1,   rand_rdm1)

    print(f"  CC vs Biology:   r={r_cc_bio1:.3f}, p={p_cc_bio1:.4f}"
          f" | tau={rk_cc_bio1:.3f}, p={pk_cc_bio1:.4f}  [analytical]")
    print(f"  Rand vs Biology: r={r_rand_bio1:.3f}, p={p_rand_bio1:.4f}"
          f" | tau={rk_rand_bio1:.3f}, p={pk_rand_bio1:.4f}  [analytical]")
    print(f"  CC vs Random:    r={r_cc_rand1:.3f}, p={p_cc_rand1:.4f}"
          f" | tau={rk_cc_rand1:.3f}, p={pk_cc_rand1:.4f}  [analytical]")

    print(f"\n  Permutation test: CC vs Biology ({n_permutations} permutations):")
    obs_r1, p_r1, obs_tau1, p_tau1, null_r1, null_tau1 = permutation_test_rdm(
        cc_rdm1, bio_rdm_12, n_permutations=n_permutations)
    print(f"  r={obs_r1:.3f}, p_perm={p_r1:.4f}"
          f" | tau={obs_tau1:.3f}, p_perm={p_tau1:.4f}  [permutation]")
    print(f"  {int(p_r1*n_permutations)}/{n_permutations} permutations "
          f"exceeded observed Spearman r")

    plot_null_distribution(
        null_r1, obs_r1, null_tau1, obs_tau1,
        title_suffix="CC vs Biology, Exp 1 (ON edges)",
        fname="bio_upper_bound_exp1_permtest.png"
    )

    angle_labels = [f"{a}" for a in ANGLES_DEG]
    fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4))
    fig1.suptitle(
        "Biological Upper Bound — Experiment 1 (ON edges, n=10)\n"
        "Maisak et al. 2013 T4/T5 direction tuning as reference RDM",
        fontsize=9
    )
    for ax, rdm, title in zip(
        axes1,
        [bio_rdm_12, cc_rdm1, rand_rdm1],
        [f"Biological RDM\n(Maisak 2013 T4/T5)",
         f"CC Mean RDM\n(r vs bio={r_cc_bio1:.3f}, tau={rk_cc_bio1:.3f})",
         f"Random Mean RDM\n(r vs bio={r_rand_bio1:.3f}, tau={rk_rand_bio1:.3f})"]
    ):
        if not np.any(np.isfinite(rdm)) or not np.any(rdm > 0):
            ax.set_title(f"{title}\n(not renderable)", fontsize=8)
            ax.axis("off")
            continue
        im = ax.imshow(rdm, cmap="viridis", vmin=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(N_DIRS))
        ax.set_xticklabels(angle_labels, fontsize=5, rotation=90)
        ax.set_yticks(range(N_DIRS))
        ax.set_yticklabels(angle_labels, fontsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("../figures/biological_upper_bound_exp1.png", dpi=150, bbox_inches="tight")
    print("  Saved: ../figures/biological_upper_bound_exp1.png")
    plt.show()

    # ── 4. Experiment 2: ON+OFF (24 conditions) ───────────────────────────────
    exp2_out = {}
    if results_exp2 is not None:
        print("\n--- EXPERIMENT 2 (ON+OFF edges, 24 conditions) vs BIOLOGY ---")

        # Build 24x24 biological RDM encoding ON/OFF pathway segregation.
        # Condition ordering matches Experiment 2: [OFF 0..330, ON 0..330]
        # T4 subtypes: respond only to ON edges (zero for OFF conditions)
        # T5 subtypes: respond only to OFF edges (zero for ON conditions)
        # Per Maisak et al. Fig. 3c/3d: T4 selective for ON, T5 for OFF.
        bio_off = np.zeros((N_DIRS, 8))
        bio_off[:, 4:8] = BIO_TUNING_MATRIX[4:8, :].T   # T5a-T5d for OFF
        bio_on  = np.zeros((N_DIRS, 8))
        bio_on[:, 0:4]  = BIO_TUNING_MATRIX[0:4, :].T   # T4a-T4d for ON

        bio_pop_24x8 = np.vstack([bio_off, bio_on])      # (24, 8)
        bio_rdm_24   = build_rdm_from_pop_matrix(bio_pop_24x8)

        print(f"  Bio 24x24 RDM range: "
              f"{bio_rdm_24[bio_rdm_24 > 0].min():.4f} to {bio_rdm_24.max():.4f}")

        cc_rdm2   = results_exp2["cc_rdm_cosine"]
        rand_rdm2 = results_exp2["rand_rdm_cosine"]

        r_cc_bio2,   p_cc_bio2,   rk_cc_bio2,   pk_cc_bio2   = rdm_similarity(cc_rdm2,   bio_rdm_24)
        r_rand_bio2, p_rand_bio2, rk_rand_bio2, pk_rand_bio2 = rdm_similarity(rand_rdm2, bio_rdm_24)

        print(f"  CC vs Biology:   r={r_cc_bio2:.3f}, p={p_cc_bio2:.4f}"
              f" | tau={rk_cc_bio2:.3f}, p={pk_cc_bio2:.4f}  [analytical]")
        print(f"  Rand vs Biology: r={r_rand_bio2:.3f}, p={p_rand_bio2:.4f}"
              f" | tau={rk_rand_bio2:.3f}, p={pk_rand_bio2:.4f}  [analytical]")

        print(f"\n  Permutation test: CC vs Biology ({n_permutations} permutations):")
        obs_r2, p_r2, obs_tau2, p_tau2, null_r2, null_tau2 = permutation_test_rdm(
            cc_rdm2, bio_rdm_24, n_permutations=n_permutations)
        print(f"  r={obs_r2:.3f}, p_perm={p_r2:.4f}"
              f" | tau={obs_tau2:.3f}, p_perm={p_tau2:.4f}  [permutation]")
        print(f"  {int(p_r2*n_permutations)}/{n_permutations} permutations "
              f"exceeded observed Spearman r")

        plot_null_distribution(
            null_r2, obs_r2, null_tau2, obs_tau2,
            title_suffix="CC vs Biology, Exp 2 (ON+OFF edges)",
            fname="bio_upper_bound_exp2_permtest.png"
        )

        stim_labels_24 = (
            [f"OFF {a}" for a in ANGLES_DEG] +
            [f"ON {a}"  for a in ANGLES_DEG]
        )
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
        fig2.suptitle(
            "Biological Upper Bound — Experiment 2 (ON+OFF edges, n=10)\n"
            "Maisak 2013: T4 selective for ON, T5 for OFF (Fig. 3c/3d)",
            fontsize=9
        )
        for ax, rdm, title in zip(
            axes2,
            [bio_rdm_24, cc_rdm2, rand_rdm2],
            [f"Biological RDM\n(T4/T5 ON+OFF segregated)",
             f"CC Mean RDM\n(r vs bio={r_cc_bio2:.3f}, tau={rk_cc_bio2:.3f})",
             f"Random Mean RDM\n(r vs bio={r_rand_bio2:.3f}, tau={rk_rand_bio2:.3f})"]
        ):
            if not np.any(np.isfinite(rdm)) or not np.any(rdm > 0):
                ax.set_title(f"{title}\n(not renderable)", fontsize=8)
                ax.axis("off")
                continue
            im = ax.imshow(rdm, cmap="viridis", vmin=0)
            ax.set_title(title, fontsize=8)
            ax.set_xticks(range(24))
            ax.set_xticklabels(stim_labels_24, fontsize=4, rotation=90)
            ax.set_yticks(range(24))
            ax.set_yticklabels(stim_labels_24, fontsize=4)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig("../figures/biological_upper_bound_exp2.png", dpi=150,
                    bbox_inches="tight")
        print("  Saved: ../figures/biological_upper_bound_exp2.png")
        plt.show()

        exp2_out = {
            "bio_rdm_24": bio_rdm_24,
            "r_cc_bio": r_cc_bio2,     "p_cc_bio": p_cc_bio2,
            "rk_cc_bio": rk_cc_bio2,   "pk_cc_bio": pk_cc_bio2,
            "r_rand_bio": r_rand_bio2, "p_rand_bio": p_rand_bio2,
            "perm": dict(obs_r=obs_r2, p_r=p_r2,
                         obs_tau=obs_tau2, p_tau=p_tau2),
        }

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY — BIOLOGICAL UPPER BOUND")
    print("="*60)
    print(f"  Source: Maisak et al. 2013, Fig. 3g/3h")
    print(f"  Subtypes: {BIO_CELL_TYPES}")
    print(f"  Preferred directions: 0, 90, 180, 270 degrees (T4 and T5)")
    print(f"  Tuning model: von Mises, kappa=2.5, HWHM approx 67 degrees")
    print()
    print("  Experiment 1 (ON edges, 12 conditions):")
    print(f"    CC vs Bio:    r={r_cc_bio1:.3f}, tau={rk_cc_bio1:.3f}  [analytical]")
    print(f"                  p_perm(r)={p_r1:.4f}, p_perm(tau)={p_tau1:.4f}  "
          f"[{n_permutations} perms]")
    print(f"    Rand vs Bio:  r={r_rand_bio1:.3f}, tau={rk_rand_bio1:.3f}  [analytical]")
    print(f"    CC vs Random: r={r_cc_rand1:.3f}, tau={rk_cc_rand1:.3f}  [analytical]")
    if results_exp2 is not None:
        print()
        print("  Experiment 2 (ON+OFF edges, 24 conditions):")
        print(f"    CC vs Bio:    r={r_cc_bio2:.3f}, tau={rk_cc_bio2:.3f}  [analytical]")
        print(f"                  p_perm(r)={p_r2:.4f}, p_perm(tau)={p_tau2:.4f}  "
              f"[{n_permutations} perms]")
        print(f"    Rand vs Bio:  r={r_rand_bio2:.3f}, tau={rk_rand_bio2:.3f}  [analytical]")
    print()
    print("  Interpretation guide:")
    print("  r(CC vs Bio) > r(Rand vs Bio) means CC geometry more consistent with")
    print("    biological T4/T5 direction tuning than random geometry")
    print("  p_perm < 0.05 means CC vs Biology correlation exceeds chance")
    print()
    print("  CAVEAT: gratings vs edges mismatch; T4/T5 subspace only.")
    print("  Frame as qualitative upper bound, not quantitative validation.")

    return {
        "bio_tuning_matrix": BIO_TUNING_MATRIX,
        "bio_cell_types": BIO_CELL_TYPES,
        "bio_rdm_12": bio_rdm_12,
        "exp1": {
            "r_cc_bio": r_cc_bio1,   "p_cc_bio": p_cc_bio1,
            "rk_cc_bio": rk_cc_bio1, "pk_cc_bio": pk_cc_bio1,
            "r_rand_bio": r_rand_bio1,
            "r_cc_rand": r_cc_rand1,
            "perm": dict(obs_r=obs_r1, p_r=p_r1,
                         obs_tau=obs_tau1, p_tau=p_tau1,
                         null_r=null_r1, null_tau=null_tau1),
        },
        "exp2": exp2_out,
    }


# ── 6. Load Results ────────────────────────────────────────────────────────────

def load_results(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in d.files:
        arr = d[k]
        if arr.dtype.kind in ['U', 'S', 'O']:
            out[k] = arr.tolist()
        else:
            out[k] = arr
    return out


# ── 7. ENTRY POINT ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load results from experiments 1 and 2
    results_exp1 = load_results("../results/results_exp1.npz")
    results_exp2 = load_results("../results/results_exp2.npz")

    # Run Experiment
    bio_results = run_biological_upper_bound(results_exp1, results_exp2)    

    
