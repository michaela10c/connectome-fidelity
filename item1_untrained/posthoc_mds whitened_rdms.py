"""
Post-hoc Analysis: MDS Visualization and Noise-Whitened RDMs
for Connectome-Constrained Neural Emulations

Loads saved population matrices from Experiments 1 and 2 and computes:
1. MDS embedding of mean CC cosine RDM (visualizes circular direction geometry)
2. Noise-whitened RDMs using population covariance across models as noise estimate
3. RSA on whitened RDMs as a robustness check against the cosine RDM result

Operates entirely on saved .npz results — no Flyvis or GPU required.
Run on CPU runtime.

References:
- Kriegeskorte & Wei 2021. Neural tuning and representational geometry.
  Nature Reviews Neuroscience 22, 703–718.
- Nili et al. 2014. PLOS Computational Biology 10(4): e1003553.
- Maisak et al. 2013. Nature 500, 212–216.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from sklearn.manifold import MDS
from matplotlib.lines import Line2D

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SEED  = 42
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


# ── 1. HELPERS ────────────────────────────────────────────────────────────────

def build_cosine_rdm(pop_matrix):
    """Cosine distance RDM from (n_stim, n_units) population matrix."""
    pop_matrix = pop_matrix + 1e-10  # guard against zero-norm rows
    norms = np.linalg.norm(pop_matrix, axis=1, keepdims=True)
    normed = pop_matrix / norms
    sim = normed @ normed.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim  # cosine distance


def build_whitened_rdm(pop_matrix, noise_cov):
    """
    Mahalanobis-distance RDM using estimated noise covariance.

    Noise covariance is estimated from model-to-model variability in
    population vectors, serving as a proxy for noise in the response
    space (Kriegeskorte & Wei 2021).

    Args:
        pop_matrix: (n_stim, n_units) — mean population matrix across models
        noise_cov:  (n_units, n_units) — estimated noise covariance

    Returns:
        rdm: (n_stim, n_stim) Mahalanobis distance RDM
    """
    ridge = 1e-4 * np.trace(noise_cov) / noise_cov.shape[0]
    cov_reg = noise_cov + ridge * np.eye(noise_cov.shape[0])
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        print("    WARNING: covariance matrix singular — using pseudoinverse")
        cov_inv = np.linalg.pinv(cov_reg)
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = pop_matrix[i] - pop_matrix[j]
                rdm[i, j] = np.sqrt(np.maximum(diff @ cov_inv @ diff, 0))
    return rdm


def estimate_noise_cov(pop_matrices):
    """
    Estimate noise covariance from model-level variability.

    Treats each model's population matrix as one sample, computes
    the covariance of residuals from the mean across models.

    Args:
        pop_matrices: (n_models, n_stim, n_units)

    Returns:
        noise_cov: (n_units, n_units)
    """
    mean_pop  = pop_matrices.mean(axis=0)
    residuals = pop_matrices - mean_pop[np.newaxis, :, :]
    flat      = residuals.reshape(-1, pop_matrices.shape[2])
    return np.cov(flat.T)


def rdm_similarity(rdm1, rdm2):
    """Spearman r and Kendall tau on upper triangle."""
    n   = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    r_s, p_s = spearmanr(rdm1[idx], rdm2[idx])
    r_k, p_k = kendalltau(rdm1[idx], rdm2[idx])
    return r_s, p_s, r_k, p_k


def permutation_test_rdm(rdm1, rdm2, n_permutations=10000, seed=42):
    """Stimulus-label permutation test (Nili et al. 2014)."""
    rng     = np.random.default_rng(seed)
    n       = rdm1.shape[0]
    idx     = np.triu_indices(n, k=1)
    obs_r,   _ = spearmanr(rdm1[idx], rdm2[idx])
    obs_tau, _ = kendalltau(rdm1[idx], rdm2[idx])
    null_r   = np.zeros(n_permutations)
    null_tau = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm      = rng.permutation(n)
        rdm2_perm = rdm2[np.ix_(perm, perm)]
        null_r[i],   _ = spearmanr(rdm1[idx], rdm2_perm[idx])
        null_tau[i], _ = kendalltau(rdm1[idx], rdm2_perm[idx])
    p_r   = np.mean(null_r   >= obs_r)
    p_tau = np.mean(null_tau >= obs_tau)
    return obs_r, p_r, obs_tau, p_tau, null_r, null_tau


def test_circular_whitened(submatrix, label, circ_ref,
                            n_permutations=10000, seed=42):
    """Permutation test for circular structure on whitened RDM submatrix."""
    rng   = np.random.default_rng(seed)
    n     = submatrix.shape[0]
    idx   = np.triu_indices(n, k=1)
    obs_r, obs_p = spearmanr(submatrix[idx], circ_ref[idx])
    null_r = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm     = rng.permutation(n)
        sub_perm = submatrix[np.ix_(perm, perm)]
        null_r[i], _ = spearmanr(sub_perm[idx], circ_ref[idx])
    p_perm = np.mean(null_r >= obs_r)
    print(f"\n  {label}:")
    print(f"    Spearman r = {obs_r:.3f}, p = {obs_p:.4f}  [analytical]")
    print(f"    p_perm = {p_perm:.4f}  [{n_permutations} permutations]")
    print(f"    {int(p_perm * n_permutations)}/{n_permutations} permutations "
          f"exceeded observed r")
    return obs_r, p_perm, null_r


# ── 2. MAIN ───────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)

    results_dir = Path("../results")
    figures_dir = Path("../figures")
    figures_dir.mkdir(exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────────
    print("Loading Experiment 1 results (ON edges, n=50)...")
    exp1  = np.load(results_dir / "results_exp1_50models_full_shiu.npz",
                    allow_pickle=True)
    cc1   = exp1["cc_pop_matrices"]    # (50, 12, 65)
    rand1 = exp1["rand_pop_matrices"]  # (50, 12, 65)
    print(f"  cc_pop_matrices:   {cc1.shape}")
    print(f"  rand_pop_matrices: {rand1.shape}")

    print("\nLoading Experiment 2 results (ON+OFF edges, n=50)...")
    exp2  = np.load(results_dir / "results_exp2_50models_full_shiu.npz",
                    allow_pickle=True)
    cc2   = exp2["cc_pop_matrices"]    # (50, 24, 65)
    rand2 = exp2["rand_pop_matrices"]  # (50, 24, 65)
    print(f"  cc_pop_matrices:   {cc2.shape}")
    print(f"  rand_pop_matrices: {rand2.shape}")

    # ── Mean population matrices ──────────────────────────────────────────────
    cc1_mean   = cc1.mean(axis=0)    # (12, 65)
    rand1_mean = rand1.mean(axis=0)  # (12, 65)
    cc2_mean   = cc2.mean(axis=0)    # (24, 65)
    rand2_mean = rand2.mean(axis=0)  # (24, 65)

    # ── 3. MDS VISUALIZATION ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MDS VISUALIZATION")
    print("="*60)

    cc1_rdm   = build_cosine_rdm(cc1_mean)
    rand1_rdm = build_cosine_rdm(rand1_mean)
    cc2_rdm   = build_cosine_rdm(cc2_mean)
    rand2_rdm = build_cosine_rdm(rand2_mean)

    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=SEED, normalized_stress=False)
    cc1_mds   = mds.fit_transform(cc1_rdm)
    rand1_mds = mds.fit_transform(rand1_rdm)
    cc2_mds   = mds.fit_transform(cc2_rdm)
    rand2_mds = mds.fit_transform(rand2_rdm)

    cmap   = plt.cm.hsv
    colors = [cmap(a / 360) for a in ANGLES]

    # Exp 1: ON edges
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4.5))
    fig1.suptitle("MDS of Representational Geometry — Exp 1: ON Edges (n=50)",
                  fontsize=11)
    for ax, coords, title in zip(
        axes1,
        [cc1_mds, rand1_mds],
        ["Connectome-Constrained", "Stability-Constrained Random"]
    ):
        for k, (x, y) in enumerate(coords):
            ax.scatter(x, y, color=colors[k], s=120, zorder=3)
            ax.annotate(f"{ANGLES[k]}°", (x, y),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        for k in range(len(ANGLES)):
            x0, y0 = coords[k]
            x1, y1 = coords[(k + 1) % len(ANGLES)]
            ax.plot([x0, x1], [y0, y1], "gray", linewidth=0.8,
                    alpha=0.5, zorder=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("MDS dim 1")
        ax.set_ylabel("MDS dim 2")
        ax.axhline(0, color="lightgray", linewidth=0.5)
        ax.axvline(0, color="lightgray", linewidth=0.5)
        ax.set_aspect("equal")
    plt.tight_layout()
    fname_mds1 = str(figures_dir / "mds_exp1_on_edges_50models.png")
    fig1.savefig(fname_mds1, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname_mds1}")
    plt.close(fig1)

    # Exp 2: ON+OFF edges
    stim_labels_2 = []
    for a in ANGLES:
        stim_labels_2.append(f"OFF {a}°")
        stim_labels_2.append(f"ON {a}°")
    off_indices = list(range(0, 24, 2))
    on_indices  = list(range(1, 24, 2))

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
    fig2.suptitle("MDS of Representational Geometry — Exp 2: ON+OFF Edges (n=50)",
                  fontsize=11)
    for ax, coords, title in zip(
        axes2,
        [cc2_mds, rand2_mds],
        ["Connectome-Constrained", "Stability-Constrained Random"]
    ):
        for k in off_indices:
            angle = ANGLES[k // 2]
            ax.scatter(coords[k, 0], coords[k, 1],
                       color=cmap(angle / 360), s=100, marker="o",
                       edgecolors="black", linewidths=0.5, zorder=3)
        for k in on_indices:
            angle = ANGLES[k // 2]
            ax.scatter(coords[k, 0], coords[k, 1],
                       color=cmap(angle / 360), s=100, marker="^",
                       edgecolors="black", linewidths=0.5, zorder=3)
        for indices in [off_indices, on_indices]:
            for i in range(len(indices)):
                x0, y0 = coords[indices[i]]
                x1, y1 = coords[indices[(i + 1) % len(indices)]]
                ax.plot([x0, x1], [y0, y1], "gray", linewidth=0.8,
                        alpha=0.5, zorder=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("MDS dim 1")
        ax.set_ylabel("MDS dim 2")
        ax.axhline(0, color="lightgray", linewidth=0.5)
        ax.axvline(0, color="lightgray", linewidth=0.5)
        ax.set_aspect("equal")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", label="OFF edges"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=8, markeredgecolor="black", label="ON edges"),
    ]
    axes2[0].legend(handles=legend_elements, fontsize=8, loc="best")
    plt.tight_layout()
    fname_mds2 = str(figures_dir / "mds_exp2_on_off_edges_50models.png")
    fig2.savefig(fname_mds2, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname_mds2}")
    plt.close(fig2)

    # ── 4. NOISE-WHITENED RDMs ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("NOISE-WHITENED RDMs")
    print("="*60)
    print("\nNote: noise covariance estimated from model-level variability across")
    print("50 CC models (residuals from mean population matrix). This is a proxy")
    print("for response variability — not single-trial neural noise. Interpret")
    print("whitened RDM results as a robustness check, not a replacement for")
    print("the cosine RDM primary result.")

    # Exp 1
    print("\n--- Experiment 1: ON edges ---")
    noise_cov1 = estimate_noise_cov(cc1)
    print(f"  Noise covariance shape: {noise_cov1.shape}")
    print(f"  Condition number: {np.linalg.cond(noise_cov1):.2e}")

    cc1_whitened_rdm   = build_whitened_rdm(cc1_mean,   noise_cov1)
    rand1_whitened_rdm = build_whitened_rdm(rand1_mean,  noise_cov1)

    r1_w, p1_w, tau1_w, ptau1_w = rdm_similarity(
        cc1_whitened_rdm, rand1_whitened_rdm)
    print(f"\n  Whitened RDM correlation (CC vs Random):")
    print(f"  Spearman r = {r1_w:.3f}, p = {p1_w:.4f} | "
          f"Kendall τ = {tau1_w:.3f}, p = {ptau1_w:.4f}  [analytical]")

    obs_r1_w, p_r1_w, obs_tau1_w, p_tau1_w, null_r1_w, null_tau1_w = \
        permutation_test_rdm(cc1_whitened_rdm, rand1_whitened_rdm,
                              n_permutations=10000, seed=SEED)
    print(f"  Spearman r = {obs_r1_w:.3f}, p_perm = {p_r1_w:.4f} | "
          f"Kendall τ = {obs_tau1_w:.3f}, p_perm = {p_tau1_w:.4f}  [permutation]")
    print(f"\n  Reference — cosine RDM (primary result):")
    print(f"  Spearman r = 0.686, p < 0.0001 | Kendall τ = 0.515, p < 0.0001")

    # Exp 2
    print("\n--- Experiment 2: ON+OFF edges ---")
    noise_cov2 = estimate_noise_cov(cc2)
    print(f"  Noise covariance shape: {noise_cov2.shape}")
    print(f"  Condition number: {np.linalg.cond(noise_cov2):.2e}")

    cc2_whitened_rdm   = build_whitened_rdm(cc2_mean,   noise_cov2)
    rand2_whitened_rdm = build_whitened_rdm(rand2_mean,  noise_cov2)

    r2_w, p2_w, tau2_w, ptau2_w = rdm_similarity(
        cc2_whitened_rdm, rand2_whitened_rdm)
    print(f"\n  Whitened RDM correlation (CC vs Random):")
    print(f"  Spearman r = {r2_w:.3f}, p = {p2_w:.4f} | "
          f"Kendall τ = {tau2_w:.3f}, p = {ptau2_w:.4f}  [analytical]")

    obs_r2_w, p_r2_w, obs_tau2_w, p_tau2_w, null_r2_w, null_tau2_w = \
        permutation_test_rdm(cc2_whitened_rdm, rand2_whitened_rdm,
                              n_permutations=10000, seed=SEED)
    print(f"  Spearman r = {obs_r2_w:.3f}, p_perm = {p_r2_w:.4f} | "
          f"Kendall τ = {obs_tau2_w:.3f}, p_perm = {p_tau2_w:.4f}  [permutation]")
    print(f"\n  Reference — cosine RDM (primary result):")
    print(f"  Spearman r = 0.846, p < 0.0001 | Kendall τ = 0.651, p < 0.0001")

    # ── 5. WHITENED RDM FIGURE ────────────────────────────────────────────────
    angle_labels = [f"{a}°" for a in ANGLES]

    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
    fig3.suptitle(
        "Noise-Whitened RDMs — CC vs Random\n"
        "(Mahalanobis distance, noise cov estimated from model variability)",
        fontsize=10
    )
    for ax, rdm_cc, labels, title in zip(
        axes3,
        [cc1_whitened_rdm, cc2_whitened_rdm],
        [angle_labels, stim_labels_2],
        ["Exp 1: ON edges", "Exp 2: ON+OFF edges"]
    ):
        n  = rdm_cc.shape[0]
        im = ax.imshow(rdm_cc, cmap="viridis", vmin=0)
        ax.set_title(f"{title} — CC Whitened RDM", fontsize=8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=5, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fname_whitened = str(figures_dir / "whitened_rdms_exp1_exp2_50models.png")
    fig3.savefig(fname_whitened, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fname_whitened}")
    plt.close(fig3)

    # ── 5a. WITHIN-POLARITY WHITENED RDM FIGURE ──────────────────────────────
    print("\n--- WITHIN-POLARITY WHITENED RDM FIGURE (Exp 2) ---")

    off_idx = list(range(0, 24, 2))
    on_idx  = list(range(1, 24, 2))

    cc2_w_off_off   = cc2_whitened_rdm[np.ix_(off_idx, off_idx)]
    cc2_w_on_on     = cc2_whitened_rdm[np.ix_(on_idx,  on_idx)]
    rand2_w_off_off = rand2_whitened_rdm[np.ix_(off_idx, off_idx)]
    rand2_w_on_on   = rand2_whitened_rdm[np.ix_(on_idx,  on_idx)]

    vmax_cc_w   = max(cc2_w_off_off.max(),   cc2_w_on_on.max())
    vmax_rand_w = max(rand2_w_off_off.max(), rand2_w_on_on.max())

    fig4, axes4 = plt.subplots(2, 2, figsize=(8, 8.5))
    fig4.subplots_adjust(right=0.91, wspace=0.05, top=0.91)
    fig4.suptitle(
        "Within-Polarity Blocks: CC vs Random — Whitened RDMs\n"
        "ON+OFF edges, n=50, full Shiu-style shuffle\n"
        "(Mahalanobis distance, noise cov estimated from model variability)",
        fontsize=9
    )
    panels4 = [
        (axes4[0, 0], cc2_w_off_off,   "CC — OFF-OFF (whitened)",     vmax_cc_w),
        (axes4[0, 1], cc2_w_on_on,     "CC — ON-ON (whitened)",       vmax_cc_w),
        (axes4[1, 0], rand2_w_off_off, "Random — OFF-OFF (whitened)", vmax_rand_w),
        (axes4[1, 1], rand2_w_on_on,   "Random — ON-ON (whitened)",   vmax_rand_w),
    ]
    ims4 = []
    for ax, block, title, vmax_row in panels4:
        im = ax.imshow(block, cmap="viridis", vmin=0, vmax=vmax_row)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(12))
        ax.set_xticklabels(angle_labels, fontsize=7, rotation=90)
        ax.set_yticks(range(12))
        ax.set_yticklabels(angle_labels, fontsize=7)
        ims4.append(im)
    for ax in [axes4[0, 1], axes4[1, 1]]:
        ax.set_yticks([])
    fig4.canvas.draw()
    pos01 = axes4[0, 1].get_position()
    pos11 = axes4[1, 1].get_position()
    cax1 = fig4.add_axes([pos01.x1 + 0.01, pos01.y0, 0.02, pos01.height])
    cax2 = fig4.add_axes([pos11.x1 + 0.01, pos11.y0, 0.02, pos11.height])
    fig4.colorbar(ims4[0], cax=cax1)
    fig4.colorbar(ims4[2], cax=cax2)
    fname_wp_whitened = str(
        figures_dir / "within_polarity_blocks_whitened_exp2_50models.png")
    fig4.savefig(fname_wp_whitened, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname_wp_whitened}")
    plt.close(fig4)

    # ── 5b. WITHIN-POLARITY CIRCULAR STRUCTURE ON WHITENED RDMs ──────────────
    print("\n" + "="*60)
    print("WITHIN-POLARITY CIRCULAR STRUCTURE — WHITENED RDMs (Exp 2)")
    print("="*60)
    print("\nNote: within-polarity structure tested on whitened mean CC RDM.")
    print("Robustness check against the cosine RDM within-polarity result.")

    n_dirs   = 12
    circ_ref = np.zeros((n_dirs, n_dirs))
    for i in range(n_dirs):
        for j in range(n_dirs):
            circ_ref[i, j] = min(abs(i - j), n_dirs - abs(i - j))

    r_on_w,  p_perm_on_w,  _ = test_circular_whitened(
        cc2_w_on_on,   "ON-ON  block (whitened)",  circ_ref, seed=SEED)
    r_off_w, p_perm_off_w, _ = test_circular_whitened(
        cc2_w_off_off, "OFF-OFF block (whitened)", circ_ref, seed=SEED)

    print(f"\n  Whitened Δr = {r_on_w - r_off_w:.3f} "
          f"(ON-ON r = {r_on_w:.3f}, OFF-OFF r = {r_off_w:.3f})")
    print(f"\n  Reference — cosine RDM (primary result):")
    print(f"  ON-ON r = 0.937, p_perm < 0.0001")
    print(f"  OFF-OFF r = 0.799, p_perm < 0.0001")
    print(f"  Δr = 0.138")

    # ── 6. SUMMARY ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  MDS figures saved:")
    print(f"    {fname_mds1}")
    print(f"    {fname_mds2}")
    print()
    print(f"  Whitened RDM results:")
    print(f"    Exp 1: Spearman r = {obs_r1_w:.3f}, p_perm = {p_r1_w:.4f} "
          f"| τ = {obs_tau1_w:.3f}, p_perm = {p_tau1_w:.4f}")
    print(f"    Exp 2: Spearman r = {obs_r2_w:.3f}, p_perm = {p_r2_w:.4f} "
          f"| τ = {obs_tau2_w:.3f}, p_perm = {p_tau2_w:.4f}")
    print(f"  Whitened within-polarity (Exp 2):")
    print(f"    ON-ON r = {r_on_w:.3f}, p_perm = {p_perm_on_w:.4f} | "
          f"OFF-OFF r = {r_off_w:.3f}, p_perm = {p_perm_off_w:.4f} | "
          f"Δr = {r_on_w - r_off_w:.3f}")
    print(f"    {fname_wp_whitened}")
    print()
    print(f"  Interpretation:")
    print(f"  If whitened results converge with cosine RDM results,")
    print(f"  the fidelity claim is robust to noise covariance structure.")
    print(f"  Whitened results should be reported as a robustness check,")
    print(f"  not as the primary inference.")

    # ── 7. SAVE RESULTS ───────────────────────────────────────────────────────
    np.savez(
        str(results_dir / "posthoc_mds_whitened_50models_full_shiu.npz"),
        cc1_mds=cc1_mds,
        rand1_mds=rand1_mds,
        cc2_mds=cc2_mds,
        rand2_mds=rand2_mds,
        cc1_whitened_rdm=cc1_whitened_rdm,
        rand1_whitened_rdm=rand1_whitened_rdm,
        cc2_whitened_rdm=cc2_whitened_rdm,
        rand2_whitened_rdm=rand2_whitened_rdm,
        r1_whitened=obs_r1_w,    p1_whitened=p_r1_w,
        tau1_whitened=obs_tau1_w, ptau1_whitened=p_tau1_w,
        r2_whitened=obs_r2_w,    p2_whitened=p_r2_w,
        tau2_whitened=obs_tau2_w, ptau2_whitened=p_tau2_w,
        r_on_whitened=r_on_w,    p_perm_on_whitened=p_perm_on_w,
        r_off_whitened=r_off_w,  p_perm_off_whitened=p_perm_off_w,
        cc2_w_off_off_block=cc2_w_off_off,
        cc2_w_on_on_block=cc2_w_on_on,
        rand2_w_off_off_block=rand2_w_off_off,
        rand2_w_on_on_block=rand2_w_on_on,
    )
    print("\nSaved: ../results/posthoc_mds_whitened_50models_full_shiu.npz")


if __name__ == "__main__":
    main()
