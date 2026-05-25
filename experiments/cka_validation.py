"""
CKA Validation: Centered Kernel Alignment as Secondary Fidelity Metric
for Connectome-Constrained Neural Emulations

This notebook computes linear CKA (Kornblith et al. 2019) between connectome-
constrained (CC) and stability-constrained random network population matrices,
as an independent validation of the RSA-based fidelity result from Experiments
1 and 2.

CKA operates on raw activation matrices rather than RDMs, making it a genuinely
independent metric from RSA. Convergence of CKA and RSA strengthens the claim
that representational geometry discriminates biological from arbitrary wiring.

References:
- Kornblith et al. 2019. Similarity of Neural Network Representations Revisited.
  ICML 2019. https://arxiv.org/abs/1905.00414
- Lappalainen et al. 2024. Nature 634, 1132–1140.
- Nili et al. 2014. PLOS Computational Biology 10(4): e1003553.

Run on Google Colab (CPU runtime sufficient — no GPU needed).
Results files must be present in results/ from Experiments 1 and 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)

# ── 1. CKA IMPLEMENTATION (linear kernel) ────────────────────────────────────
# Kornblith et al. 2019, ICML

def center_kernel(K):
    """Double-center a kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def linear_cka(X, Y):
    """
    Linear CKA between activation matrices X and Y.

    Args:
        X: (n_stimuli, n_units) — population matrix
        Y: (n_stimuli, n_units) — population matrix

    Returns:
        scalar CKA in [0, 1]; 1 = identical geometry, 0 = no similarity
    """
    KX = X @ X.T
    KY = Y @ Y.T
    KX_c = center_kernel(KX)
    KY_c = center_kernel(KY)
    hsic_xy = np.sum(KX_c * KY_c)
    hsic_xx = np.sum(KX_c * KX_c)
    hsic_yy = np.sum(KY_c * KY_c)
    if hsic_xx == 0 or hsic_yy == 0:
        return float("nan")
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

def bootstrap_cka_ci(cc_matrices, rand_matrices, n_bootstrap=10000, seed=42):
    """
    Model-level bootstrap 95% CI for CKA(mean CC, mean random).
    Resamples models with replacement on each iteration.

    Args:
        cc_matrices:   (n_models, n_stimuli, n_units)
        rand_matrices: (n_models, n_stimuli, n_units)
        n_bootstrap:   number of bootstrap samples
        seed:          random seed

    Returns:
        ci_low, ci_high: 2.5th and 97.5th percentiles of bootstrap distribution
        null_cka:        full bootstrap distribution (for plotting)
    """
    rng = np.random.default_rng(seed)
    n_models = cc_matrices.shape[0]
    cka_vals = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_models, size=n_models)
        X = cc_matrices[idx].mean(axis=0)
        Y = rand_matrices[idx].mean(axis=0)
        cka_vals[i] = linear_cka(X, Y)
    ci_low, ci_high = np.percentile(cka_vals, [2.5, 97.5])
    return ci_low, ci_high, cka_vals

def permutation_test_cka(cc_matrices, rand_matrices,
                          n_permutations=10000, seed=42):
    """
    Stimulus-label permutation test for CKA.
    Permutes stimulus rows of mean random matrix to build null distribution.
    One-sided p-value: proportion of permuted CKA >= observed CKA.

    Args:
        cc_matrices:   (n_models, n_stimuli, n_units)
        rand_matrices: (n_models, n_stimuli, n_units)

    Returns:
        obs_cka: observed CKA
        p_val:   permutation p-value
        null_cka: null distribution
    """
    rng = np.random.default_rng(seed)
    X = cc_matrices.mean(axis=0)    # (n_stimuli, n_units)
    Y = rand_matrices.mean(axis=0)  # (n_stimuli, n_units)
    obs_cka = linear_cka(X, Y)
    n_stim = X.shape[0]
    null_cka = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_stim)
        null_cka[i] = linear_cka(X, Y[perm])
    p_val = np.mean(null_cka >= obs_cka)
    return obs_cka, p_val, null_cka


def main():
    # ── 2. LOAD RESULTS ───────────────────────────────────────────────────────
    results_dir = Path("../results")

    print("Loading Experiment 1 results (ON edges, n=50)...")
    exp1 = np.load(
        results_dir / "results_exp1_50models_full_shiu.npz",
        allow_pickle=True
    )
    cc1   = exp1["cc_pop_matrices"]    # (50, 12, 65)
    rand1 = exp1["rand_pop_matrices"]  # (50, 12, 65)
    print(f"  cc_pop_matrices:   {cc1.shape}")
    print(f"  rand_pop_matrices: {rand1.shape}")

    print("\nLoading Experiment 2 results (ON+OFF edges, n=50)...")
    exp2 = np.load(
        results_dir / "results_exp2_50models_full_shiu.npz",
        allow_pickle=True
    )
    cc2   = exp2["cc_pop_matrices"]    # (50, 24, 65)
    rand2 = exp2["rand_pop_matrices"]  # (50, 24, 65)
    print(f"  cc_pop_matrices:   {cc2.shape}")
    print(f"  rand_pop_matrices: {rand2.shape}")

    # ── 3. COMPUTE CKA ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CKA RESULTS")
    print("="*60)

    cc1_mean   = cc1.mean(axis=0)
    rand1_mean = rand1.mean(axis=0)
    cc2_mean   = cc2.mean(axis=0)
    rand2_mean = rand2.mean(axis=0)

    cka_exp1 = linear_cka(cc1_mean, rand1_mean)
    cka_exp2 = linear_cka(cc2_mean, rand2_mean)

    print(f"\nExp 1 (ON edges, 12 cond.)     — CKA(CC, Random) = {cka_exp1:.4f}")
    print(f"Exp 2 (ON+OFF edges, 24 cond.) — CKA(CC, Random) = {cka_exp2:.4f}")

    print(f"\n--- PERMUTATION TEST (10,000 permutations) ---")
    obs1, p1, null1 = permutation_test_cka(cc1, rand1, seed=SEED)
    obs2, p2, null2 = permutation_test_cka(cc2, rand2, seed=SEED)
    print(f"Exp 1: CKA = {obs1:.4f}, p = {p1:.4f}")
    print(f"Exp 2: CKA = {obs2:.4f}, p = {p2:.4f}")

    print(f"\n--- BOOTSTRAP 95% CI (10,000 samples, model-level resampling) ---")
    ci1_low, ci1_high, boot1 = bootstrap_cka_ci(cc1, rand1, seed=SEED)
    ci2_low, ci2_high, boot2 = bootstrap_cka_ci(cc2, rand2, seed=SEED)
    print(f"Exp 1: 95% CI [{ci1_low:.4f}, {ci1_high:.4f}]")
    print(f"Exp 2: 95% CI [{ci2_low:.4f}, {ci2_high:.4f}]")

    # ── 4. FIGURE ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "CKA Validation: Connectome-Constrained vs Stability-Constrained Random\n"
        "(n=50, seed=42, Kornblith et al. 2019)",
        fontsize=11
    )

    for ax, null, obs, p, title in zip(
        axes[0],
        [null1, null2],
        [obs1, obs2],
        [p1, p2],
        ["Exp 1: ON edges (12 cond.)", "Exp 2: ON+OFF edges (24 cond.)"]
    ):
        ax.hist(null, bins=50, color="steelblue", alpha=0.7,
                label="Null distribution")
        ax.axvline(obs, color="red", linewidth=2,
                   label=f"Observed CKA = {obs:.4f}")
        ax.set_xlabel("CKA (permuted)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\np = {p:.4f}")
        ax.legend(fontsize=8)

    for ax, boot, obs, ci_low, ci_high, title in zip(
        axes[1],
        [boot1, boot2],
        [obs1, obs2],
        [ci1_low, ci2_low],
        [ci1_high, ci2_high],
        ["Exp 1: ON edges (12 cond.)", "Exp 2: ON+OFF edges (24 cond.)"]
    ):
        ax.hist(boot, bins=50, color="seagreen", alpha=0.7,
                label="Bootstrap distribution")
        ax.axvline(obs, color="red", linewidth=2,
                   label=f"Observed CKA = {obs:.4f}")
        ax.axvline(ci_low,  color="gray", linewidth=1.5, linestyle="--",
                   label=f"95% CI [{ci_low:.4f}, {ci_high:.4f}]")
        ax.axvline(ci_high, color="gray", linewidth=1.5, linestyle="--")
        ax.set_xlabel("CKA (bootstrap)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\n95% CI [{ci_low:.4f}, {ci_high:.4f}]")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = "../figures/cka_validation_exp1_exp2.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fname}")
    plt.show()

    # ── 5. SUMMARY ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Exp 1 (ON edges):     CKA = {obs1:.4f} | p = {p1:.4f} "
          f"| 95% CI [{ci1_low:.4f}, {ci1_high:.4f}]")
    print(f"  Exp 2 (ON+OFF edges): CKA = {obs2:.4f} | p = {p2:.4f} "
          f"| 95% CI [{ci2_low:.4f}, {ci2_high:.4f}]")
    print()
    print("  Interpretation:")
    print("  CKA significantly greater than chance (p < 0.05, permutation test) → CC and random geometry differ")
    print("  Note: Exp 2 bootstrap distribution is bimodal — near-overflow activations in")
    print("  some stable random models drive CKA toward zero under resampling; permutation")
    print("  test is the primary inference")
    print("  Convergence with RSA result strengthens the fidelity claim")
    print()
    print("  RSA reference (from Experiments 1 & 2, n=50 canonical):")
    print("  Exp 1: Spearman r = 0.686, p < 0.0001")
    print("  Exp 2: Spearman r = 0.846, p < 0.0001")
    
    # ── 6. SAVE CKA RESULTS ──────────────────────────────────────────────────
    np.savez(
        "../results/cka_validation_50models_full_shiu.npz",
        cka_exp1=obs1,
        cka_exp2=obs2,
        p_exp1=p1,
        p_exp2=p2,
        ci1_low=ci1_low,
        ci1_high=ci1_high,
        ci2_low=ci2_low,
        ci2_high=ci2_high,
        null_exp1=null1,
        null_exp2=null2,
        boot_exp1=boot1,
        boot_exp2=boot2,
    )
    print("Saved: ../results/cka_validation_50models_full_shiu.npz")


if __name__ == "__main__":
    main()
