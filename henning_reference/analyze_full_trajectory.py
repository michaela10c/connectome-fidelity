#!/usr/bin/env python
"""
Computes the individual-pairwise CC-vs-null Mann-Whitney trajectory across
every checkpoint, from the raw RDMs saved by
evaluate_full_trajectory_all_conditions.py. Answers the question flagged in
item 3: is the untrained-to-trained convergence gradual, abrupt at some
specific point, or non-monotonic?

Real CC RDMs are loaded from the same results_exp*_50models_full_shiu.npz
files used throughout item 1, not re-derived -- consistent with every other
script in this pipeline.

Usage:
    python analyze_full_trajectory.py \
        --trajectory_npz full_trajectory_moving_edge_12dir_on_off.npz \
        --cc_data results_exp2_50models_full_shiu.npz \
        --out_plot trajectory_convergence.png
"""
import argparse
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu


def build_rdm_from_matrix(pop_matrix):
    from scipy.spatial.distance import cosine
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3) + 1e-10
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rdm[i, j] = cosine(pop_matrix[i], pop_matrix[j])
    return rdm


def pairwise_r(rdms_a, rdms_b, idx):
    out = []
    for ra in rdms_a:
        for rb in rdms_b:
            r, _ = spearmanr(ra[idx], rb[idx])
            out.append(r)
    return np.array(out)


def within_group_r(rdms, idx):
    out = []
    for i in range(len(rdms)):
        for j in range(i + 1, len(rdms)):
            r, _ = spearmanr(rdms[i][idx], rdms[j][idx])
            out.append(r)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory_npz", required=True)
    ap.add_argument("--cc_data", required=True)
    ap.add_argument("--n_models", type=int, default=10)
    ap.add_argument("--out_plot", default=None)
    args = ap.parse_args()

    traj = np.load(args.trajectory_npz, allow_pickle=True)
    cc_data = np.load(args.cc_data, allow_pickle=True)
    cc_pop_matrices = list(cc_data["cc_pop_matrices"])[:args.n_models]
    cc_rdms = [build_rdm_from_matrix(m) for m in cc_pop_matrices]
    n_stim = cc_rdms[0].shape[0]
    idx = np.triu_indices(n_stim, k=1)
    within_cc = within_group_r(cc_rdms, idx)
    print(f"Within-CC baseline: {within_cc.mean():.3f} +/- {within_cc.std():.3f}\n")

    schemes = sorted(set(k.split("_0")[0] for k in traj.files if k.endswith("_rdms")
                          and not k.startswith("angles")))
    results = {}
    for scheme in ["degree_preserving_swap", "erdos_renyi"]:
        net_keys = sorted(k for k in traj.files if k.startswith(f"{scheme}_") and k.endswith("_rdms"))
        if not net_keys:
            continue
        print(f"=== {scheme} ({len(net_keys)} networks) ===")
        all_rdms = [traj[k] for k in net_keys]  # each: (n_ckpts, n_stim, n_stim)
        n_ckpts = min(r.shape[0] for r in all_rdms)  # in case of uneven counts

        trajectory_r = []
        trajectory_p = []
        for position in range(n_ckpts):
            null_rdms_at_ckpt = [r[position] for r in all_rdms]
            cc_vs_null = pairwise_r(cc_rdms, null_rdms_at_ckpt, idx)
            n_nan = np.isnan(cc_vs_null).sum()
            cc_vs_null_clean = cc_vs_null[~np.isnan(cc_vs_null)]
            if len(cc_vs_null_clean) < 2:
                trajectory_r.append(np.nan)
                trajectory_p.append(np.nan)
                if n_nan > 0:
                    print(f"    [!] checkpoint {position}: {n_nan} NaN pairs, too few valid pairs to test")
                continue
            _, p = mannwhitneyu(within_cc, cc_vs_null_clean, alternative="greater")
            trajectory_r.append(cc_vs_null_clean.mean())
            trajectory_p.append(p)
            if n_nan > 0:
                print(f"    [!] checkpoint {position}: dropped {n_nan} NaN pair(s) "
                      f"(likely a constant/non-responsive network), {len(cc_vs_null_clean)} used")
            if position % 10 == 0 or position == n_ckpts - 1:
                print(f"  checkpoint {position}: mean r = {cc_vs_null.mean():.3f}, "
                      f"Mann-Whitney p = {p:.2e}")

        results[scheme] = dict(r=np.array(trajectory_r), p=np.array(trajectory_p))
        print()

    if args.out_plot and results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for scheme, d in results.items():
            ax.plot(d["r"], label=scheme, linewidth=1.5)
        ax.axhline(within_cc.mean(), color="gray", linestyle="--", alpha=0.6,
                   label="within-CC baseline")
        ax.set_xlabel("Checkpoint position (training progress)")
        ax.set_ylabel("Individual-pairwise CC-vs-null mean r")
        print(args.out_plot)
        if args.out_plot.endswith('henning.png'):
            stim_label = "8-Direction Henning" 
        elif args.out_plot.endswith('onoff.png'):
            stim_label = "ON-OFF Polarity"
        else:
            stim_label = "ON Polarity"
        ax.set_title(f"Convergence trajectory: does the gap close gradually or abruptly? ({stim_label})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(args.out_plot, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.out_plot}")


if __name__ == "__main__":
    main()
