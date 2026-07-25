#!/usr/bin/env python
"""
Cheap follow-up to the OFF-OFF trained-Erdos-Renyi-indistinguishability finding
(Table 5): does Erdos-Renyi's OFF-OFF sub-block cross into non-significance
earlier, or more steeply, than degree-preserving swap's does, across the full
training trajectory? Uses only data already on disk (the full-checkpoint-sweep
.npz from evaluate_full_trajectory_all_conditions.py) -- no new training.

Reuses pairwise_r/within_group_r verbatim from
analyze_within_polarity_decomposition.py (the already-validated sub-block
extraction logic) and the checkpoint-loop/CC-loading structure from
analyze_full_trajectory.py, merged rather than reimplemented.

Usage:
    python analyze_offoff_trajectory.py \
        --trajectory_npz full_trajectory_moving_edge_12dir_on_off.npz \
        --cc_data results_exp2_50models_full_shiu.npz \
        --out_plot offoff_trajectory.png
"""
import argparse
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from scipy.spatial.distance import cosine


def build_rdm_from_matrix(pop_matrix):
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3) + 1e-10
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rdm[i, j] = cosine(pop_matrix[i], pop_matrix[j])
    return rdm


def pairwise_r(rdms_a, rdms_b, indices):
    """Verbatim from analyze_within_polarity_decomposition.py."""
    idx = np.triu_indices(len(indices), k=1)
    out = []
    for ra in rdms_a:
        sub_a = ra[np.ix_(indices, indices)]
        for rb in rdms_b:
            sub_b = rb[np.ix_(indices, indices)]
            r, _ = spearmanr(sub_a[idx], sub_b[idx])
            out.append(r)
    return np.array(out)


def within_group_r(rdms, indices):
    """Verbatim from analyze_within_polarity_decomposition.py."""
    idx = np.triu_indices(len(indices), k=1)
    out = []
    for i in range(len(rdms)):
        for j in range(i + 1, len(rdms)):
            sub_i = rdms[i][np.ix_(indices, indices)]
            sub_j = rdms[j][np.ix_(indices, indices)]
            r, _ = spearmanr(sub_i[idx], sub_j[idx])
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
    if n_stim != 24:
        raise ValueError(f"Expected 24 stimuli (ON+OFF), got {n_stim}.")
    # Same interleaving as plotting_utils.py: even=OFF, odd=ON
    on_idx = np.arange(1, 24, 2)
    off_idx = np.arange(0, 24, 2)

    results = {}
    for scheme in ["degree_preserving_swap", "erdos_renyi"]:
        net_keys = sorted(k for k in traj.files if k.startswith(f"{scheme}_") and k.endswith("_rdms"))
        if not net_keys:
            print(f"[!] No trajectory data found for {scheme} in {args.trajectory_npz} -- skipping")
            continue
        all_rdms = [traj[k] for k in net_keys]
        n_ckpts = min(r.shape[0] for r in all_rdms)
        print(f"=== {scheme} ({len(net_keys)} networks, {n_ckpts} checkpoints) ===")

        for label, indices in [("ON-ON", on_idx), ("OFF-OFF", off_idx)]:
            traj_r, traj_p = [], []
            n_nan_total = 0
            for position in range(n_ckpts):
                null_rdms_at_ckpt = [r[position] for r in all_rdms]
                within_cc = within_group_r(cc_rdms, indices)
                cc_vs_null = pairwise_r(cc_rdms, null_rdms_at_ckpt, indices)
                n_nan = np.isnan(cc_vs_null).sum()
                cc_vs_null_clean = cc_vs_null[~np.isnan(cc_vs_null)]
                if len(cc_vs_null_clean) < 2:
                    traj_r.append(np.nan)
                    traj_p.append(np.nan)
                    n_nan_total += n_nan
                    continue
                _, p = mannwhitneyu(within_cc, cc_vs_null_clean, alternative="greater")
                traj_r.append(cc_vs_null_clean.mean())
                traj_p.append(p)
                n_nan_total += n_nan
            traj_r, traj_p = np.array(traj_r), np.array(traj_p)
            if n_nan_total > 0:
                print(f"  [!] {label}: dropped {n_nan_total} NaN pair(s) across all checkpoints "
                      f"(likely a constant/non-responsive network at one or more checkpoints)")
            results[f"{scheme}_{label}"] = dict(r=traj_r, p=traj_p)

            # Report when (if ever) this sub-block first crosses into
            # non-significance and stays there through the end.
            nonsig = traj_p >= 0.05
            first_nonsig_run_start = None
            for i in range(n_ckpts):
                if nonsig[i:].all():
                    first_nonsig_run_start = i
                    break
            print(f"  {label}: final r={traj_r[-1]:.3f}, final p={traj_p[-1]:.2e}", end="")
            if first_nonsig_run_start is not None:
                print(f" -- crosses into non-significance at checkpoint {first_nonsig_run_start} "
                      f"and stays there through the end")
            else:
                print(" -- never sustains non-significance through to the final checkpoint")
        print()

    if args.out_plot and results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        styles = {
            "degree_preserving_swap_ON-ON": ("tab:blue", "-"),
            "degree_preserving_swap_OFF-OFF": ("tab:blue", "--"),
            "erdos_renyi_ON-ON": ("tab:orange", "-"),
            "erdos_renyi_OFF-OFF": ("tab:orange", "--"),
        }
        for key, d in results.items():
            color, ls = styles.get(key, ("gray", "-"))
            ax.plot(d["r"], label=key, color=color, linestyle=ls, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.4)
        ax.set_xlabel("Checkpoint position (training progress)")
        ax.set_ylabel("CC-vs-null mean r (individual-pairwise)")
        ax.set_title("ON-ON vs OFF-OFF trajectories, both schemes: does Erdos-Renyi's\nOFF-OFF gap close earlier or more steeply?")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(args.out_plot, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.out_plot}")


if __name__ == "__main__":
    main()
