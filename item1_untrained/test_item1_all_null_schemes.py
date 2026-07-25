#!/usr/bin/env python
"""
Extends item 1's original real-vs-random untrained comparison to the two
null schemes actually used in items 3-5 (degree_preserving_swap,
erdos_renyi), which item 1 itself never tested. Item 1's own random
baseline is a WEIGHT SHUFFLE of the real, trained topology (randomize_weights
in moving_edge_on_off.ipynb) -- a fundamentally different kind of null from
degree-preserving swap or Erdos-Renyi, which randomize the CONNECTOME
TOPOLOGY itself via external connectome files. Item 1's pipeline has no
mechanism to load an arbitrary connectome file, so this script is new
where needed, verbatim-reused where possible.

KEY INSIGHT, no new training required: every training run's own log shows
"Initialized network with NumberOfParams..." BEFORE any training iterations
happen. Checkpoint position 0, for any already-trained exp5 seed network, IS
the untrained, freshly-initialized state on that connectome file. The
original n=10 degree_preserving_swap and erdos_renyi populations from item 3
already exist on disk (used throughout items 3-5 for training); this script
pulls their checkpoint 0 specifically, evaluates each with item 1's own
population-vector/RDM pipeline, and compares against the real CC ensemble
the same way item 1 already does.

Every function below through permutation_test_rdm is copied VERBATIM from
moving_edge_on_off.ipynb's cell 4, unmodified, for exact methodological
consistency with the r=0.686/0.846 numbers already reported in item 1.
Only the network-loading step (load_null_network) and the top-level
orchestration are new.

Usage:
    python test_item1_all_null_schemes.py --n_models 10 --n_permutations 10000
"""
import argparse
import json

import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import spearmanr, kendalltau, mannwhitneyu

import flyvis
from flyvis import results_dir, EnsembleView
from flyvis.network import NetworkView
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.utils.activity_utils import LayerActivity
from flyvis.utils.chkpt_utils import checkpoint_index_to_path_map
from plotting_utils import plot_comparison

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

STIMULUS_SETS = {
    "moving_edge_12dir": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    "henning_8dir": [0, 45, 90, 135, 180, 225, 270, 315],
}


def build_dataset(stimulus_set):
    """stimulus_set: 'moving_edge_12dir' (this project's own 30-spaced set,
    supports --polarity on/on_off) or 'henning_8dir' (the Henning-matched,
    45-spaced, ON-only-only set confirmed from results_exp1_8dir_50models_full_shiu.npz's
    own 'angles' key -- NOT a subset of the 12dir set, only 4 of 8 directions
    overlap, genuinely different stimuli, not a filter on the same dataset)."""
    angles = STIMULUS_SETS[stimulus_set]
    return MovingEdge(
        offsets=[-10, 11], intensities=[0, 1], speeds=[19], height=80,
        post_pad_mode="continue", t_pre=1.0, t_post=1.0, dt=1/200, angles=angles,
    ), angles
# Dataset always includes both polarities; which subset gets USED as stimuli
# is filtered per --polarity below, matching how moving_edge_on.ipynb (ON-only)
# and moving_edge_on_off.ipynb (both) differ -- confirmed by direct diff of
# both notebooks' cell 4, the only functional difference between them.
# henning_8dir has no on_off variant on file -- it's ON-only, matching the
# one real-CC file that exists for it (results_exp1_8dir_50models_full_shiu.npz,
# shape (50, 8, 65), 8 stimuli = 8 directions x 1 polarity).


# --- Verbatim from moving_edge_on_off.ipynb, unmodified ---

def get_population_vector(network_view, stimulus, dt, use_fade_in=True):
    network = network_view.init_network()
    if stimulus.dim() == 2:
        stimulus = stimulus.unsqueeze(1)
    if use_fade_in:
        initial_state = network.fade_in_state(1.0, dt, stimulus[[0]])
    else:
        initial_state = None
    with torch.no_grad():
        responses = network.simulate(stimulus[None], dt, initial_state=initial_state).cpu()
    layer_act = LayerActivity(responses, network.connectome, keepref=True)
    cell_types = [ct.decode() if isinstance(ct, bytes) else ct
                  for ct in network.connectome.unique_cell_types[:]]
    pop_vec = np.array([layer_act.central[ct].squeeze().numpy().max() for ct in cell_types])
    pop_vec = np.clip(pop_vec, -1e6, 1e6)
    del network, responses, layer_act
    torch.cuda.empty_cache()
    return pop_vec, cell_types


def build_rdm(pop_matrix, metric="cosine"):
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3)
    if metric == "cosine":
        norms = np.linalg.norm(pop_matrix, axis=1, keepdims=True)
        zero_norm_rows = (norms < 1e-10).flatten()
        if np.any(zero_norm_rows):
            print(f"    WARNING: {zero_norm_rows.sum()} zero-norm population vectors -- adding epsilon")
        pop_matrix = pop_matrix + 1e-10
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rdm[i, j] = cosine(pop_matrix[i], pop_matrix[j]) if metric == "cosine" else euclidean(pop_matrix[i], pop_matrix[j])
    return rdm


def rdm_similarity(rdm1, rdm2):
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    r_s, p_s = spearmanr(rdm1[idx], rdm2[idx])
    r_k, p_k = kendalltau(rdm1[idx], rdm2[idx])
    return r_s, p_s, r_k, p_k


def permutation_test_rdm(rdm1, rdm2, n_permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    obs_r, _ = spearmanr(rdm1[idx], rdm2[idx])
    obs_tau, _ = kendalltau(rdm1[idx], rdm2[idx])
    null_r = np.zeros(n_permutations)
    null_tau = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        rdm2_perm = rdm2[np.ix_(perm, perm)]
        null_r[i], _ = spearmanr(rdm1[idx], rdm2_perm[idx])
        null_tau[i], _ = kendalltau(rdm1[idx], rdm2_perm[idx])
    p_r = np.mean(null_r >= obs_r)
    p_tau = np.mean(null_tau >= obs_tau)
    return obs_r, p_r, obs_tau, p_tau, null_r, null_tau


# --- New: load checkpoint 0 (pre-training, freshly initialized) for an
# already-trained exp5 null-scheme network -- no new training required ---

def load_null_network(scheme, net_idx, checkpoint="first", results_root=None):
    """scheme: 'degree_preserving_swap' or 'erdos_renyi'. checkpoint: 'first'
    (position 0, before any training -- what item 1's untrained comparison
    used) or 'last' (final checkpoint, fully trained -- what item 3 trained
    on). Same NetworkDir/checkpoint_index_to_path_map machinery already
    verified and reused throughout this project's other checkpoint-loading
    scripts, just parameterized on which position to select."""
    if results_root is None:
        results_root = str(flyvis.results_dir)
    network_name = f"exp5_{scheme}/{net_idx:04d}"
    path = f"{results_root}/{network_name}"

    def checkpoint_fn(p, **kwargs):
        networkdir = flyvis.NetworkDir(p)
        checkpoint_dir = networkdir.chkpts.path
        indices, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
        if not paths:
            raise FileNotFoundError(f"no checkpoints found for {p}")
        return paths[0] if checkpoint == "first" else paths[-1]

    return NetworkView(path, best_checkpoint_fn=checkpoint_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=10)
    ap.add_argument("--n_permutations", type=int, default=10000)
    ap.add_argument("--results_root", default=None,
                     help="Override flyvis.results_dir for the null-scheme networks, "
                          "e.g. /workspace/flow_v3")
    ap.add_argument("--stimulus_set", choices=["moving_edge_12dir", "henning_8dir"],
                     default="moving_edge_12dir",
                     help="'moving_edge_12dir' is this project's own 30-spaced set "
                          "(supports --polarity on/on_off). 'henning_8dir' is the "
                          "Henning-matched, 45-spaced set confirmed from "
                          "results_exp1_8dir_50models_full_shiu.npz's own 'angles' key -- "
                          "NOT a subset of the 12dir set, genuinely different stimuli, "
                          "ON-only only (no on_off variant exists for it).")
    ap.add_argument("--cc_data", default=None,
                     help="Pre-computed real CC ensemble population matrices. Defaults "
                          "based on --stimulus_set and --polarity: "
                          "results_exp1_50models_full_shiu.npz (moving_edge_12dir, on), "
                          "results_exp2_50models_full_shiu.npz (moving_edge_12dir, on_off), "
                          "results_exp1_8dir_50models_full_shiu.npz (henning_8dir).")
    ap.add_argument("--polarity", choices=["on", "on_off"], default="on_off",
                     help="Ignored for --stimulus_set henning_8dir, which is ON-only only, "
                          "matching the one real-CC file that exists for it.")
    ap.add_argument("--checkpoint", choices=["first", "last"], default="first",
                     help="'first' = checkpoint 0, before any training (item 1's original "
                          "untrained comparison). 'last' = final, fully-trained checkpoint "
                          "(directly tests whether training makes real and random wiring's "
                          "geometry converge, using item 1's own individual-pairwise "
                          "methodology rather than item 3's biology-mediated one). Real CC "
                          "data (from --cc_data) is already fully trained either way, "
                          "Flyvis's pretrained ensemble -- only affects the null-scheme side.")
    ap.add_argument("--out", default=None,
                     help="Defaults to item1_all_null_schemes_results_{stimulus_set}_"
                          "{polarity}_{checkpoint}.json so different runs don't overwrite "
                          "each other.")
    args = ap.parse_args()

    if args.stimulus_set == "henning_8dir" and args.polarity != "on":
        print(f"[!] --stimulus_set henning_8dir is ON-only only -- ignoring "
              f"--polarity={args.polarity}, forcing 'on'")
        args.polarity = "on"

    dataset, angles = build_dataset(args.stimulus_set)
    intensities = [1] if args.polarity == "on" else [0, 1]
    print(f"Stimulus set: {args.stimulus_set}, polarity: {args.polarity}")

    results_root = args.results_root if args.results_root else str(flyvis.results_dir)
    print(f"Using results_root (null schemes only): {results_root}")

    stim_indices = [i for i, row in dataset.arg_df.iterrows() if row["intensity"] in intensities]
    n_stim = len(stim_indices)
    print(f"Stimulus conditions: {n_stim}")

    cc_data_path = args.cc_data
    if cc_data_path is None:
        if args.stimulus_set == "henning_8dir":
            cc_data_path = "results_exp1_8dir_50models_full_shiu.npz"
        else:
            cc_data_path = ("results_exp1_50models_full_shiu.npz" if args.polarity == "on"
                             else "results_exp2_50models_full_shiu.npz")
    print(f"\n=== Loading real CC ensemble from {cc_data_path} (already computed, not re-derived) ===")
    cc_data = np.load(cc_data_path, allow_pickle=True)
    cc_pop_matrices = list(cc_data["cc_pop_matrices"])[:args.n_models]
    print(f"  Loaded {len(cc_pop_matrices)} real CC population matrices, "
          f"shape {cc_pop_matrices[0].shape} each")
    assert cc_pop_matrices[0].shape[0] == n_stim, (
        f"Stimulus count mismatch: cc_data has {cc_pop_matrices[0].shape[0]} conditions, "
        f"this script's dataset ({args.stimulus_set}, --polarity={args.polarity}) has "
        f"{n_stim} -- these must match before the comparison means anything.")

    cc_rdms_individual = [build_rdm(m, "cosine") for m in cc_pop_matrices]
    cc_rdm_cosine = np.mean(cc_rdms_individual, axis=0)  # kept for mean-vs-mean, for
    # direct comparability against item 1's originally reported r=0.686/0.846

    idx = np.triu_indices(n_stim, k=1)
    within_cc = []
    for i in range(len(cc_rdms_individual)):
        for j in range(i + 1, len(cc_rdms_individual)):
            r_ij, _ = spearmanr(cc_rdms_individual[i][idx], cc_rdms_individual[j][idx])
            within_cc.append(r_ij)
    within_cc = np.array(within_cc)
    print(f"  Within-CC baseline: n={len(within_cc)} pairs, mean r = {within_cc.mean():.3f} "
          f"+/- {within_cc.std():.3f}")

    checkpoint_label = "untrained" if args.checkpoint == "first" else "trained"
    results = {}
    raw_plot_data = {}  # saved to a companion .npz so plots can be regenerated
    # later without rerunning any network evaluation -- see --out's .npz sibling
    for scheme in ["degree_preserving_swap", "erdos_renyi"]:
        print(f"\n=== {scheme}, {args.checkpoint} checkpoint ({checkpoint_label}) ===")
        null_pop_matrices = []
        for net_idx in range(args.n_models):
            try:
                nv = load_null_network(scheme, net_idx, checkpoint=args.checkpoint,
                                        results_root=results_root)
            except FileNotFoundError as e:
                print(f"  [!] {scheme}/{net_idx:04d}: {e} -- skipping")
                continue
            pop_vecs = []
            for stim_idx in stim_indices:
                stimulus = dataset[stim_idx]
                if not isinstance(stimulus, torch.Tensor):
                    stimulus = torch.tensor(stimulus, dtype=torch.float32)
                pop_vec, _ = get_population_vector(nv, stimulus, dataset.dt)
                pop_vecs.append(pop_vec)
            null_pop_matrices.append(np.stack(pop_vecs, axis=0))
            print(f"  {scheme}/{net_idx:04d} ({args.checkpoint} checkpoint) done")
            del nv
            torch.cuda.empty_cache()

        if not null_pop_matrices:
            print(f"  No usable networks for {scheme} -- skipping comparison")
            continue

        null_rdms_individual = [build_rdm(m, "cosine") for m in null_pop_matrices]
        null_rdm_cosine = np.mean(null_rdms_individual, axis=0)

        # Old (weaker) mean-vs-mean statistic, kept for direct comparability
        r, p, tau, p_tau = rdm_similarity(cc_rdm_cosine, null_rdm_cosine)
        obs_r, p_r_perm, obs_tau, p_tau_perm, null_r_dist, _ = permutation_test_rdm(
            cc_rdm_cosine, null_rdm_cosine, n_permutations=args.n_permutations, seed=SEED)
        print(f"  [mean-vs-mean, weaker statistic] Spearman r = {r:.3f}, p = {p:.4f} [analytical]")
        print(f"  [mean-vs-mean, weaker statistic] Permutation p = {p_r_perm:.4f}")

        # New, proper full individual-pairwise comparison
        cc_vs_null = []
        for cc_rdm in cc_rdms_individual:
            for null_rdm in null_rdms_individual:
                r_ij, _ = spearmanr(cc_rdm[idx], null_rdm[idx])
                cc_vs_null.append(r_ij)
        cc_vs_null = np.array(cc_vs_null)
        u_stat, mw_p = mannwhitneyu(within_cc, cc_vs_null, alternative="greater")
        print(f"  [individual-pairwise, proper statistic] n={len(cc_vs_null)} pairs, "
              f"mean r = {cc_vs_null.mean():.3f} +/- {cc_vs_null.std():.3f}")
        print(f"  [individual-pairwise, proper statistic] Mann-Whitney "
              f"(within-CC > CC-vs-{scheme}): p = {mw_p:.2e}")
        if mw_p < 0.001:
            print(f"  -> CC is significantly MORE similar to itself than to {scheme}: "
                  f"real wiring IS distinguishable ({checkpoint_label}).")
        else:
            print(f"  -> No significant gap between within-CC and CC-vs-{scheme}: "
                  f"real wiring is NOT clearly distinguishable ({checkpoint_label}).")

        plot_comparison(cc_rdm_cosine, null_rdm_cosine, scheme, args.polarity, angles,
                         null_r_dist, obs_r, out_prefix=f"item1_null_comparison_{args.stimulus_set}",
                         checkpoint_label=checkpoint_label)

        results[scheme] = dict(
            n_models=len(null_pop_matrices),
            mean_vs_mean=dict(r=float(r), p=float(p), perm_p=float(p_r_perm),
                               tau=float(tau), perm_tau_p=float(p_tau_perm)),
            individual_pairwise=dict(n_pairs=len(cc_vs_null), mean_r=float(cc_vs_null.mean()),
                                      std_r=float(cc_vs_null.std()), mannwhitney_p=float(mw_p)),
            within_cc_baseline=dict(n_pairs=len(within_cc), mean_r=float(within_cc.mean()),
                                     std_r=float(within_cc.std())),
        )
        # Raw data needed to REGENERATE plots later without rerunning any
        # network evaluation -- the JSON above only has summary statistics.
        raw_plot_data[f"{scheme}_cc_rdm"] = cc_rdm_cosine
        raw_plot_data[f"{scheme}_null_rdm"] = null_rdm_cosine
        raw_plot_data[f"{scheme}_null_r_dist"] = null_r_dist
        raw_plot_data[f"{scheme}_obs_r"] = obs_r
        # Individual (not just mean) RDMs -- needed for any proper individual-
        # pairwise sub-block analysis (e.g. within-polarity decomposition) done
        # later without rerunning network evaluation. cc_rdms_individual is the
        # same across all schemes (same real ensemble), saved once under a
        # scheme-independent key to avoid redundant storage.
        raw_plot_data[f"{scheme}_null_rdms_individual"] = np.array(null_rdms_individual)
        raw_plot_data["cc_rdms_individual"] = np.array(cc_rdms_individual)

    out_path = args.out if args.out else f"item1_all_null_schemes_results_{args.stimulus_set}_{args.polarity}_{args.checkpoint}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    npz_path = out_path.replace(".json", "_raw_for_replotting.npz")
    np.savez(npz_path, angles=np.array(angles), polarity=args.polarity,
              checkpoint_label=checkpoint_label, **raw_plot_data)
    print(f"wrote {npz_path} (raw RDMs + null distributions, for replotting "
          f"without rerunning any network evaluation)")
    print("\nSee the per-scheme Mann-Whitney result above for the actual interpretation --")
    print("that individual-pairwise comparison against the within-CC baseline is the")
    print("correct statistic; the raw mean-vs-mean r alone is not reliable on its own.")


if __name__ == "__main__":
    main()
