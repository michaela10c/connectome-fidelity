#!/usr/bin/env python
"""
Full-checkpoint-sweep version of test_item1_all_null_schemes.py's
first/last comparison. Evaluates each null-scheme network at EVERY
checkpoint (not just checkpoint 0 and the final one), for whichever
stimulus condition is specified, so the trajectory of convergence can be
examined later rather than just its two endpoints -- the follow-up flagged
in item 3 ("whether this convergence is gradual, abrupt, or non-monotonic
remains untested").

Real CC data needs no per-checkpoint evaluation (it's a fixed, pretrained
ensemble with no trajectory of its own) -- only the null-scheme networks
are swept. All core functions (get_population_vector, build_rdm,
STIMULUS_SETS, build_dataset) are imported directly from
test_item1_all_null_schemes.py, not reimplemented, for exact methodological
consistency with the already-validated first/last comparison.

Saves raw per-checkpoint RDMs only -- no analysis is done here. Statistics
(individual-pairwise Mann-Whitney at each checkpoint, etc.) belong in a
separate analysis script, run afterward on the saved .npz, matching the
evaluate/analyze split already used for the K=8 divergence-timing work.

Usage:
    python evaluate_full_trajectory_all_conditions.py \
        --stimulus_set moving_edge_12dir --polarity on_off \
        --results_root /workspace/flow_v3 --n_models 10
"""
import argparse
import numpy as np
import torch

import flyvis
from flyvis.network import NetworkView
from flyvis.utils.chkpt_utils import checkpoint_index_to_path_map

from test_item1_all_null_schemes import (
    STIMULUS_SETS, build_dataset, get_population_vector, build_rdm,
)


def load_all_checkpoints(scheme, net_idx, results_root):
    """Returns a list of NetworkView objects, one per checkpoint position,
    in order. Same NetworkDir/checkpoint_index_to_path_map machinery as
    load_null_network in the main script, generalized to every position
    instead of just first/last."""
    network_name = f"exp5_{scheme}/{net_idx:04d}"
    path = f"{results_root}/{network_name}"
    networkdir = flyvis.NetworkDir(path)
    checkpoint_dir = networkdir.chkpts.path
    indices, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
    if not paths:
        raise FileNotFoundError(f"no checkpoints found for {path}")

    views = []
    for position in range(len(paths)):
        def checkpoint_fn(p, _paths=paths, _pos=position, **kwargs):
            return _paths[_pos]
        views.append(NetworkView(path, best_checkpoint_fn=checkpoint_fn))
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus_set", choices=list(STIMULUS_SETS.keys()),
                     default="moving_edge_12dir")
    ap.add_argument("--polarity", choices=["on", "on_off"], default="on_off",
                     help="Ignored for henning_8dir (ON-only only).")
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--n_models", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.stimulus_set == "henning_8dir" and args.polarity != "on":
        print(f"[!] henning_8dir is ON-only only -- ignoring --polarity={args.polarity}")
        args.polarity = "on"

    dataset, angles = build_dataset(args.stimulus_set)
    intensities = [1] if args.polarity == "on" else [0, 1]
    stim_indices = [i for i, row in dataset.arg_df.iterrows() if row["intensity"] in intensities]
    n_stim = len(stim_indices)
    print(f"Stimulus set: {args.stimulus_set}, polarity: {args.polarity}, {n_stim} conditions")

    out_path = args.out or f"full_trajectory_{args.stimulus_set}_{args.polarity}.npz"
    raw_data = {"angles": np.array(angles), "polarity": args.polarity}

    for scheme in ["degree_preserving_swap", "erdos_renyi"]:
        for net_idx in range(args.n_models):
            print(f"\n=== {scheme}/{net_idx:04d}, full checkpoint sweep ===")
            try:
                views = load_all_checkpoints(scheme, net_idx, args.results_root)
            except FileNotFoundError as e:
                print(f"  [!] {e} -- skipping")
                continue

            n_ckpts = len(views)
            rdms = np.zeros((n_ckpts, n_stim, n_stim))
            for position, nv in enumerate(views):
                pop_vecs = []
                for stim_idx in stim_indices:
                    stimulus = dataset[stim_idx]
                    if not isinstance(stimulus, torch.Tensor):
                        stimulus = torch.tensor(stimulus, dtype=torch.float32)
                    pop_vec, _ = get_population_vector(nv, stimulus, dataset.dt)
                    pop_vecs.append(pop_vec)
                pop_matrix = np.stack(pop_vecs, axis=0)
                rdms[position] = build_rdm(pop_matrix, "cosine")
                if (position + 1) % 10 == 0 or position == n_ckpts - 1:
                    print(f"  checkpoint {position + 1}/{n_ckpts} done")
                torch.cuda.empty_cache()

            raw_data[f"{scheme}_{net_idx:04d}_rdms"] = rdms
            # Save incrementally after each network, so a crash partway
            # through doesn't lose everything already computed.
            np.savez(out_path, **raw_data)
            print(f"  Saved progress to {out_path} ({n_ckpts} checkpoints, "
                  f"shape {rdms.shape})")

    print(f"\nDone. Final file: {out_path}")
    print("Contains raw per-checkpoint RDMs only -- run a separate analysis")
    print("script against this file to compute trajectory statistics.")


if __name__ == "__main__":
    main()
