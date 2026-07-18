#!/usr/bin/env python
"""
training_trajectory_henning.py -- tests the training-vs-wiring hypothesis with a
within-subject dose-response design instead of a between-group comparison.

WHY THIS IS THE RIGHT NEXT TEST
---------------------------------
The pooled N=20/N=70 trained-vs-untrained comparison (pool_trained_vs_untrained.py)
is well-powered but still correlational: it rests on exactly one TYPE of untrained
comparator (weight-shuffled random). A second, genuinely independent untrained
population would strengthen this, but Experiment 4's untrained-at-initialization
networks are almost certainly still unresolvable at any stimulus count -- that
failure was about response MAGNITUDE near initialization (RDM dynamic range below
the float32 round-off floor), not about how many directions were tested, so
re-evaluating them at 8 directions would not fix it.

The better test uses data already on disk: each Exp5 network saved dozens of
checkpoints across its own 250,000-iteration training run (net 0000 had 72). If we
evaluate the SAME network at multiple points along its own trajectory -- early,
middle, late -- and its correlation with the Henning reference becomes
progressively more negative as training proceeds, that is a within-subject
dose-response relationship, not a between-group comparison. It is immune to the
single-comparator-type concern entirely: the "untrained" and "trained" conditions
are the same network's own weights at different points in the same training run.

WHAT THIS REUSES, AND WHY (do not reimplement any of these differently)
--------------------------------------------------------------------------
  - build_rdm_cosine, rank_residualize, circular_reference, N_DIRECTIONS, ANGLES,
    get_population_vector, get_cell_types: imported directly from
    exp5_henning_evaluate.py, so every checkpoint's RDM is built through the exact
    same numerically-locked pipeline validated throughout this project's
    Henning-reference work.
  - The checkpoint-loading workaround (avoiding Flyvis's confirmed
    best_checkpoint_default_fn bug) is the same approach as
    exp5_henning_evaluate.py's load_exp5_network, generalized here to load a
    SPECIFIC checkpoint index rather than always the last one.
  - True training iteration per checkpoint index is read from chkpt_iter.h5, the
    same file production.py's latest_checkpoint_iter reads -- checkpoint FILENAME
    indices are sequential positions, not iteration counts, and must not be
    conflated (this distinction is documented directly in production.py).

USAGE:
    python training_trajectory_henning.py \
        --schemes degree_preserving_swap,erdos_renyi \
        --net_indices 0,1 \
        --n_checkpoints 6

This evaluates 2 networks per scheme (4 total) at ~6 evenly-spaced checkpoints
each -- 24 network-checkpoint evaluations, 8 forward passes each (192 total),
comparable in cost to the original 160-forward-pass Exp5 re-evaluation. Increase
--net_indices to cover more networks per scheme if the trend from this first pass
is worth firming up with more statistical power.

Requires (same directory): exp5_henning_evaluate.py (imported),
henning_population_matrix.npy, raw_population_matrix.npy, and the trained Exp5
checkpoint trees under flyvis.results_dir/flow/exp5_<scheme>/<net_idx>/chkpts/.
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from exp5_henning_evaluate import (
    build_rdm_cosine,
    rank_residualize,
    circular_reference,
    build_dataset,
    get_population_vector,
    get_cell_types,
    INTENSITY,
)

# --- Precision guard, reused verbatim from Experiment 4b (exp4_perturbation_sweep.py /
# exp4_synapse_sweep.py), NOT reimplemented from description. This project already
# established -- expensively, via Experiment 4's original retraction -- that
# untrained/near-initialization network RDMs can fall below the float32 round-off
# floor, making any rank-correlation statistic computed from them meaningless
# (or, as seen directly in this script's first n=16-network run, literally NaN).
# The pre-training checkpoint here is exactly the initialization regime Experiment 4
# already showed is at risk -- these Exp5 null-wiring networks share the same Flyvis
# prior initialization scheme, just with randomized connectivity, so there is no
# reason to expect them to be exempt. This guard is applied to EVERY checkpoint
# evaluated, not just the pre-training one: some early-training checkpoints may sit
# close enough to the floor to be unreliable without producing an outright NaN.
FLOAT32_EPS = float(np.finfo(np.float32).eps)   # 1.19e-07, exact match to exp4b's constant
RESOLVE_MARGIN = 10.0                            # exact match to exp4b's constant


def upper(M):
    return M[np.triu_indices_from(M, k=1)]


def resolvability(rdm, pop_matrix):
    """Return (span, floor, ratio, is_resolvable). The gate -- identical logic
    to exp4_perturbation_sweep.py's resolvability(), reused rather than
    reimplemented so the threshold is guaranteed consistent with the rest of
    this project's precision-guard history."""
    off = upper(rdm)
    span = float(off.max() - off.min())
    floor = FLOAT32_EPS * float(np.abs(pop_matrix).max())
    ratio = span / floor if floor > 0 else np.inf
    return span, floor, ratio, ratio >= RESOLVE_MARGIN


def get_checkpoint_count_and_iteration_anchors(scheme, net_idx, results_root=None):
    """Ground-truth checkpoint COUNT comes from the real file list
    (checkpoint_index_to_path_map), never from chkpt_iter.h5's length --
    chkpt_iter.h5's length does not reliably equal the real checkpoint count
    and must never be used to select a checkpoint position. CORRECTED: an
    earlier version of this docstring described a "confirmed 2:1 ratio"
    between chkpt_iter.h5 length and real file count as if it were a general
    property of this experiment -- verify_checkpoint_spacing.py has since
    checked this across all 20 network/scheme combinations in use (both
    schemes, net_indices 0-9) and found ratio=1.00 for 19 of 20; the one
    exception (degree_preserving_swap/0001, ratio=2.00) is an outlier, not
    the norm. The underlying point stands regardless -- chkpt_iter.h5's
    length is not a reliable proxy for checkpoint count under either ratio,
    which is why the real file list is used -- but the "2:1 pattern" framing
    itself should not be repeated or relied on elsewhere.

    Returns (n_checkpoints, first_true_iteration, last_true_iteration). The
    first and last entries of chkpt_iter.h5 are trustworthy regardless of the
    internal indexing ambiguity: the first checkpoint is unambiguously
    pre-training (iteration -1) and the last logged iteration unambiguously
    corresponds to the last real checkpoint, since training and logging both
    stop together. Iterations for checkpoints in between are NOT read from
    chkpt_iter.h5 directly (that mapping is not established); see
    approx_iteration_for_position() for how they're estimated instead.
    """
    import h5py
    import flyvis
    from flyvis.utils.chkpt_utils import checkpoint_index_to_path_map
    if results_root is None:
        results_root = str(flyvis.results_dir)
    network_name = f"exp5_{scheme}/{net_idx:04d}"
    net_dir = os.path.join(results_root, "flow", network_name)

    networkdir = flyvis.NetworkDir(net_dir)
    checkpoint_dir = networkdir.chkpts.path
    _, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
    n_checkpoints = len(paths)

    iter_h5 = os.path.join(net_dir, "chkpt_iter.h5")
    with h5py.File(iter_h5, "r") as f:
        data = [int(x) for x in list(f["data"][()])]
    first_true_iter = data[0]     # trustworthy: pre-training, always -1
    last_true_iter = data[-1]     # trustworthy: training and logging stop together

    return n_checkpoints, first_true_iter, last_true_iter


def approx_iteration_for_position(position, n_checkpoints, first_true_iter, last_true_iter):
    """Linear interpolation between the two trusted anchors, scaled by
    checkpoint ORDINAL position (not by any chkpt_iter.h5 index). This is an
    ESTIMATE, not ground truth -- defensible because checkpoints are saved at
    a roughly fixed cadence (confirmed directly: the checkpoint files' own
    mtimes are ~33-36 minutes apart, consistent with fixed-interval
    checkpointing), but should be reported and used as an approximation, not
    treated as an exact iteration count."""
    if position == 0:
        return first_true_iter
    frac = position / (n_checkpoints - 1)
    return int(round(first_true_iter + (last_true_iter - first_true_iter) * frac))


def load_network_at_checkpoint(scheme, net_idx, checkpoint_position, results_root=None):
    """Loads a specific checkpoint by its POSITION in the sorted checkpoint list
    (0-indexed), not by iteration count. Reuses the same best_checkpoint_fn
    override pattern as exp5_henning_evaluate.py's load_exp5_network -- Flyvis's
    default best-checkpoint selection is confirmed buggy and must not be used."""
    from flyvis.network import NetworkView
    from flyvis.utils.chkpt_utils import checkpoint_index_to_path_map
    import flyvis
    if results_root is None:
        results_root = str(flyvis.results_dir)

    network_name = f"exp5_{scheme}/{net_idx:04d}"
    path = os.path.join(results_root, "flow", network_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"no trained network at {path}")

    def specific_checkpoint_fn(p, **kwargs):
        networkdir = flyvis.NetworkDir(p)
        checkpoint_dir = networkdir.chkpts.path
        indices, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
        if not paths:
            raise FileNotFoundError(f"no checkpoints found for {p}")
        if checkpoint_position >= len(paths):
            raise IndexError(f"checkpoint position {checkpoint_position} out of range "
                              f"(only {len(paths)} checkpoints exist for {p})")
        return paths[checkpoint_position]

    nv = NetworkView(path, best_checkpoint_fn=specific_checkpoint_fn)
    return nv.init_network()


def evaluate_checkpoint(net, dataset, on_edge_indices, dt, cell_types, device):
    pop_vecs = []
    for stim_idx in on_edge_indices:
        stimulus = dataset[stim_idx]
        if not isinstance(stimulus, torch.Tensor):
            stimulus = torch.tensor(stimulus, dtype=torch.float32)
        stimulus = stimulus.to(device)
        pop_vecs.append(get_population_vector(net, stimulus, dt, cell_types))
    pop_matrix = np.stack(pop_vecs, axis=0)
    return build_rdm_cosine(pop_matrix), pop_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="degree_preserving_swap,erdos_renyi")
    ap.add_argument("--net_indices", default="0,1",
                     help="comma-separated net indices, applied to every scheme")
    ap.add_argument("--n_checkpoints", type=int, default=6,
                     help="number of evenly-spaced checkpoints to evaluate per network")
    ap.add_argument("--out", default="training_trajectory_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    schemes = args.schemes.split(",")
    net_indices = [int(x) for x in args.net_indices.split(",")]

    print("=== Loading Henning references ===")
    vm_pop = np.load("henning_population_matrix.npy")
    vm_rdm = build_rdm_cosine(vm_pop)
    raw_pop = np.load("raw_population_matrix.npy")
    raw_rdm = build_rdm_cosine(raw_pop)
    circ_ref = circular_reference()
    references = {"von_mises": vm_rdm, "raw": raw_rdm}
    ref_resid = {name: rank_residualize(rdm, circ_ref) for name, rdm in references.items()}

    print("\n=== Building the 8-direction (Henning-matched) stimulus set ===")
    dataset = build_dataset()
    on_edge_indices = [i for i, row in dataset.arg_df.iterrows()
                        if row["intensity"] == INTENSITY]
    dt = dataset.dt

    all_results = []
    cell_types = None

    for scheme in schemes:
        for net_idx in net_indices:
            print(f"\n=== {scheme} net {net_idx:04d} ===")
            n_ckpts, first_iter, last_iter = get_checkpoint_count_and_iteration_anchors(
                scheme, net_idx)
            print(f"  {n_ckpts} real checkpoint files (ground truth from the file list, "
                  f"not chkpt_iter.h5's length), spanning iteration {first_iter} to "
                  f"{last_iter} (endpoints trusted exactly; positions in between are "
                  f"linearly interpolated, not exact)")

            positions = np.linspace(0, n_ckpts - 1, args.n_checkpoints, dtype=int)
            positions = sorted(set(positions.tolist()))

            for pos in positions:
                approx_iter = approx_iteration_for_position(pos, n_ckpts, first_iter, last_iter)
                print(f"  checkpoint position {pos} (approx. iteration {approx_iter})...",
                      end=" ")
                net = load_network_at_checkpoint(scheme, net_idx, pos).to(device)
                if cell_types is None:
                    cell_types = get_cell_types(net)
                rdm, pop_matrix = evaluate_checkpoint(net, dataset, on_edge_indices, dt,
                                                       cell_types, device)

                span, floor, ratio, ok = resolvability(rdm, pop_matrix)
                row = dict(scheme=scheme, net_idx=net_idx, checkpoint_position=pos,
                           n_checkpoints_total=n_ckpts,
                           approx_iteration=approx_iter, iteration_is_exact=(pos == 0),
                           rdm_span=span, rdm_floor=floor, resolvability_ratio=ratio,
                           resolvable=bool(ok))

                if not ok:
                    print(f"NOT RESOLVABLE (span/floor = {ratio:.2f}x, need >= "
                          f"{RESOLVE_MARGIN:.0f}x) -- no statistic reported, per "
                          f"Experiment 4b's established precision guard")
                    row["r_von_mises"] = None
                    row["r_raw"] = None
                else:
                    model_resid = rank_residualize(rdm, circ_ref)
                    for ref_name in references:
                        r, _ = spearmanr(model_resid, ref_resid[ref_name])
                        row[f"r_{ref_name}"] = float(r)
                    print(f"resolvable ({ratio:.1f}x floor)  "
                          f"r_von_mises={row['r_von_mises']:+.3f}  r_raw={row['r_raw']:+.3f}")
                all_results.append(row)

                del net
                if device == "cuda":
                    torch.cuda.empty_cache()

    n_unresolvable = sum(1 for r in all_results if not r["resolvable"])
    print(f"\n{'='*78}\nDOSE-RESPONSE TEST: does fidelity vs. Henning reference trend with "
          f"training iteration?\n{'='*78}")
    print("(iteration values are linear-interpolation estimates between trusted "
          "endpoints, not exact -- see get_checkpoint_count_and_iteration_anchors' "
          "docstring; only position 0, iteration -1, is exact)")
    if n_unresolvable:
        print(f"\n[!] {n_unresolvable}/{len(all_results)} checkpoint evaluations were "
              f"NOT RESOLVABLE (RDM span below {RESOLVE_MARGIN:.0f}x the float32 "
              f"round-off floor) and are excluded from every statistic below, not "
              f"averaged in as zero or dropped silently:")
        for r in all_results:
            if not r["resolvable"]:
                print(f"    {r['scheme']}/{r['net_idx']:04d} position {r['checkpoint_position']} "
                      f"(approx. iteration {r['approx_iteration']}): "
                      f"span/floor = {r['resolvability_ratio']:.2f}x")

    for ref_name in references:
        resolvable_results = [r for r in all_results if r["resolvable"]]
        iters = np.array([r["approx_iteration"] for r in resolvable_results])
        rs = np.array([r[f"r_{ref_name}"] for r in resolvable_results])
        trained_mask = iters >= 0
        if trained_mask.sum() >= 3:
            rho, p = spearmanr(iters[trained_mask], rs[trained_mask])
            print(f"\n[{ref_name}] Spearman(approx. iteration, fidelity_r) across all "
                  f"RESOLVABLE network-checkpoint pairs (n={trained_mask.sum()}): "
                  f"rho={rho:+.3f}, p={p:.4f}")
        pretrain_mask = iters < 0
        if pretrain_mask.sum() > 0:
            print(f"[{ref_name}] pre-training checkpoint(s) mean r = "
                  f"{rs[pretrain_mask].mean():+.3f} (n={pretrain_mask.sum()} resolvable "
                  f"of {sum(1 for r in all_results if r['approx_iteration'] == -1)} total, "
                  f"exact iteration -1)")

        print(f"[{ref_name}] per-network trajectories (X = not resolvable, excluded):")
        for scheme in schemes:
            for net_idx in net_indices:
                traj = [r for r in all_results
                        if r["scheme"] == scheme and r["net_idx"] == net_idx]
                parts = []
                for r in sorted(traj, key=lambda x: x["approx_iteration"]):
                    if r["resolvable"]:
                        parts.append(f"({r['approx_iteration']}, {r[f'r_{ref_name}']:+.3f})")
                    else:
                        parts.append(f"({r['approx_iteration']}, X)")
                print(f"    {scheme}/{net_idx:04d}: {'  '.join(parts)}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
