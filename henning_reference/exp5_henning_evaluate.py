#!/usr/bin/env python
"""
exp5_henning_evaluate.py -- re-evaluates the ALREADY-TRAINED Experiment 5
networks (degree_preserving_swap and erdos_renyi, n=10 each, full
250,000-iteration recipe) against the Henning et al. (2022) non-circular
biological reference, in place of the Maisak/von-Mises reference Exp5 was
originally evaluated against (which is 97.8% explained by circular
stimulus structure and left the degree-preserving-vs-ER contrast
uninterpretable).

WHY THIS IS CHEAP: no retraining. The Exp5 checkpoints already exist on
disk. This script changes ONLY the evaluation: the 12-direction,
30-degree MovingEdge stimulus set (matching Maisak) is replaced by the
8-direction, 45-degree set (matching Henning) -- the exact set already
validated in the "Experiment 1b" notebook, which replicated the
CC-vs-random fidelity signal at these 8 directions (cosine RDM
Spearman r = 0.691, p_perm = 0.0001). A forward pass on new stimuli
through an already-trained network is seconds per network; the whole
n=20 re-evaluation should take minutes, not GPU-hours.

WHAT THIS REUSES, AND WHY (do not reimplement any of these differently):
  - The 8-direction ANGLES and MovingEdge construction from
    moving_edge_on.ipynb's Experiment 1b -- byte-for-byte the same
    stimulus set the CC/random replication already validated.
  - The cosine-RDM construction from check_per_model_consistency_raw.py
    (manual per-pair loop, nan_to_num, +1e-10 epsilon) -- NOT
    scipy.spatial.distance.pdist, which was confirmed to diverge from
    this implementation at the ~1e-7 level, enough to flip rank order
    in a 28-entry RDM and change the aggregate per-model statistic.
    Every other Henning-reference result in this project uses this
    implementation; this script must match it for numbers to be
    comparable.
  - The rank-residualization circularity correction from
    correct_henning_reference.py / correct_exp5_circularity.py -- same
    technique, reused as a function rather than re-derived.
  - The population-vector extraction logic (peak central-cell voltage
    per cell type, fade-in state, overflow clipping) from
    production.py's get_population_vector -- unchanged from every other
    experiment in this project.

WHAT YOU NEED ON DISK, RELATIVE TO WHERE YOU RUN THIS:
  - flow/exp5_degree_preserving_swap/0000 .. 0009  (trained checkpoints)
  - flow/exp5_erdos_renyi/0000 .. 0009              (trained checkpoints)
    (adjust NET_INDICES below if your stable n=10 aren't indices 0-9 --
    check which ones actually produced rdm_net*.npy, not *.flag, in the
    original Exp5 run)
  - henning_population_matrix.npy, henning_reference_rdm.npy  (von Mises)
  - raw_population_matrix.npy                                (raw R_teta)

USAGE:
    python exp5_henning_evaluate.py \
        --schemes degree_preserving_swap,erdos_renyi \
        --net_indices 0,1,2,3,4,5,6,7,8,9

If your stable networks aren't a clean 0-9 per scheme, pass the actual
indices (comma-separated, same list applied to every scheme) or edit
NET_INDICES_BY_SCHEME below directly for a per-scheme list.
"""

import argparse
import os

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr, mannwhitneyu
from scipy.spatial.distance import cosine

N_DIRECTIONS = 8
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]  # matches Henning et al. 2022 exactly
INTENSITY = 1  # ON edges only
OVERFLOW_LIMIT = 1e6


# ======================================================================
#  RDM construction -- LOCKED to check_per_model_consistency_raw.py's
#  implementation. Do not swap in pdist; see module docstring.
# ======================================================================
def build_rdm_cosine(pop_matrix):
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3)
    pop_matrix = pop_matrix + 1e-10
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rdm[i, j] = cosine(pop_matrix[i], pop_matrix[j])
    return rdm


def rank_residualize(rdm, against_rdm):
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    r_vals = rankdata(rdm[iu])
    r_against = rankdata(against_rdm[iu])
    slope, intercept = np.polyfit(r_against, r_vals, 1)
    predicted = slope * r_against + intercept
    return r_vals - predicted


def circular_reference(n=N_DIRECTIONS, angles=ANGLES):
    ref = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(angles[i] - angles[j])
            ref[i, j] = min(d, 360 - d)
    return ref


# ======================================================================
#  Stimulus set -- identical to moving_edge_on.ipynb's Experiment 1b
# ======================================================================
def build_dataset():
    from flyvis.datasets.moving_bar import MovingEdge
    return MovingEdge(
        offsets=[-10, 11],
        intensities=[0, 1],
        speeds=[19],
        height=80,
        post_pad_mode="continue",
        t_pre=1.0,
        t_post=1.0,
        dt=1 / 200,
        angles=ANGLES,
    )


# ======================================================================
#  Population vector extraction -- unchanged from production.py /
#  moving_edge_on.ipynb
# ======================================================================
def get_population_vector(network, stimulus, dt, cell_types):
    from flyvis.utils.activity_utils import LayerActivity
    if stimulus.dim() == 2:
        stimulus = stimulus.unsqueeze(1)
    initial_state = network.fade_in_state(1.0, dt, stimulus[[0]])
    with torch.no_grad():
        responses = network.simulate(stimulus[None], dt, initial_state=initial_state).cpu()
    layer_act = LayerActivity(responses, network.connectome, keepref=True)
    pop_vec = np.array([layer_act.central[ct].squeeze().numpy().max() for ct in cell_types])
    pop_vec = np.clip(pop_vec, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
    del responses, layer_act
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pop_vec


def get_cell_types(network):
    return [ct.decode() if isinstance(ct, bytes) else ct
            for ct in network.connectome.unique_cell_types[:]]


def load_exp5_network(scheme, net_idx, results_root=None):
    """Loads a fully-trained Exp5 network from its flyvis results dir.

    Reuses production.py's load_trained_network logic exactly (not a
    plain NetworkView().init_network() call): Flyvis's default
    best-checkpoint selection (best_checkpoint_default_fn) has a
    confirmed bug -- it uses np.argmin's position WITHIN the
    validation-loss array as a direct index into the (usually shorter)
    list of checkpoint indices, which crashes whenever validation is
    recorded more often than checkpoints are saved. That was observed
    on these exact training logs. The workaround below sidesteps the
    bug entirely by not touching the loss array -- it just returns the
    highest-iteration checkpoint directly, which is what "fully
    trained, evaluate the final result" should mean here anyway.
    """
    from flyvis.network import NetworkView
    from flyvis.utils.chkpt_utils import checkpoint_index_to_path_map
    import flyvis
    if results_root is None:
        results_root = str(flyvis.results_dir)

    network_name = f"exp5_{scheme}/{net_idx:04d}"
    path = os.path.join(results_root, "flow", network_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"no trained network at {path} -- check scheme name, net_idx, "
            f"and that this network actually completed training (has a "
            f"final checkpoint, not just partial/crashed ones)")

    def last_checkpoint_fn(p, **kwargs):
        networkdir = flyvis.NetworkDir(p)
        checkpoint_dir = networkdir.chkpts.path
        indices, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
        if not paths:
            raise FileNotFoundError(f"no checkpoints found for {p}")
        return paths[-1]

    nv = NetworkView(
        os.path.join(results_root, "flow", network_name),
        best_checkpoint_fn=last_checkpoint_fn,
    )
    return nv.init_network()


# ======================================================================
#  Per-scheme evaluation
# ======================================================================
def evaluate_scheme(scheme, net_indices, dataset, on_edge_indices, dt, device):
    print(f"\n=== {scheme}: evaluating {len(net_indices)} trained networks "
          f"on {len(on_edge_indices)} Henning-matched (8-direction) stimuli ===")
    rdms = []
    cell_types = None
    for net_idx in net_indices:
        net = load_exp5_network(scheme, net_idx).to(device)
        if cell_types is None:
            cell_types = get_cell_types(net)
        pop_vecs = []
        for stim_idx in on_edge_indices:
            stimulus = dataset[stim_idx]
            if not isinstance(stimulus, torch.Tensor):
                stimulus = torch.tensor(stimulus, dtype=torch.float32)
            stimulus = stimulus.to(device)
            pop_vecs.append(get_population_vector(net, stimulus, dt, cell_types))
        pop_matrix = np.stack(pop_vecs, axis=0)
        rdms.append(build_rdm_cosine(pop_matrix))
        print(f"  net {net_idx:04d}: done")
        del net
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.array(rdms), cell_types


def summarize_scheme(name, rdms, ref_resid_dict, circ_ref):
    """ref_resid_dict: {'von_mises': resid, 'raw': resid} -- both references."""
    mean_rdm = rdms.mean(axis=0)
    mean_resid = rank_residualize(mean_rdm, circ_ref)

    results = {"scheme": name, "n": rdms.shape[0]}
    for ref_name, ref_resid in ref_resid_dict.items():
        # ensemble-mean
        r_mean, p_mean = spearmanr(mean_resid, ref_resid)
        # per-network
        per_net = np.array([
            spearmanr(rank_residualize(rdm, circ_ref), ref_resid)[0] for rdm in rdms
        ])
        print(f"  [{ref_name}] ensemble-mean partial r = {r_mean:+.4f} (p={p_mean:.4f})  "
              f"| per-network mean = {per_net.mean():+.4f} "
              f"({100*np.mean(per_net < 0):.0f}% negative)")
        results[ref_name] = dict(
            ensemble_mean_r=float(r_mean), ensemble_mean_p=float(p_mean),
            per_network_r=per_net.tolist(), per_network_mean=float(per_net.mean()),
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="degree_preserving_swap,erdos_renyi")
    ap.add_argument("--net_indices", default="0,1,2,3,4,5,6,7,8,9",
                     help="comma-separated net indices, applied to every scheme "
                          "unless you edit the per-scheme dict below")
    ap.add_argument("--out", default="exp5_henning_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_indices = [int(x) for x in args.net_indices.split(",")]
    schemes = args.schemes.split(",")

    print("=== Loading Henning references ===")
    vm_pop = np.load("henning_population_matrix.npy")
    vm_rdm = build_rdm_cosine(vm_pop)
    raw_pop = np.load("raw_population_matrix.npy")
    raw_rdm = build_rdm_cosine(raw_pop)

    circ_ref = circular_reference()
    ref_resid = {
        "von_mises": rank_residualize(vm_rdm, circ_ref),
        "raw": rank_residualize(raw_rdm, circ_ref),
    }

    print("\n=== Building the 8-direction (Henning-matched) stimulus set ===")
    dataset = build_dataset()
    on_edge_indices = [i for i, row in dataset.arg_df.iterrows()
                        if row["intensity"] == INTENSITY]
    print(f"  {len(on_edge_indices)} ON-edge conditions")

    all_results = {}
    scheme_rdms = {}
    for scheme in schemes:
        rdms, _ = evaluate_scheme(scheme, net_indices, dataset, on_edge_indices,
                                   dataset.dt, device)
        scheme_rdms[scheme] = rdms
        all_results[scheme] = summarize_scheme(scheme, rdms, ref_resid, circ_ref)

    # the interpretable quantity: degree-preserving vs ER contrast, per reference
    if "degree_preserving_swap" in all_results and "erdos_renyi" in all_results:
        print(f"\n{'='*70}\nDEGREE-PRESERVING vs ER CONTRAST (the interpretable quantity)\n{'='*70}")
        for ref_name in ref_resid:
            ds = np.array(all_results["degree_preserving_swap"][ref_name]["per_network_r"])
            er = np.array(all_results["erdos_renyi"][ref_name]["per_network_r"])
            u, p = mannwhitneyu(ds, er, alternative="two-sided")
            print(f"  [{ref_name}] degree_preserving_swap mean={ds.mean():+.4f}  "
                  f"vs  erdos_renyi mean={er.mean():+.4f}  "
                  f"(Mann-Whitney U={u:.1f}, p={p:.4f})")
            all_results.setdefault("contrast", {})[ref_name] = dict(
                U=float(u), p=float(p))

    import json
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")

    for scheme in schemes:
        np.save(f"exp5_henning_rdms_{scheme}.npy", scheme_rdms[scheme])
        print(f"wrote exp5_henning_rdms_{scheme}.npy "
              f"(shape {scheme_rdms[scheme].shape}, per-network RDMs)")


if __name__ == "__main__":
    main()
