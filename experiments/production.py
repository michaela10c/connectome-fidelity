#!/usr/bin/env python3
"""
experiment5_production.py  —  Experiment 5 (Version A): trained-random baseline

Trains an ensemble of random-WIRED Flyvis networks from scratch on the optic-flow
task using the SAME training recipe that produced the connectome-constrained (CC)
ensemble, then evaluates their representational geometry against the Maisak T4/T5
biological reference with RSA — the SAME readout as Experiments 1-3.

Logic: Experiment 3 found CC-vs-biology r=0.930 vs random-vs-biology r=0.603, but
that random baseline was a WEIGHT shuffle on fixed real wiring, and the CC nets were
both wired AND trained. Experiment 5 holds TRAINING CONSTANT (train random-WIRED nets
identically) to probe the WIRING axis — the complement of Exp 4 (which perturbs wiring
WITHOUT training).

IMPORTANT — do NOT read a single trained-random r as a "training vs wiring" verdict:
  - The Exp 3 0.603 baseline is filtered by resample-to-N-stable; this Exp 5 null is
    filtered by training divergence. Different nulls, different stability filters —
    r is not a drop-in substitute for 0.603.
  - Degree-PRESERVING schemes are the stringent case: the project's MICrONS result
    found the real connectome indistinguishable from a degree-preserving null
    (z=1.30, p=0.157). A high r on a degree-preserving scheme means "degree structure
    suffices", NOT "training suffices". The interpretable quantity is the CONTRAST
    between a degree-preserving scheme and a degree-BREAKING one (e.g. erdos_renyi).
  - Any r-vs-bio needs the within-polarity / direction-tuning decomposition before it
    can be interpreted (cf. Exp 3's Exp-2 near-parity that resolved only after
    decomposition).
See Exp5_scheme_and_sizing_plan for the scheme-priority and phased-sizing rationale.

============================ HOW TRAINING WORKS ============================
We do NOT hand-roll a training loop (that would diverge from how the CC ensemble
was trained and reintroduce a confound). We call Flyvis's own entry point,
`flyvis train-single`, with the connectome file swapped to a randomized JSON.
Everything else — Adam, stepwise LR 5e-5->5e-6, dt=0.02, l2norm flow loss,
DecoderGAVP(shape=[8,2], kernel_size=5), activity penalty, checkpoint-every-300-
epochs — is inherited from the default flow config, identical to the CC ensemble.

Confirmed from config:
  - override key:   network.connectome.file=<randomized JSON>
  - task.n_iters:   250000  (full production; this is the CC recipe)
  - batch_size:     4
  - decoder/loss:   DecoderGAVP[8,2] k=5 / l2norm  (matches Exp 1-3 readout pipeline)

============================ COMPUTE REALITY ============================
250k iterations of BPTT per network. On a T4 this is many hours to >1 day PER
network; 50 networks is a CLUSTER job, not Colab. This script is structured to
run per-network and is resumable (skip networks whose checkpoint/RDM already
exists), so it can be driven by a cluster array job (one network per task) or a
checkpointed long-running process. Do NOT expect to finish this in one Colab
session.

For a fast end-to-end correctness test (NOT a result), set --n_iters small
(e.g. 2000) and --n_networks 2; the numbers will be meaningless but it proves
the train -> recover -> evaluate -> RSA path runs unbroken.

============================ EVALUATION ============================
The evaluation half is lifted VERBATIM from Experiment 4 so the geometry and the
biological reference are byte-identical to the CC experiments:
  MovingEdge (ON, 12 dir) -> get_population_vector (peak central-cell/cell-type,
  65-dim) -> build_rdm cosine -> permutation_test_rdm vs von Mises Maisak RDM.

USAGE (after Choi picks a scheme, e.g. sign_preserving_target_perm):
    python experiment5_production.py \
        --scheme sign_preserving_target_perm \
        --n_networks 50 \
        --seeds 0-9 \
        --connectome_dir /content/rand_out \
        --out_dir /content/exp5_out
"""

import os, sys, glob, json, argparse, subprocess
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import cosine as cosine_dist

OVERFLOW_LIMIT = 1e6
SEED = 42
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
INTENSITY = 1  # ON edges, matches Experiment 1/3


# ======================================================================
#  EVALUATION  — lifted verbatim from Experiment 4 (do not "improve")
# ======================================================================
def get_cell_types(network):
    return [ct.decode() if isinstance(ct, bytes) else ct
            for ct in network.connectome.unique_cell_types[:]]


def get_population_vector(network, stimulus, dt, cell_types):
    from flyvis.utils.activity_utils import LayerActivity
    if stimulus.dim() == 2:
        stimulus = stimulus.unsqueeze(1)
    initial_state = network.fade_in_state(1.0, dt, stimulus[[0]])
    with torch.no_grad():
        responses = network.simulate(stimulus[None], dt, initial_state=initial_state).cpu()
    layer_act = LayerActivity(responses, network.connectome, keepref=True)
    pop_vec = np.array([layer_act.central[ct].squeeze().numpy().max()
                        for ct in cell_types])
    pop_vec = np.clip(pop_vec, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
    del responses, layer_act
    torch.cuda.empty_cache()
    return pop_vec


def build_rdm(pop_matrix, metric="cosine"):
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3)
    if metric == "cosine":
        pop_matrix = pop_matrix + 1e-10
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rdm[i, j] = cosine_dist(pop_matrix[i], pop_matrix[j])
    return rdm


def rdm_similarity(rdm1, rdm2):
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    r_s, p_s = spearmanr(rdm1[idx], rdm2[idx])
    r_k, p_k = kendalltau(rdm1[idx], rdm2[idx])
    return r_s, p_s, r_k, p_k


def permutation_test_rdm(rdm1, rdm2, n_permutations=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    obs_r, _ = spearmanr(rdm1[idx], rdm2[idx])
    obs_tau, _ = kendalltau(rdm1[idx], rdm2[idx])
    null_r = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        rdm2_p = rdm2[np.ix_(perm, perm)]
        null_r[i], _ = spearmanr(rdm1[idx], rdm2_p[idx])
    p_r = float(np.mean(null_r >= obs_r))
    return obs_r, p_r, obs_tau, null_r


def build_bio_rdm():
    """Von Mises T4/T5 reference RDM, 12 ON directions (Maisak 2013). Verbatim Exp 4."""
    directions = np.linspace(0, 330, 12)
    preferred = [180, 0, 90, 270, 180, 0, 90, 270]
    kappa = 2.5

    def vm(theta, mu):
        r = np.exp(kappa * np.cos(np.radians(theta - mu)))
        r = r - np.exp(-kappa)
        return max(r, 0)

    pop = np.array([[vm(d, mu) for mu in preferred] for d in directions])
    return build_rdm(pop, metric="cosine")


# ======================================================================
#  TRAINING  — delegate to flyvis train-single (the CC recipe)
# ======================================================================
def train_one_network(network_name, connectome_file, n_iters, extra_overrides=None):
    """Train one random-wired network from scratch via the Flyvis CLI.

    Identical recipe to the CC ensemble; the ONLY change is the connectome file.
    Resumable: resumes only if a checkpoint already exists for this network, so
    a fresh network does not enter solver.recover() (which is broken in some
    Flyvis versions: resolve_checkpoints() arity mismatch on the resume path).
    """
    from flyvis import results_dir
    # A network is resumable only if it already has a checkpoint on disk.
    # Path includes the task_name prefix (flow/) that the train CLI composes.
    chkpt_glob = os.path.join(str(results_dir), "flow", network_name, "chkpts", "*")
    has_checkpoint = len(glob.glob(chkpt_glob)) > 0
    resume_flag = "false"
    overrides = [
        f"ensemble_and_network_id={network_name}",
        "task_name=flow",
        "train=true",
        f"resume={resume_flag}",              # only resume when a checkpoint exists
        f"task.n_iters={n_iters}",
        f"network.connectome.file={connectome_file}",
        "description=exp5_trained_random",
    ]
    if extra_overrides:
        overrides += extra_overrides
    # datamate enforces strict config-matching when (re)opening a NetworkDir.
    # On RESUME this rejects the existing dir with a FileExistsError citing
    # delete_if_exists (TuragaLab/flyvis#4). The maintainer-blessed fix is:
    #   import datamate; datamate.enforce_config_match(False)
    # which just sets a module global: context.enforce_config_match = False
    # (confirmed from datamate 1.0.0 source).
    #
    # Training runs in a SUBPROCESS via the `flyvis train-single` console script,
    # so we cannot set that global from our process. We also cannot call flyvis's
    # Hydra `main` directly (Hydra loses its config_path context -> "Primary
    # config module 'flyvis.config' not found"). Solution: a sitecustomize.py
    # shim. Python auto-imports `sitecustomize` at interpreter startup, BEFORE
    # flyvis imports datamate, in every subprocess whose PYTHONPATH includes it.
    # The shim disables strict matching; we then call the NORMAL console script
    # so Hydra's context stays intact. We never use delete_if_exists=true (that
    # would delete the checkpoint we are resuming from).
    shim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dm_shim")
    os.makedirs(shim_dir, exist_ok=True)
    shim_file = os.path.join(shim_dir, "sitecustomize.py")
    if not os.path.exists(shim_file):
        with open(shim_file, "w") as f:
            f.write(
                "# Auto-generated: disable datamate strict config-match in any\n"
                "# subprocess that has this dir on PYTHONPATH (needed for resume).\n"
                "try:\n"
                "    import datamate\n"
                "    datamate.enforce_config_match(False)\n"
                "except Exception as _e:\n"
                "    import sys; print('sitecustomize: datamate disable failed:', _e, file=sys.stderr)\n"
            )
    env = dict(os.environ)
    env["PYTHONPATH"] = shim_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd = ["flyvis", "train-single", *overrides]
    print(f"    > [resume={resume_flag}] (datamate strict-match disabled via shim)",
          " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def latest_checkpoint_iter(network_name, task_name="flow"):
    """Return the highest TRUE iteration among this network's checkpoints, or -1
    if none. Used to distinguish a FINISHED network from a partially-trained one.

    Flyvis records the actual iteration of each checkpoint in chkpt_iter.h5
    (dataset 'data'), e.g. [-1, 11, 2003] for a 2000-iter run (the -1 is the
    pre-training checkpoint; 2003 is the final). The checkpoint FILENAMES
    (chkpt_00000, 00001, 00002) are sequential INDICES, NOT iterations — do not
    parse them. We read chkpt_iter.h5 and take the max.

    Path note: chkpt_iter.h5 lives in the network dir (NOT inside chkpts/).
    Returns -1 if the file is absent/unreadable (fail safe: treated as not
    finished, so we never falsely skip an unfinished network).
    """
    from flyvis import results_dir as _rd
    net_dir = os.path.join(str(_rd), task_name, network_name)
    iter_h5 = os.path.join(net_dir, "chkpt_iter.h5")
    if not os.path.exists(iter_h5):
        return -1
    try:
        import h5py
        with h5py.File(iter_h5, "r") as f:
            data = f["data"][()]
        vals = [int(x) for x in list(data)]
        return max(vals) if vals else -1
    except Exception as e:
        print(f"    [warn] could not read {iter_h5}: {e} -> treating as not finished")
        return -1


def is_training_complete(network_name, n_iters, task_name="flow", tol_frac=0.99):
    """True only if the latest checkpoint reached (near) the target n_iters.
    tol_frac guards small overshoot/undershoot (final checkpoint often lands a
    few iters past, e.g. 2003 for n_iters=2000, or a bit short). Conservative:
    partial -> False -> resume/retrain rather than risk evaluating an
    undertrained network.
    """
    latest = latest_checkpoint_iter(network_name, task_name)
    return latest >= int(tol_frac * n_iters)


def load_trained_network(network_name, task_name="flow"):
    from flyvis.network import NetworkView
    from flyvis import results_dir
    from flyvis.utils.chkpt_utils import checkpoint_index_to_path_map
    import flyvis

    def last_checkpoint_fn(path, **kwargs):
        """Replacement for Flyvis's best_checkpoint_default_fn, which has a
        real bug: it uses np.argmin's position WITHIN the validation-loss
        array as a direct index into the (usually much shorter) list of
        checkpoint indices -- crashes whenever validation is recorded more
        often than checkpoints are saved (confirmed: this network's log
        showed several 'Test on validation data' events between each single
        'Checkpointed' event). Sidesteps the bug by not touching the loss
        array at all -- reuses Flyvis's own (correct) checkpoint-discovery
        logic, just returns the highest-iteration checkpoint directly.
        Accepts **kwargs to match whatever best_checkpoint_fn_kwargs Flyvis
        passes (validation_subdir, loss_file_name), even though unused.
        """
        networkdir = flyvis.NetworkDir(path)
        checkpoint_dir = networkdir.chkpts.path
        indices, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
        if not paths:
            raise FileNotFoundError(f"no checkpoints found for {path}")
        return paths[-1]

    nv = NetworkView(
        results_dir / task_name / network_name,
        best_checkpoint_fn=last_checkpoint_fn,
    )
    return nv.init_network()

# ======================================================================
#  ENSEMBLE ORCHESTRATION
# ======================================================================
def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def is_stable(network, stimulus, dt):
    """Forward-pass stability test. Returns True iff output is finite and
    < OVERFLOW_LIMIT in absolute value.

    Criterion lifted VERBATIM from Experiment 4 (untrained_networks.py
    is_stable, which itself mirrors randomize_weights_stable from Exp 1-3),
    so trained-random nets are filtered by the IDENTICAL rule used to filter
    every other ensemble in the project. Exp 1-4 could resample to N stable
    cheaply (a shuffle is milliseconds); Exp 5 cannot (a draw is ~20 GPU-hours),
    so here we EXCLUDE-AND-REPORT rather than resample: an unstable trained net
    is dropped from the ensemble and the surviving n is reported, exactly as
    Exp 4 reports `accepted`.
    """
    if stimulus.dim() == 2:
        stim = stimulus.unsqueeze(1)
    else:
        stim = stimulus
    try:
        initial_state = network.fade_in_state(1.0, dt, stim[[0]])
        out = network.simulate(stim[None], dt, initial_state=initial_state)
        out_np = out.cpu().numpy()
        return bool(torch.all(torch.isfinite(out)) and
                    np.all(np.abs(out_np) < OVERFLOW_LIMIT))
    except Exception:
        return False


def evaluate_network(net, dataset, on_edge_indices, dt, cell_types):
    pop_vecs = []
    for stim_idx in on_edge_indices:
        stim = dataset[stim_idx]
        if not isinstance(stim, torch.Tensor):
            stim = torch.tensor(stim, dtype=torch.float32)
        pop_vecs.append(get_population_vector(net, stim, dt, cell_types))
    return build_rdm(np.stack(pop_vecs, axis=0), "cosine")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True,
                    choices=["degree_preserving", "degree_preserving_swap",
                             "rf_shuffle", "sign_preserving_target_perm",
                             "erdos_renyi"])
    ap.add_argument("--n_networks", type=int, default=50)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--connectome_dir", default="/content/rand_out")
    ap.add_argument("--out_dir", default="/content/exp5_out")
    ap.add_argument("--n_iters", type=int, default=250000,
                    help="250000 = full CC recipe. Use small for a smoke test only.")
    ap.add_argument("--n_perm", type=int, default=10000)
    # --- array-job support ---
    # In a SLURM array, each task trains ONE network then exits; no single task
    # ever holds all 50 RDMs in memory, so the in-memory ensemble RSA at the end
    # cannot run per-task. Two modes handle this:
    #   --only_net N      : process exactly network N (one array task = one network)
    #   --aggregate_only  : skip all training/eval; read the saved rdm_net*.npy from
    #                       disk and compute the ensemble RSA (run as a final task
    #                       after the array completes)
    ap.add_argument("--only_net", type=int, default=None,
                    help="Process only this single network index (for array jobs).")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip train/eval; compute ensemble RSA from saved RDMs on disk.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  scheme: {args.scheme}  n_iters: {args.n_iters}")
    if args.n_iters < 250000:
        print("[!] n_iters < 250000 — SMOKE TEST mode. Results are NOT valid; "
              "this only checks the train->recover->evaluate->RSA path runs.")

    # ---- stimulus dataset (verbatim Exp 4) ----
    from flyvis.datasets.moving_bar import MovingEdge
    dataset = MovingEdge(
        offsets=[-10, 11], intensities=[0, 1], speeds=[19], height=80,
        post_pad_mode="continue", t_pre=1.0, t_post=1.0, dt=1/200, angles=ANGLES)
    on_edge_indices = [i for i, row in dataset.arg_df.iterrows()
                       if row["intensity"] == INTENSITY]
    dt = dataset.dt

    bio_rdm = build_bio_rdm()
    seeds = parse_seeds(args.seeds)

    def aggregate_from_disk():
        """Compute ensemble RSA from the rdm_net*.npy files on disk, over the
        STABLE subset only.

        Three states per network index:
          - rdm_net{k}.npy exists      -> stable, included in ensemble
          - unstable_net{k}.flag exists -> trained but failed is_stable, EXCLUDED
          - neither                     -> not yet trained/evaluated (pending)
        Reports n_stable / n_unstable / n_pending the way Exp 4 reports `accepted`,
        so Exp 5's stability accounting is visibly identical to the rest of the
        project. The ensemble mean is over stable nets ONLY — unstable nets are
        never averaged in.
        """
        rdms, unstable, pending = [], [], []
        for k in range(args.n_networks):
            rdm_p  = os.path.join(args.out_dir, f"rdm_net{k:03d}.npy")
            flag_p = os.path.join(args.out_dir, f"unstable_net{k:03d}.flag")
            if os.path.exists(rdm_p):
                rdms.append(np.load(rdm_p))
            elif os.path.exists(flag_p):
                unstable.append(k)
            else:
                pending.append(k)
        n_trained = len(rdms) + len(unstable)
        if unstable:
            print(f"[aggregate] {len(unstable)} unstable nets EXCLUDED: {unstable}")
        if pending:
            print(f"[aggregate] {len(pending)} nets not yet trained/evaluated: {pending}")
        if not rdms:
            sys.exit("[aggregate] no stable RDMs on disk; nothing to aggregate.")
        mean_rdm = np.mean(np.stack(rdms), axis=0)
        r, p, tau, _ = permutation_test_rdm(mean_rdm, bio_rdm, n_permutations=args.n_perm)
        # Instability rate is itself a result (cf. the 66-80% trained-shuffle
        # instability documented in Exp 1-2); report it when any net has trained.
        instab_rate = (len(unstable) / n_trained) if n_trained else float("nan")
        print("\n=== EXPERIMENT 5 RESULT ===")
        print(f"scheme: {args.scheme}")
        print(f"ensemble: n={len(rdms)} STABLE of {n_trained} trained "
              f"({len(unstable)} unstable excluded, {len(pending)} pending; "
              f"target {args.n_networks})")
        if n_trained:
            print(f"trained-random instability rate: {instab_rate:.1%} "
                  f"({len(unstable)}/{n_trained})")
        print(f"trained-random vs biology:  r = {r:.3f}  p_perm = {p:.4f}  tau = {tau:.3f}")
        print(f"reference (Exp 3, SAME readout): CC vs bio = 0.930,  "
              f"weight-shuffle random vs bio = 0.603")
        print("  INTERPRETATION — report, decompose, THEN conclude (do not over-read one r):")
        print("  - The Exp 3 'random' (0.603) is a WEIGHT shuffle on FIXED real wiring,")
        print("    stability-filtered by resample-to-N. This Exp 5 null is a WIRING scramble")
        print("    + retrain, filtered by TRAINING divergence (see instability rate above).")
        print("    Different nulls, different stability filters: r is NOT a drop-in substitute")
        print("    for 0.603, and this is the complement of Exp 4 (rewire-then-retrain vs")
        print("    rewire-without-training), not a standalone 'training vs wiring' verdict.")
        print(f"  - Scheme = '{args.scheme}'. Degree-PRESERVING nulls are the STRINGENT case:")
        print("    the MICrONS mouse result found the real connectome indistinguishable from")
        print("    a degree-preserving null (z=1.30, p=0.157). A high r on a degree-preserving")
        print("    scheme is consistent with 'DEGREE STRUCTURE suffices', NOT necessarily")
        print("    'training suffices'. Degree-BREAKING schemes (e.g. erdos_renyi) localize")
        print("    whether wiring BEYOND degree carries signal; the degree-vs-degree-breaking")
        print("    CONTRAST is the interpretable quantity, not either scheme's r alone.")
        print("  - A bare r-vs-bio still needs the within-polarity / direction-tuning")
        print("    decomposition (cf. Exp 3 Exp-2, where apparent near-parity resolved into a")
        print("    real signal only after decomposition) before it can be interpreted.")
        result = dict(scheme=args.scheme,
                      n_stable=len(rdms), n_unstable=len(unstable),
                      n_pending=len(pending), n_trained=n_trained,
                      n_target=args.n_networks,
                      instability_rate=(float(instab_rate) if n_trained else None),
                      unstable_indices=unstable, pending_indices=pending,
                      r_trained_random_vs_bio=float(r), p_perm=float(p),
                      cc_vs_bio_ref=0.930, random_vs_bio_ref=0.603, n_iters=args.n_iters)
        with open(os.path.join(args.out_dir, "exp5_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        np.save(os.path.join(args.out_dir, "mean_rand_rdm.npy"), mean_rdm)
        print(f"\nsaved {os.path.join(args.out_dir, 'exp5_result.json')}")

    if args.aggregate_only:
        aggregate_from_disk()
        return

    cell_types = None
    net_indices = ([args.only_net] if args.only_net is not None
                   else range(args.n_networks))
    for net_idx in net_indices:
        seed = seeds[net_idx % len(seeds)]
        connectome_file = os.path.abspath(
            os.path.join(args.connectome_dir, f"fib25_{args.scheme}_seed{seed}.json"))
        network_name = f"exp5_{args.scheme}/{net_idx:04d}"
        rdm_path = os.path.join(args.out_dir, f"rdm_net{net_idx:03d}.npy")

        if os.path.exists(rdm_path):
            print(f"[{net_idx:03d}] cached RDM, skipping")
            continue
        if not os.path.exists(connectome_file):
            sys.exit(f"missing connectome: {connectome_file} (generate seeds first)")

        # Skip training ONLY if this network's training actually REACHED n_iters.
        # A non-empty chkpts dir is NOT sufficient: Flyvis checkpoints periodically,
        # so a network that crashed mid-training also has checkpoints. We compare the
        # latest checkpoint's iteration against the target (see is_training_complete).
        # - complete            -> skip training, go straight to evaluation
        # - partial (crashed)   -> fall through to train_one_network, which sees the
        #                          existing checkpoint and resumes (resume=true)
        # - none                -> train from scratch
        if is_training_complete(network_name, args.n_iters):
            print(f"[{net_idx:03d}] training complete (reached n_iters) -> evaluating")
        else:
            latest = latest_checkpoint_iter(network_name)
            if latest >= 0:
                print(f"[{net_idx:03d}] partial (latest iter {latest} < {args.n_iters}) "
                      f"-> resuming")
            else:
                print(f"[{net_idx:03d}] train (seed {seed})  connectome={connectome_file}")
            train_one_network(network_name, connectome_file, args.n_iters)

        net = load_trained_network(network_name).to(device)
        if cell_types is None:
            cell_types = get_cell_types(net)

        # --- STABILITY GATE (exclude-and-report; matches Exp 1-4 methodology) ---
        # A trained-random net can be dynamically unstable (the trained random
        # baselines in Exp 1-2 were 66-80% unstable). Exp 1-4 excluded unstable
        # nets by construction (resample-to-N-stable). We cannot resample here
        # (each draw is ~20 GPU-hours), so we EXCLUDE the unstable net from the
        # ensemble and RECORD it, rather than averaging its clipped/sanitized
        # garbage RDM into the mean. Test on the first ON stimulus, same as Exp 4's
        # run_condition uses a single check-stimulus.
        check_stim = dataset[on_edge_indices[0]]
        if not isinstance(check_stim, torch.Tensor):
            check_stim = torch.tensor(check_stim, dtype=torch.float32)
        check_stim = check_stim.to(device)
        if not is_stable(net, check_stim, dt):
            flag_path = os.path.join(args.out_dir, f"unstable_net{net_idx:03d}.flag")
            with open(flag_path, "w") as f:
                f.write(f"network {network_name} failed is_stable "
                        f"(non-finite or >= {OVERFLOW_LIMIT:.0e}); excluded from ensemble\n")
            print(f"[{net_idx:03d}] UNSTABLE -> excluded from ensemble "
                  f"(marker: {flag_path}); no RDM saved", flush=True)
            del net
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        print(f"[{net_idx:03d}] evaluating: building RDM over {len(on_edge_indices)} stimuli...", flush=True)
        this_rdm = evaluate_network(net, dataset, on_edge_indices, dt, cell_types)
        print(f"[{net_idx:03d}] RDM built, shape={this_rdm.shape}; saving to {rdm_path}", flush=True)
        # Atomic write: save to a temp file then rename, so a job killed mid-write
        # never leaves a truncated .npy that the resume logic would load as valid.
        # NOTE: np.save auto-appends ".npy" if the path lacks it, so a temp path
        # ending in ".tmp" becomes "<...>.tmp.npy" on disk — name the temp with a
        # .npy suffix so the actual written file matches what we rename.
        tmp_path = rdm_path + ".tmp.npy"
        np.save(tmp_path, this_rdm)          # writes exactly tmp_path (already ends .npy)
        os.replace(tmp_path, rdm_path)       # atomic on POSIX
        print(f"[{net_idx:03d}] saved RDM -> {rdm_path} (exists={os.path.exists(rdm_path)})", flush=True)
        del net
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- ensemble RSA vs biology ----
    # A single-network array task (--only_net) must NOT aggregate: it has produced
    # only its own RDM and the others may not exist yet. Aggregation is done either
    # by the full single-process run (below) or by a final --aggregate_only task.
    if args.only_net is not None:
        print(f"[{args.only_net:03d}] done (single-network task; run --aggregate_only "
              f"after the array completes to compute the ensemble result).")
        return
    aggregate_from_disk()


if __name__ == "__main__":
    main()
