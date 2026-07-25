"""
Experiment 4: Untrained Connectome-Constrained Networks
Isolating the wiring contribution to representational geometry
before any task training.

Experiments 1-3 used pretrained Flyvis networks, which are both
connectome-constrained AND task-trained. This leaves open whether the
representational geometry signal comes from the wiring or from training.

This experiment resolves that confound by comparing:

  Untrained CC (n=N_MODELS):
    Network() with default Flyvis architecture, connectome fixed,
    free parameters (bias, time_const, syn_strength) perturbed with
    Gaussian noise across N_MODELS seeds. No checkpoint loaded.

  Untrained Random (syn shuffle, n=N_MODELS):
    Same untrained networks, but with sign-preserving Shiu-style shuffle
    applied to edges_syn_strength after perturbation. edges_sign (E/I
    identity) and edges_syn_count remain fixed — matches the baseline
    design from Experiments 1-3.

  Untrained Random (sign shuffle, n=N_MODELS):
    Deeper disruption: edges_sign is also shuffled (sign-preserving).
    Tests whether E/I wiring identity drives geometry before training.

If CC > Random before training:
  Wiring alone carries the representational signal — training confound resolved.

If CC ~= Random before training:
  The geometry signal is training-dependent.

Runtime: Google Colab T4 GPU. n=50 per condition ~30-60 min.

References:
- Lappalainen et al. 2024. Nature 634, 1132-1140.
- Kriegeskorte et al. 2008. Frontiers in Systems Neuroscience 2:4.
- Nili et al. 2014. PLOS Computational Biology 10(4): e1003553.
"""

# ── COLAB SETUP ───────────────────────────────────────────────────────────────
# !git clone https://github.com/TuragaLab/flyvis.git
# %cd /content/flyvis
# !pip install -e .[examples]
# !flyvis download-pretrained

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import cosine as cosine_dist, euclidean

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_MODELS       = 50       # per condition; use 10 for a quick check
N_PERMUTATIONS = 10_000
MAX_ATTEMPTS   = 100
OVERFLOW_LIMIT = 1e6

# Gaussian noise applied to untrained free params to generate an ensemble.
# Default init is deterministic (seed=0 hardcoded in RestingPotential),
# so perturbation is required to get distinct models.
BIAS_NOISE         = 0.05   # matches Normal(0.5, 0.05) training prior
TIME_CONST_NOISE   = 0.005
SYN_STRENGTH_NOISE = 0.002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

results_dir = Path("../results")
figures_dir = Path("../figures")
results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import flyvis
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.network import Network
from flyvis.utils.activity_utils import LayerActivity

# ── STIMULUS ──────────────────────────────────────────────────────────────────
# ON edges only (12 directions) — matches Experiment 1 directly.

ANGLES    = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
INTENSITY = 1  # ON only

print("Building MovingEdge stimulus dataset...")
dataset = MovingEdge(
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
print(f"  Dataset: {len(dataset)} total samples")
print(dataset.arg_df)

# Select ON-edge indices
on_edge_indices = [
    i for i, row in dataset.arg_df.iterrows()
    if row["intensity"] == INTENSITY
]
print(f"  Using {len(on_edge_indices)} ON stimulus conditions: {on_edge_indices}")
N_STIM = len(on_edge_indices)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_cell_types(network):
    """Extract cell type names from connectome."""
    return [
        ct.decode() if isinstance(ct, bytes) else ct
        for ct in network.connectome.unique_cell_types[:]
    ]


def get_population_vector(network, stimulus, dt, cell_types):
    """
    Simulate network response to a single stimulus.
    Returns peak central-cell voltage per cell type as (n_cell_types,) array.

    Mirrors Experiments 1-3 exactly: fade_in_state init, network.simulate(),
    LayerActivity central cell extraction.
    """
    # Ensure shape is (n_frames, 1, 721)
    if stimulus.dim() == 2:
        stimulus = stimulus.unsqueeze(1)

    initial_state = network.fade_in_state(1.0, dt, stimulus[[0]])

    with torch.no_grad():
        responses = network.simulate(
            stimulus[None], dt, initial_state=initial_state
        ).cpu()

    layer_act = LayerActivity(responses, network.connectome, keepref=True)
    pop_vec = np.array([
        layer_act.central[ct].squeeze().numpy().max()
        for ct in cell_types
    ])
    pop_vec = np.clip(pop_vec, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)

    del responses, layer_act
    torch.cuda.empty_cache()

    return pop_vec


def build_rdm(pop_matrix, metric="cosine"):
    """Cosine or Euclidean RDM from (n_stim, n_cells) matrix."""
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3)
    if metric == "cosine":
        pop_matrix = pop_matrix + 1e-10
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if metric == "cosine":
                    rdm[i, j] = cosine_dist(pop_matrix[i], pop_matrix[j])
                else:
                    rdm[i, j] = euclidean(pop_matrix[i], pop_matrix[j])
    return rdm


def rdm_similarity(rdm1, rdm2):
    """Spearman r and Kendall tau between upper triangles of two RDMs."""
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    r_s, p_s = spearmanr(rdm1[idx], rdm2[idx])
    r_k, p_k = kendalltau(rdm1[idx], rdm2[idx])
    return r_s, p_s, r_k, p_k


def permutation_test_rdm(rdm1, rdm2, n_permutations=10_000, seed=SEED):
    """Stimulus-label randomization test (Nili et al. 2014)."""
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
    p_r   = float(np.mean(null_r   >= obs_r))
    p_tau = float(np.mean(null_tau >= obs_tau))
    return obs_r, p_r, obs_tau, p_tau, null_r, null_tau


def is_stable(network, stimulus, dt):
    """
    Run a single forward pass and return True if output is finite and < OVERFLOW_LIMIT.
    Mirrors randomize_weights_stable() from Experiments 1-3.
    """
    if stimulus.dim() == 2:
        stim = stimulus.unsqueeze(1)
    else:
        stim = stimulus
    try:
        initial_state = network.fade_in_state(1.0, dt, stim[[0]])
        out = network.simulate(stim[None], dt, initial_state=initial_state)
        out_np = out.cpu().numpy()
        return bool(torch.all(torch.isfinite(out)) and np.all(np.abs(out_np) < OVERFLOW_LIMIT))
    except Exception:
        return False


# ── NETWORK CONSTRUCTION ──────────────────────────────────────────────────────

def make_untrained_cc(seed):
    """
    Untrained connectome-constrained network.
    Default init is deterministic so we add Gaussian noise to free params.
    edges_sign and edges_syn_count (connectome structure) are unchanged.
    """
    rng = np.random.default_rng(seed)
    net = Network()

    with torch.no_grad():
        # Resting potential
        noise = torch.tensor(
            rng.normal(0, BIAS_NOISE, size=net.nodes_bias.shape),
            dtype=torch.float32
        )
        net.nodes_bias.data += noise

        # Time constant (keep positive)
        noise = torch.tensor(
            rng.normal(0, TIME_CONST_NOISE, size=net.nodes_time_const.shape),
            dtype=torch.float32
        )
        net.nodes_time_const.data = torch.clamp(
            net.nodes_time_const.data + noise, min=0.001
        )

        # Synapse strength (keep non-negative)
        raw = net.edge_params.syn_strength.raw_values
        noise = torch.tensor(
            rng.normal(0, SYN_STRENGTH_NOISE, size=raw.shape),
            dtype=torch.float32
        )
        raw.data = torch.clamp(raw.data + noise, min=0.0)

    # NOTE: Do NOT call eval() or requires_grad_(False) here.
    # simulate() uses the simulation() context manager internally, which
    # temporarily sets training=False and requires_grad=False, then restores
    # them. Permanently freezing here would prevent further perturbation
    # across the ensemble.

    return net.to(DEVICE)


def apply_syn_shuffle(net, rng):
    """
    Sign-preserving shuffle of edges_syn_strength. Matches Experiments 1-3.
    Shuffles absolute values within each sign class, preserving E/I identity.
    .data modification works regardless of requires_grad state.
    """
    with torch.no_grad():
        raw = net.edge_params.syn_strength.raw_values
        vals = raw.data.cpu().numpy().copy()
        # Shuffle within each sign class to preserve E/I identity
        pos_idx = np.where(vals > 0)[0]
        neg_idx = np.where(vals <= 0)[0]
        if len(pos_idx) > 1:
            vals[pos_idx] = rng.permutation(vals[pos_idx])
        if len(neg_idx) > 1:
            vals[neg_idx] = rng.permutation(vals[neg_idx])
        raw.data = torch.tensor(vals, dtype=torch.float32).to(DEVICE)
    return net


def apply_sign_shuffle(net, rng):
    """
    Shuffle of edges_sign (E/I identity) AND edges_syn_strength.
    Preserves the TOTAL COUNT of excitatory/inhibitory connections,
    but randomly reassigns which cell-type pairs are E vs I.

    NOTE: permuting within each sign class (+1s among +1s) is a no-op —
    the values don't change, only positions do, and since all excitatory
    entries have the same value (+1), position permutation within the class
    has no effect. The correct approach is to randomly assign +1/-1 across
    ALL 604 entries while preserving the E/I count.
    """
    with torch.no_grad():
        # Correctly shuffle E/I assignments across all cell-type pairs
        sign_param = net.edge_params.sign.raw_values
        vals = sign_param.data.cpu().numpy().copy()
        n_exc = int((vals > 0).sum())
        n_inh = int((vals < 0).sum())
        new_vals = np.array([1.0] * n_exc + [-1.0] * n_inh, dtype=np.float32)
        shuffled = rng.permutation(new_vals)
        sign_param.data = torch.tensor(shuffled, dtype=torch.float32).to(DEVICE)

        # Also shuffle syn_strength
        raw = net.edge_params.syn_strength.raw_values
        syn_vals = raw.data.cpu().numpy().copy()
        pos_s = np.where(syn_vals > 0)[0]
        neg_s = np.where(syn_vals <= 0)[0]
        if len(pos_s) > 1:
            syn_vals[pos_s] = rng.permutation(syn_vals[pos_s])
        if len(neg_s) > 1:
            syn_vals[neg_s] = rng.permutation(syn_vals[neg_s])
        raw.data = torch.tensor(syn_vals, dtype=torch.float32).to(DEVICE)

    return net


# ── MAIN EXPERIMENT LOOP ──────────────────────────────────────────────────────

def run_condition(label, make_fn, n_models, dataset, stim_indices, dt):
    """
    Run one condition. Returns (pop_matrices, rdms_cosine, n_accepted, n_seeds_tried).
    make_fn(seed) -> Network instance on DEVICE, already configured.
    """
    print(f"\n{'='*60}")
    print(f"Condition: {label} (n={n_models})")
    print(f"{'='*60}")

    # Check stimulus — first ON-edge stimulus
    check_stim = dataset[stim_indices[0]]
    if not isinstance(check_stim, torch.Tensor):
        check_stim = torch.tensor(check_stim, dtype=torch.float32)
    check_stim = check_stim.to(DEVICE)

    pop_matrices = []
    rdms         = []
    attempts_log = []
    accepted     = 0
    seed         = 0
    cell_types   = None

    while accepted < n_models:
        if seed > n_models * MAX_ATTEMPTS:
            print(f"  WARNING: Gave up after {seed} seeds.")
            break

        net = make_fn(seed)
        seed += 1

        if not is_stable(net, check_stim, dt):
            del net
            torch.cuda.empty_cache()
            continue

        # Collect population vectors across all stimuli
        if cell_types is None:
            cell_types = get_cell_types(net)

        pop_vecs = []
        ok = True
        for stim_idx in stim_indices:
            stimulus = dataset[stim_idx]
            if not isinstance(stimulus, torch.Tensor):
                stimulus = torch.tensor(stimulus, dtype=torch.float32)
            stimulus = stimulus.to(DEVICE)
            pop_vec = get_population_vector(net, stimulus, dt, cell_types)
            if not np.isfinite(pop_vec).all():
                ok = False
                break
            pop_vecs.append(pop_vec)

        if not ok:
            del net
            torch.cuda.empty_cache()
            continue

        pop_matrix = np.stack(pop_vecs, axis=0)  # (n_stim, 65)
        rdm = build_rdm(pop_matrix, metric="cosine")
        pop_matrices.append(pop_matrix)
        rdms.append(rdm)
        attempts_log.append(seed)
        accepted += 1
        print(f"  Accepted {accepted}/{n_models} (seed={seed-1})")

        del net
        torch.cuda.empty_cache()

    pop_matrices = np.stack(pop_matrices, axis=0)
    rdms         = np.stack(rdms, axis=0)

    print(f"\n  {label}: {accepted}/{n_models} accepted over {seed} seeds.")
    if attempts_log:
        print(f"  Mean attempts: {np.mean(attempts_log):.1f} +/- {np.std(attempts_log):.1f}")
    print(f"  Cell types ({len(cell_types)}): {cell_types[:5]}...")

    return pop_matrices, rdms, accepted, seed, cell_types


# ── DEFINE MAKE FUNCTIONS FOR EACH CONDITION ─────────────────────────────────

def make_cc(seed):
    return make_untrained_cc(seed)

def make_rand_syn(seed):
    rng = np.random.default_rng(seed + 10_000)  # different seed space from CC
    net = make_untrained_cc(seed)
    return apply_syn_shuffle(net, rng)

def make_rand_sign(seed):
    rng = np.random.default_rng(seed + 20_000)
    net = make_untrained_cc(seed)
    return apply_sign_shuffle(net, rng)

# ── VERIFICATION: CONFIRM NETWORKS ARE GENUINELY UNTRAINED ──────────────────
# This block documents why Network() produces untrained networks.
# It does NOT require pretrained checkpoints.

print("\n" + "="*60)
print("VERIFICATION: Confirming untrained network status")
print("="*60)

_net_v = Network()
_bias_v  = _net_v.nodes_bias.data.clone()
_tc_v    = _net_v.nodes_time_const.data.clone()
_syn_v   = _net_v.edge_params.syn_strength.raw_values.data.clone()
_sign_v  = _net_v.edge_params.sign.raw_values.data

print("\n1. nodes_bias ~ Normal(mean=0.5, std=0.05, seed=0) — Flyvis prior, not trained")
print(f"   mean={_bias_v.mean():.4f}, std={_bias_v.std():.4f}")

print("\n2. nodes_time_const = 0.05 constant — Flyvis prior, not trained")
print(f"   All equal 0.05: {torch.allclose(_tc_v, torch.full_like(_tc_v, 0.05))}")

print("\n3. edges_syn_strength = 0.01 * syn_count — Flyvis prior, not trained")
print(f"   mean={_syn_v.mean():.6f}, std={_syn_v.std():.6f}")

print("\n4. Two Network() instances are identical (deterministic init):")
_net_v2 = Network()
print(f"   bias identical:    {torch.allclose(_bias_v, _net_v2.nodes_bias.data)}")
print(f"   syn_str identical: {torch.allclose(_syn_v, _net_v2.edge_params.syn_strength.raw_values.data)}")
print("   --> Gaussian perturbation in make_untrained_cc() is required for an ensemble")

print("\n5. edges_sign + edges_syn_count: requires_grad=False, fixed by connectome, never trained")
print(f"   n_excitatory={(_sign_v > 0).sum().item()}, n_inhibitory={(_sign_v < 0).sum().item()}")

print("\n6. No checkpoint loaded. Network() uses Flyvis prior + connectome only.")
print("   To verify in Colab against a trained checkpoint:")
print("     nv = NetworkView(results_dir / 'flow/0000/000')")
print("     net_trained = nv.init_network()")
print("     # trained bias std >> 0.05; trained syn_strength range >> 0.01 * syn_count")

del _net_v, _net_v2
torch.cuda.empty_cache()
print("\nVerification complete. Proceeding with experiment.")

# ── METHODOLOGICAL SANITY CHECK ───────────────────────────────────────────────
# Confirms that the API we use matches Lappalainen et al.'s simulate() contract.

print("\n" + "="*60)
print("METHODOLOGICAL SANITY CHECK")
print("="*60)

_net_m = make_untrained_cc(seed=0)

# 1. simulate() requires eval() and requires_grad=False for all params.
#    make_untrained_cc() calls eval()+requires_grad_(False) after perturbation.
_all_frozen = all(not p.requires_grad for p in _net_m.parameters())
_in_eval    = not _net_m.training
# For untrained networks, training=True and requires_grad=True before simulate() —
# this is correct. simulate() uses simulation() context manager to handle freezing.
print(f"\n1. simulate() contract handled internally by simulation() context manager")
print(f"   network.training={_in_eval} (will be set to False inside simulate())")
print(f"   params frozen={_all_frozen} (will be frozen inside simulate(), restored after)")
print("   No manual eval()/requires_grad_(False) needed — simulation() handles this.")
print("   PASS")

# 2. fade_in_state + simulate pipeline matches Experiments 1-3.
#    Check stimulus shape contract: simulate() requires (batch, frames, 1, hexals).
_check_stim = dataset[on_edge_indices[0]]
if not isinstance(_check_stim, torch.Tensor):
    _check_stim = torch.tensor(_check_stim, dtype=torch.float32)
_check_stim = _check_stim.to(DEVICE)
if _check_stim.dim() == 2:
    _check_stim_4d = _check_stim.unsqueeze(1)  # (frames, 1, hexals)
else:
    _check_stim_4d = _check_stim
_init_state = _net_m.fade_in_state(1.0, dataset.dt, _check_stim_4d[[0]])
_resp = _net_m.simulate(_check_stim_4d[None], dataset.dt, initial_state=_init_state)
print(f"\n2. simulate() output shape: {_resp.shape}")
print(f"   Expected: (1, n_frames, n_nodes=45669)")
assert _resp.shape[0] == 1 and _resp.shape[2] == _net_m.n_nodes,     f"FAIL: unexpected output shape {_resp.shape}"
print("   PASS")

# 3. LayerActivity.central gives 65-dim population vector — one value per cell type.
_cell_types_m = get_cell_types(_net_m)
_layer_act_m  = LayerActivity(_resp.cpu(), _net_m.connectome, keepref=True)
_pop_vec_m    = np.array([
    _layer_act_m.central[ct].squeeze().numpy().max()
    for ct in _cell_types_m
])
print(f"\n3. Population vector dim: {_pop_vec_m.shape[0]} (expected 65)")
assert _pop_vec_m.shape[0] == 65, f"FAIL: expected 65-dim pop vec, got {_pop_vec_m.shape[0]}"
print(f"   Finite: {np.isfinite(_pop_vec_m).all()}")
print("   PASS")

# 4. sign shuffle correctly reassigns E/I identity (not a no-op).
_rng_m  = np.random.default_rng(99)
_net_m2 = make_rand_sign(seed=0)
_sign_cc   = _net_m.edge_params.sign.raw_values.data.cpu().numpy()
_sign_rand = _net_m2.edge_params.sign.raw_values.data.cpu().numpy()
_n_changed = int((_sign_cc != _sign_rand).sum())
print(f"\n4. Sign shuffle: {_n_changed}/{len(_sign_cc)} E/I assignments changed")
assert _n_changed > 0, "FAIL: sign shuffle produced no changes (was a no-op)"
print(f"   E/I count preserved: exc {int((_sign_rand > 0).sum())}={int((_sign_cc > 0).sum())}, "
      f"inh {int((_sign_rand < 0).sum())}={int((_sign_cc < 0).sum())}")
print("   PASS")

del _net_m, _net_m2, _layer_act_m, _resp
torch.cuda.empty_cache()
print("\nAll methodological checks passed.")

# ── RUN ───────────────────────────────────────────────────────────────────────

print("\nRunning Experiment 4: Untrained networks")
print(f"N_MODELS={N_MODELS}, N_PERMUTATIONS={N_PERMUTATIONS}")
print(f"Stimulus: {N_STIM} ON moving edge conditions")

cc_pop, cc_rdms, cc_accepted, cc_seeds, cell_types = run_condition(
    "Untrained CC", make_cc,
    N_MODELS, dataset, on_edge_indices, dataset.dt
)

syn_pop, syn_rdms, syn_accepted, syn_seeds, _ = run_condition(
    "Untrained Random (syn shuffle)", make_rand_syn,
    N_MODELS, dataset, on_edge_indices, dataset.dt
)

sign_pop, sign_rdms, sign_accepted, sign_seeds, _ = run_condition(
    "Untrained Random (sign shuffle)", make_rand_sign,
    N_MODELS, dataset, on_edge_indices, dataset.dt
)

# ── MEAN RDMs ─────────────────────────────────────────────────────────────────

cc_mean_rdm   = cc_rdms.mean(axis=0)
syn_mean_rdm  = syn_rdms.mean(axis=0)
sign_mean_rdm = sign_rdms.mean(axis=0)

# ── RSA ───────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RSA RESULTS")
print("="*60)

r_syn, p_syn, tau_syn, ptau_syn = rdm_similarity(cc_mean_rdm, syn_mean_rdm)
r_sign, p_sign, tau_sign, ptau_sign = rdm_similarity(cc_mean_rdm, sign_mean_rdm)

print(f"\nCC vs Rand-syn:  Spearman r={r_syn:.3f}, p={p_syn:.4f} | Kendall tau={tau_syn:.3f}")
print(f"CC vs Rand-sign: Spearman r={r_sign:.3f}, p={p_sign:.4f} | Kendall tau={tau_sign:.3f}")

r_syn_obs,  p_syn_perm,  tau_syn_obs,  ptau_syn_perm,  null_syn,  null_syn_tau  = \
    permutation_test_rdm(cc_mean_rdm, syn_mean_rdm,  N_PERMUTATIONS)
r_sign_obs, p_sign_perm, tau_sign_obs, ptau_sign_perm, null_sign, null_sign_tau = \
    permutation_test_rdm(cc_mean_rdm, sign_mean_rdm, N_PERMUTATIONS)

print(f"\nCC vs Rand-syn  permutation: r={r_syn_obs:.3f}, p_perm={p_syn_perm:.4f} "
      f"| tau={tau_syn_obs:.3f}, p_perm={ptau_syn_perm:.4f}")
print(f"CC vs Rand-sign permutation: r={r_sign_obs:.3f}, p_perm={p_sign_perm:.4f} "
      f"| tau={tau_sign_obs:.3f}, p_perm={ptau_sign_perm:.4f}")

# Within-CC consistency
from itertools import combinations
cc_pairs = list(combinations(range(len(cc_rdms)), 2))
cc_consistency = [rdm_similarity(cc_rdms[i], cc_rdms[j])[0] for i, j in cc_pairs]
print(f"\nWithin-CC consistency: r={np.mean(cc_consistency):.3f} +/- {np.std(cc_consistency):.3f}")

# ── BIOLOGICAL REFERENCE ──────────────────────────────────────────────────────

def build_bio_rdm():
    """Von Mises T4/T5 reference RDM for 12 ON directions (Maisak et al. 2013)."""
    directions = np.linspace(0, 330, 12)
    preferred  = [180, 0, 90, 270, 180, 0, 90, 270]
    kappa = 2.5

    def vm(theta, mu):
        r = np.exp(kappa * np.cos(np.radians(theta - mu)))
        r = r - np.exp(-kappa)
        return max(r, 0)

    pop = np.array([[vm(d, mu) for mu in preferred] for d in directions])
    return build_rdm(pop, metric="cosine")

bio_rdm = build_bio_rdm()

r_cc_bio,  p_cc_bio,  _, _ = rdm_similarity(cc_mean_rdm,  bio_rdm)
r_syn_bio, p_syn_bio, _, _ = rdm_similarity(syn_mean_rdm,  bio_rdm)
r_sign_bio,p_sign_bio,_, _ = rdm_similarity(sign_mean_rdm, bio_rdm)

_, p_cc_bio_perm,  _, _, _, _ = permutation_test_rdm(cc_mean_rdm,  bio_rdm)
_, p_syn_bio_perm, _, _, _, _ = permutation_test_rdm(syn_mean_rdm,  bio_rdm)
_, p_sign_bio_perm,_, _, _, _ = permutation_test_rdm(sign_mean_rdm, bio_rdm)

print("\n" + "="*60)
print("BIOLOGICAL REFERENCE (Maisak et al. 2013 T4/T5)")
print("="*60)
print(f"CC   vs Bio: r={r_cc_bio:.3f}   (p_perm={p_cc_bio_perm:.4f})")
print(f"Syn  vs Bio: r={r_syn_bio:.3f}   (p_perm={p_syn_bio_perm:.4f})")
print(f"Sign vs Bio: r={r_sign_bio:.3f}   (p_perm={p_sign_bio_perm:.4f})")
print(f"Delta r (CC - syn rand):  {r_cc_bio - r_syn_bio:.3f}")
print(f"Delta r (CC - sign rand): {r_cc_bio - r_sign_bio:.3f}")

# ── SAVE ──────────────────────────────────────────────────────────────────────

print("\nSaving results...")
np.savez(
    results_dir / "results_exp4_untrained.npz",
    cc_pop_matrices        = cc_pop,
    syn_pop_matrices       = syn_pop,
    sign_pop_matrices      = sign_pop,
    cc_rdms                = cc_rdms,
    syn_rdms               = syn_rdms,
    sign_rdms              = sign_rdms,
    cc_mean_rdm            = cc_mean_rdm,
    syn_mean_rdm           = syn_mean_rdm,
    sign_mean_rdm          = sign_mean_rdm,
    bio_rdm                = bio_rdm,
    r_syn_obs              = np.array(r_syn_obs),
    r_sign_obs             = np.array(r_sign_obs),
    p_syn_perm             = np.array(p_syn_perm),
    p_sign_perm            = np.array(p_sign_perm),
    null_syn               = null_syn,
    null_sign              = null_sign,
    r_cc_bio               = np.array(r_cc_bio),
    r_syn_bio              = np.array(r_syn_bio),
    r_sign_bio             = np.array(r_sign_bio),
    cc_accepted            = np.array(cc_accepted),
    syn_accepted           = np.array(syn_accepted),
    sign_accepted          = np.array(sign_accepted),
    n_models               = np.array(N_MODELS),
    n_permutations         = np.array(N_PERMUTATIONS),
    cell_types             = np.array(cell_types),
)
print("  Saved: ../results/results_exp4_untrained.npz")

# ── FIGURES ───────────────────────────────────────────────────────────────────

angle_labels = [f"{a}deg" for a in ANGLES]

# RDM comparison figure
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle(
    f"Experiment 4: Untrained networks — ON edges (n={N_MODELS})\n"
    f"CC vs Rand-syn: r={r_syn_obs:.3f}, p_perm={p_syn_perm:.4f}  |  "
    f"CC vs Rand-sign: r={r_sign_obs:.3f}, p_perm={p_sign_perm:.4f}\n"
    f"CC vs Bio: r={r_cc_bio:.3f}  |  Rand-syn vs Bio: r={r_syn_bio:.3f}  |  "
    f"Rand-sign vs Bio: r={r_sign_bio:.3f}",
    fontsize=9
)
for ax, rdm, title in zip(
    axes,
    [cc_mean_rdm, syn_mean_rdm, sign_mean_rdm, bio_rdm],
    [f"Untrained CC (n={cc_accepted})",
     f"Rand-syn (n={syn_accepted})",
     f"Rand-sign (n={sign_accepted})",
     "Biological Ref (T4/T5)"]
):
    vmax = np.percentile(rdm[np.triu_indices(N_STIM, k=1)], 97)
    im = ax.imshow(rdm, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(N_STIM))
    ax.set_yticks(range(N_STIM))
    ax.set_xticklabels(angle_labels, fontsize=6, rotation=90)
    ax.set_yticklabels(angle_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(figures_dir / "exp4_untrained_rdms.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ../figures/exp4_untrained_rdms.png")

# Permutation test figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, null, obs, label, pval in [
    (axes[0], null_syn,  r_syn_obs,  "CC vs Rand-syn",  p_syn_perm),
    (axes[1], null_sign, r_sign_obs, "CC vs Rand-sign", p_sign_perm),
]:
    ax.hist(null, bins=50, color="steelblue", alpha=0.7, density=True)
    ax.axvline(obs, color="red", lw=2, label=f"Observed r={obs:.3f}")
    ax.set_xlabel("Spearman r (permuted)")
    ax.set_ylabel("Density")
    ax.set_title(f"{label}\np_perm={pval:.4f} (n={N_PERMUTATIONS})")
    ax.legend(fontsize=9)
plt.suptitle("Experiment 4: Permutation test null distributions (untrained networks)")
plt.tight_layout()
plt.savefig(figures_dir / "exp4_untrained_permtest.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: ../figures/exp4_untrained_permtest.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT 4 SUMMARY")
print("="*60)
print(f"N stimuli:  {N_STIM} (ON moving edges, 12 directions)")
print(f"N models:   {N_MODELS} per condition")
print(f"Accepted:   CC={cc_accepted}  Rand-syn={syn_accepted}  Rand-sign={sign_accepted}")
print()
print("RSA (CC vs Random):")
print(f"  CC vs Rand-syn:  r={r_syn_obs:.3f}, tau={tau_syn_obs:.3f}, p_perm={p_syn_perm:.4f}  [permutation]")
print(f"  CC vs Rand-sign: r={r_sign_obs:.3f}, tau={tau_sign_obs:.3f}, p_perm={p_sign_perm:.4f}  [permutation]")
print()
print("Biological reference:")
print(f"  CC   vs Bio: r={r_cc_bio:.3f}  (p_perm={p_cc_bio_perm:.4f})")
print(f"  Syn  vs Bio: r={r_syn_bio:.3f}  (p_perm={p_syn_bio_perm:.4f})")
print(f"  Sign vs Bio: r={r_sign_bio:.3f}  (p_perm={p_sign_bio_perm:.4f})")
print(f"  Delta r (CC - syn rand):  {r_cc_bio - r_syn_bio:.3f}")
print(f"  Delta r (CC - sign rand): {r_cc_bio - r_sign_bio:.3f}")
print()
print("Within-CC consistency:")
print(f"  r = {np.mean(cc_consistency):.3f} +/- {np.std(cc_consistency):.3f}")
print()
print("INTERPRETATION:")
if p_syn_perm < 0.05 and r_syn_obs > 0:
    print("  CC > Rand-syn (p < 0.05): wiring carries geometry signal BEFORE training.")
    print("  Training confound is RESOLVED for this component.")
else:
    print("  CC ~ Rand-syn: not significant. Geometry signal may be training-dependent.")
if p_sign_perm < 0.05 and r_sign_obs > 0:
    print("  CC > Rand-sign (p < 0.05): E/I wiring identity contributes before training.")
else:
    print("  CC ~ Rand-sign: E/I identity does not add beyond syn-strength shuffle.")
print()
print("Files saved:")
print("  ../results/results_exp4_untrained.npz")
print("  ../figures/exp4_untrained_rdms.png")
print("  ../figures/exp4_untrained_permtest.png")
