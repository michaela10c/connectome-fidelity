"""
exp4_perturbation_sweep.py — at what perturbation scale does untrained
connectome geometry become measurable, and does it distinguish CC from shuffled?

WHY THIS EXPERIMENT
-------------------
The corrected Experiment 4 aborted at the precision guard. At the default
perturbation (BIAS_NOISE = 0.05), the untrained CC mean RDM has an off-diagonal
span of 1.66e-08 against a float32 round-off floor of 1.93e-07 — the floor is
eleven times the signal (span/floor = 0.09x). Its rank order, and therefore
every statistic derived from it, is rounding.

A cancellation-free cross-check confirms this is not a metric artifact: Kendall
tau between the cosine and euclidean-normalized rank orders is 1.0000 for every
one of 150 individual models. Both metrics see the same thing. What they see is
noise.

That result is not a failure. It says untrained networks at this perturbation
scale produce population responses so nearly identical across the twelve
directions that the distances between them are unresolvable. The question it
raises is: does directional geometry appear at ANY perturbation scale, and if
so, is it a property of the connectome or merely of large perturbations?

WHAT THIS SCRIPT MEASURES
-------------------------
For each BIAS_NOISE value in a sweep, and for all three conditions
(untrained CC, syn-shuffled, sign-shuffled):

  1. RESOLVABILITY. The RDM's off-diagonal span against the float32 round-off
     floor implied by the response magnitude. Below ~10x, no statistic is
     meaningful. This is the gate: nothing else is reported until it clears.

  2. CIRCULARITY. Spearman r between the mean RDM and the circular-distance
     reference min(|i-j|, 12-|i-j|). This is the direct question — does the
     network order directions by angle? — asked against an explicit reference
     rather than through the degenerate von Mises proxy.

  3. THE CONTROL THAT MAKES IT INTERPRETABLE. Whether CC gains circular
     structure that the shuffled conditions do NOT. If all three become
     resolvable and all three become circular, the sweep has demonstrated that
     large perturbations produce large responses, not that wiring carries a
     geometric prior. The CC-minus-shuffle difference in circularity is the
     quantity of interest.

  4. REGIME DIAGNOSTICS. Response magnitude, stability rejections, and
     metric-agreement (cosine vs euclidean-normalized). Raising BIAS_NOISE
     moves the network away from the Flyvis prior; at some point "untrained CC"
     stops meaning "the connectome at initialization" and starts meaning "the
     connectome with large random biases." The script reports the numbers needed
     to judge where that line is; it does not pretend to know.

WHAT THE OUTCOMES MEAN
----------------------
  * Guard never clears within a plausible range
      -> Untrained networks have no measurable representational geometry.
         Training CREATES the directional structure rather than amplifying a
         wiring prior. This inverts the original Experiment 4 claim and is a
         stronger, cleaner result.

  * Guard clears, and CC circularity exceeds both shuffles
      -> The connectome prior is real but requires perturbation to be visible.
         Report the threshold scale and the CC-vs-shuffle gap at that scale.

  * Guard clears, and CC circularity matches the shuffles
      -> Perturbation magnitude, not wiring, produced the geometry. No prior.

Only the first two support any claim about the connectome. The third is the null
that the original experiment could not have detected, because it never checked
whether its RDMs were resolvable.

USAGE
-----
    python exp4_perturbation_sweep.py                     # default sweep
    python exp4_perturbation_sweep.py --n_models 10       # fast pilot
    python exp4_perturbation_sweep.py --noise 0.05 0.1 0.2 0.4 0.8

Runtime: roughly (n_models x 13 forward passes x ~2.6 s) per condition per noise
level. The default (n_models=20, 6 noise levels, 3 conditions) is ~3.5 hours on
a T4. Use --n_models 5 for a first look; the guard verdict is usually obvious.

Writes: results/exp4_sweep.npz, figures/exp4_sweep.png
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)

OVERFLOW_LIMIT = 1e6
MAX_ATTEMPTS   = 20
FLOAT32_EPS    = float(np.finfo(np.float32).eps)   # 1.19e-07
RESOLVE_MARGIN = 10.0   # heuristic; see note in untrained_networks_corrected.py

# Time-constant and synapse-strength noise are held FIXED across the sweep.
# Only the bias noise is varied, so the sweep has one axis and the result can be
# attributed to it. Varying all three together would confound the interpretation.
TIME_CONST_NOISE   = 0.005
SYN_STRENGTH_NOISE = 0.002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
N_DIR  = len(ANGLES)

results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
figures_dir = Path("figures"); figures_dir.mkdir(exist_ok=True)

import flyvis
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.network import Network
from flyvis.utils.activity_utils import LayerActivity


# ── distances (float64, cancellation-free) ───────────────────────────────────
def cosine_distance_stable(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return np.nan
    d = a / na - b / nb
    return float(d @ d) / 2.0


def euclidean_normalized(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return np.nan
    return float(np.linalg.norm(a / na - b / nb))


def build_rdm(P, metric="cosine"):
    P = np.asarray(P, dtype=np.float64)
    if not np.isfinite(P).all():
        raise ValueError("non-finite population vector")
    fn = cosine_distance_stable if metric == "cosine" else euclidean_normalized
    n = P.shape[0]
    R = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = fn(P[i], P[j])
            if np.isnan(d):
                raise ValueError(f"zero-norm population vector at stimulus {i} or {j}")
            R[i, j] = R[j, i] = d
    return R


def upper(M):
    return M[np.triu_indices_from(M, k=1)]


def circular_reference(n=N_DIR):
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    d = np.abs(i - j)
    return np.minimum(d, n - d).astype(float)


def resolvability(rdm, pop_matrix):
    """Return (span, floor, ratio, is_resolvable). The gate."""
    off = upper(rdm)
    span = float(off.max() - off.min())
    floor = FLOAT32_EPS * float(np.abs(pop_matrix).max())
    ratio = span / floor if floor > 0 else np.inf
    return span, floor, ratio, ratio >= RESOLVE_MARGIN


# ── network construction (bias noise is the swept parameter) ─────────────────
def make_untrained_cc(seed, bias_noise):
    rng = np.random.default_rng(seed)
    net = Network()
    with torch.no_grad():
        net.nodes_bias.data += torch.tensor(
            rng.normal(0, bias_noise, size=net.nodes_bias.shape), dtype=torch.float32)
        net.nodes_time_const.data = torch.clamp(
            net.nodes_time_const.data + torch.tensor(
                rng.normal(0, TIME_CONST_NOISE, size=net.nodes_time_const.shape),
                dtype=torch.float32), min=0.001)
        raw = net.edge_params.syn_strength.raw_values
        raw.data = torch.clamp(raw.data + torch.tensor(
            rng.normal(0, SYN_STRENGTH_NOISE, size=raw.shape),
            dtype=torch.float32), min=0.0)
    return net.to(DEVICE)


def apply_syn_shuffle(net, rng):
    with torch.no_grad():
        raw = net.edge_params.syn_strength.raw_values
        v = raw.data.cpu().numpy().copy()
        pos, neg = np.where(v > 0)[0], np.where(v <= 0)[0]
        if len(pos) > 1: v[pos] = rng.permutation(v[pos])
        if len(neg) > 1: v[neg] = rng.permutation(v[neg])
        raw.data = torch.tensor(v, dtype=torch.float32).to(DEVICE)
    return net


def apply_sign_shuffle(net, rng):
    with torch.no_grad():
        sp = net.edge_params.sign.raw_values
        v = sp.data.cpu().numpy().copy()
        n_exc, n_inh = int((v > 0).sum()), int((v < 0).sum())
        sp.data = torch.tensor(
            rng.permutation(np.array([1.0]*n_exc + [-1.0]*n_inh, dtype=np.float32)),
            dtype=torch.float32).to(DEVICE)
        raw = net.edge_params.syn_strength.raw_values
        s = raw.data.cpu().numpy().copy()
        pos, neg = np.where(s > 0)[0], np.where(s <= 0)[0]
        if len(pos) > 1: s[pos] = rng.permutation(s[pos])
        if len(neg) > 1: s[neg] = rng.permutation(s[neg])
        raw.data = torch.tensor(s, dtype=torch.float32).to(DEVICE)
    return net


# ── simulation ───────────────────────────────────────────────────────────────
def get_cell_types(net):
    return [ct.decode() if isinstance(ct, bytes) else ct
            for ct in net.connectome.unique_cell_types[:]]


def get_population_vector(net, stim, dt, cell_types):
    if stim.dim() == 2:
        stim = stim.unsqueeze(1)
    s0 = net.fade_in_state(1.0, dt, stim[[0]])
    with torch.no_grad():
        resp = net.simulate(stim[None], dt, initial_state=s0).cpu()
    la = LayerActivity(resp, net.connectome, keepref=True)
    pv = np.array([la.central[ct].squeeze().numpy().max() for ct in cell_types],
                  dtype=np.float64)
    pv = np.clip(pv, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
    del resp, la
    torch.cuda.empty_cache()
    return pv


def is_stable(net, stim, dt):
    st = stim.unsqueeze(1) if stim.dim() == 2 else stim
    s0 = net.fade_in_state(1.0, dt, st[[0]])
    out = net.simulate(st[None], dt, initial_state=s0)
    o = out.cpu().numpy()
    return bool(torch.all(torch.isfinite(out)) and np.all(np.abs(o) < OVERFLOW_LIMIT))


def run_condition(make_fn, n_models, dataset, stim_idx, dt):
    """Return (pop_matrices, rdms_cos, rdms_euc, n_rejected)."""
    check = dataset[stim_idx[0]]
    if not isinstance(check, torch.Tensor):
        check = torch.tensor(check, dtype=torch.float32)
    check = check.to(DEVICE)

    pops, rc, re_ = [], [], []
    accepted = seed = rejected = 0
    cts = None

    while accepted < n_models and seed < n_models * MAX_ATTEMPTS:
        net = make_fn(seed); seed += 1
        if not is_stable(net, check, dt):
            rejected += 1; del net; torch.cuda.empty_cache(); continue
        if cts is None:
            cts = get_cell_types(net)
        pv_list, ok = [], True
        for si in stim_idx:
            s = dataset[si]
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.float32)
            pv = get_population_vector(net, s.to(DEVICE), dt, cts)
            if not np.isfinite(pv).all():
                ok = False; break
            pv_list.append(pv)
        if not ok:
            rejected += 1; del net; torch.cuda.empty_cache(); continue

        pm = np.stack(pv_list, 0)
        pops.append(pm)
        rc.append(build_rdm(pm, "cosine"))
        re_.append(build_rdm(pm, "euclidean_normalized"))
        accepted += 1
        del net; torch.cuda.empty_cache()

    return np.stack(pops, 0), np.stack(rc, 0), np.stack(re_, 0), rejected


# ── the sweep ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
                    help="BIAS_NOISE values to sweep (default starts at the "
                         "original 0.05, which is known to fail the guard)")
    ap.add_argument("--n_models", type=int, default=20)
    args = ap.parse_args()

    dataset = MovingEdge(offsets=[-10, 11], intensities=[0, 1], speeds=[19],
                         height=80, post_pad_mode="continue",
                         t_pre=1.0, t_post=1.0, dt=1/200, angles=ANGLES)
    on_idx = [i for i, r in dataset.arg_df.iterrows() if r["intensity"] == 1]
    assert len(on_idx) == N_DIR

    circ = circular_reference()
    rows = []

    print(f"Device: {DEVICE}   n_models={args.n_models}   "
          f"noise levels: {args.noise}\n")
    print("Only BIAS_NOISE varies. TIME_CONST_NOISE and SYN_STRENGTH_NOISE are")
    print("held fixed, so any change is attributable to the bias perturbation.\n")

    for bn in args.noise:
        print("=" * 78)
        print(f"BIAS_NOISE = {bn}")
        print("=" * 78)

        conds = {
            "CC":        lambda s, bn=bn: make_untrained_cc(s, bn),
            "Rand-syn":  lambda s, bn=bn: apply_syn_shuffle(
                             make_untrained_cc(s, bn), np.random.default_rng(s + 10_000)),
            "Rand-sign": lambda s, bn=bn: apply_sign_shuffle(
                             make_untrained_cc(s, bn), np.random.default_rng(s + 20_000)),
        }

        level = {}
        for name, fn in conds.items():
            pops, rc, re_, rej = run_condition(fn, args.n_models, dataset,
                                               on_idx, dataset.dt)
            mean_cos = rc.mean(0)
            mean_euc = re_.mean(0)
            span, floor, ratio, ok = resolvability(
                mean_cos, pops.reshape(-1, pops.shape[-1]))

            # metric agreement: per-model (strict) and on the mean
            taus = np.array([kendalltau(upper(rc[i]), upper(re_[i])).statistic
                             for i in range(rc.shape[0])])
            tau_mean = kendalltau(upper(mean_cos), upper(mean_euc)).statistic

            # circularity, reported ONLY if resolvable
            r_circ = spearmanr(upper(mean_cos), upper(circ)).statistic if ok else np.nan

            level[name] = dict(span=span, floor=floor, ratio=ratio, ok=ok,
                               r_circ=r_circ, rej=rej,
                               resp=float(np.abs(pops).max()),
                               tau_med=float(np.median(taus)),
                               tau_mean=float(tau_mean))

            flag = "RESOLVABLE" if ok else "below floor"
            circ_s = f"{r_circ:+.3f}" if ok else "  n/a "
            print(f"  {name:10s} span {span:.2e}  floor {floor:.2e}  "
                  f"ratio {ratio:6.2f}x  {flag:11s}  r(circ) {circ_s}  "
                  f"|resp| {level[name]['resp']:.2f}  rej {rej}  "
                  f"tau_pm {level[name]['tau_med']:.3f}")

        # THE CONTROL. Does CC gain circular structure the shuffles do not?
        if level["CC"]["ok"] and level["Rand-syn"]["ok"] and level["Rand-sign"]["ok"]:
            d_syn  = level["CC"]["r_circ"] - level["Rand-syn"]["r_circ"]
            d_sign = level["CC"]["r_circ"] - level["Rand-sign"]["r_circ"]
            print(f"\n  All resolvable. CC circularity advantage: "
                  f"vs syn {d_syn:+.3f}, vs sign {d_sign:+.3f}")
            if max(d_syn, d_sign) < 0.05:
                print("  -> CC does NOT exceed the shuffles. Perturbation "
                      "magnitude, not wiring,\n     produced any geometry here.")
            else:
                print("  -> CC exceeds the shuffles. Consistent with a wiring "
                      "prior that requires\n     perturbation to become visible.")
        elif not level["CC"]["ok"]:
            print("\n  CC not resolvable. No statistic reported at this scale.")

        rows.append((bn, level))
        print()

    # ── summary ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'noise':>7} {'CC ratio':>9} {'CC r(circ)':>11} "
          f"{'syn r(circ)':>12} {'sign r(circ)':>13}")
    for bn, lv in rows:
        f = lambda k: (f"{lv[k]['r_circ']:+.3f}" if lv[k]["ok"] else "   n/a")
        print(f"  {bn:>7.3f} {lv['CC']['ratio']:>8.2f}x {f('CC'):>11} "
              f"{f('Rand-syn'):>12} {f('Rand-sign'):>13}")

    first_ok = next((bn for bn, lv in rows if lv["CC"]["ok"]), None)
    print()
    if first_ok is None:
        print("  The guard never cleared. Untrained CC networks have no")
        print("  measurable representational geometry across the swept range.")
        print("  Training therefore CREATES the directional structure rather")
        print("  than amplifying a wiring prior at initialization. This is the")
        print("  opposite of the original Experiment 4 conclusion, and it is a")
        print("  claim the original could not have tested, because it never")
        print("  checked whether its RDMs were resolvable.")
    else:
        print(f"  Untrained CC geometry first becomes resolvable at "
              f"BIAS_NOISE = {first_ok}.")
        print("  Whether that constitutes a connectome prior depends on the")
        print("  CC-vs-shuffle comparison at and above that scale, and on")
        print("  whether the network remains in a defensible regime. Report the")
        print("  threshold, the gap, and the response magnitudes together.")

    print("\n  Caveat, unavoidable: raising BIAS_NOISE moves the network away")
    print("  from the Flyvis prior. At large perturbations 'untrained CC' no")
    print("  longer means 'the connectome at initialization.' The response")
    print("  magnitudes and rejection counts above are reported so that line")
    print("  can be drawn deliberately rather than by accident.")

    np.savez(results_dir / "exp4_sweep.npz",
             noise=np.array([r[0] for r in rows]),
             **{f"{c}_{k}": np.array([r[1][c][k] for r in rows])
                for c in ("CC", "Rand-syn", "Rand-sign")
                for k in ("span", "floor", "ratio", "r_circ", "resp",
                          "tau_med", "tau_mean", "rej")})

    # ── figure ───────────────────────────────────────────────────────────────
    noise = np.array([r[0] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for c, col in [("CC", "k"), ("Rand-syn", "tab:blue"), ("Rand-sign", "tab:red")]:
        ax[0].plot(noise, [r[1][c]["ratio"] for r in rows], "o-", color=col, label=c)
    ax[0].axhline(RESOLVE_MARGIN, ls="--", c="gray",
                  label=f"resolvability threshold ({RESOLVE_MARGIN:g}x)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("BIAS_NOISE"); ax[0].set_ylabel("RDM span / float32 floor")
    ax[0].set_title("Is the RDM resolvable?"); ax[0].legend(fontsize=8)

    for c, col in [("CC", "k"), ("Rand-syn", "tab:blue"), ("Rand-sign", "tab:red")]:
        y = [r[1][c]["r_circ"] for r in rows]
        ax[1].plot(noise, y, "o-", color=col, label=c)
    ax[1].axhline(0, ls=":", c="gray")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("BIAS_NOISE")
    ax[1].set_ylabel("r(mean RDM, circular reference)")
    ax[1].set_title("Circular direction structure\n(only where resolvable)")
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "exp4_sweep.png", dpi=150, bbox_inches="tight")
    print("\nSaved results/exp4_sweep.npz and figures/exp4_sweep.png")


if __name__ == "__main__":
    main()
