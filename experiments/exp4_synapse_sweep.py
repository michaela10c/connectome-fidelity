#!/usr/bin/env python3
"""
exp4_synapse_sweep.py — the synapse-strength half of the Exp4b perturbation
sweep, sized to actually measure what the preprint's Table tab:sweep and
surrounding prose claim.

WHY THIS SCRIPT EXISTS
-----------------------
exp4_perturbation_sweep.py (already verified against exp4_sweep.npz) only
sweeps BIAS_NOISE; SYN_STRENGTH_NOISE is held fixed there by explicit design.
A separate synapse-noise sweep produced the preprint's CC span/ratio/|resp|/
rejection numbers at sigma = 0.002/0.008/0.032/0.128 -- those are CONFIRMED
correct (matched an n_models=5 run exactly). But that same run was killed
before finishing, and even finished it would not have produced two other
claims currently in the preprint text:

  1. "stability-check acceptance rate collapses: 100%, 100%, 68%, 5% across
     the four levels (n = 25, 25, 40, 60 seeds)"
  2. "accepted and rejected models are silenced at 43.3% and 43.7%
     respectively (Mann-Whitney p=0.38)" at sigma=0.032

Both describe a MUCH larger sample than n_models=5, and the acceptance-rate
claim in particular requires a FIXED seed budget per level (not the
resample-to-N-accepted design run_condition uses), since resample-to-N-
accepted by construction always reports n_models accepted regardless of the
true rate -- it cannot produce a "5% acceptance" statistic on its own.

DESIGN
------
Two separate measurements per synapse-noise level, kept separate on purpose
so the (already-verified) small-n geometry numbers are not disturbed by
changing the large-n stability characterization:

  (A) GEOMETRY (n_models=5, resample-to-N-accepted): calls run_condition()
      UNCHANGED from exp4_perturbation_sweep.py. Reproduces the already-
      confirmed CC span/ratio/|resp| numbers, and -- new -- completes
      Rand-sign at sigma=0.128, which the killed run never reached.

  (B) STABILITY + PRUNING CHARACTERIZATION (fixed budget, no resampling):
      draws N_SEEDS_PER_LEVEL (default 100) FIXED seeds per level for the CC
      condition only (this is what the preprint's acceptance-rate and
      Mann-Whitney claims describe), checks is_stable on each with a single
      forward pass (cheap -- one stimulus, not the full 12-condition RDM),
      records the zeroed-synapse fraction for every seed whether accepted or
      rejected, and reports: acceptance rate, and a Mann-Whitney U test
      comparing zeroed-fraction between accepted and rejected seeds (the
      "does the stability filter select on deletion extent" check).

      100 seeds/level is sized so even a 5% acceptance rate yields ~5
      successes with a reasonably estimated rate (95% CI roughly +/-4pp at
      n=100, p=0.05) -- tighter than would be safe at n=60. Override with
      --n_seeds for a faster/coarser pilot.

Everything reused below (cosine_distance_stable, euclidean_normalized,
build_rdm, upper, circular_reference, resolvability, apply_syn_shuffle,
apply_sign_shuffle, get_cell_types, get_population_vector, is_stable,
run_condition) is copied VERBATIM from exp4_perturbation_sweep.py -- not
reimplemented -- specifically so the geometry numbers stay consistent with
what's already been confirmed against exp4_sweep.npz. Only make_untrained_cc
is changed (bias_noise fixed at 0.05, syn_noise is now the swept argument),
and measure_stability_and_pruning is new.

USAGE
-----
    python exp4_synapse_sweep.py                    # default: n_seeds=100
    python exp4_synapse_sweep.py --n_seeds 30        # faster pilot
    python exp4_synapse_sweep.py --syn 0.002 0.032   # just the two levels
                                                       # that matter most

Writes: results/exp4_synapse_sweep.npz, figures/exp4_synapse_sweep.png
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, kendalltau, mannwhitneyu

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)

OVERFLOW_LIMIT = 1e6
MAX_ATTEMPTS   = 20
FLOAT32_EPS    = float(np.finfo(np.float32).eps)
RESOLVE_MARGIN = 10.0

# Bias and time-constant noise are held FIXED; only synapse-strength noise is
# swept. Mirrors exp4_perturbation_sweep.py's design principle in reverse.
BIAS_NOISE        = 0.05
TIME_CONST_NOISE  = 0.005

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
N_DIR  = len(ANGLES)

results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
figures_dir = Path("figures"); figures_dir.mkdir(exist_ok=True)

import flyvis
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.network import Network
from flyvis.utils.activity_utils import LayerActivity


# ── copied verbatim from exp4_perturbation_sweep.py ─────────────────────────
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
    off = upper(rdm)
    span = float(off.max() - off.min())
    floor = FLOAT32_EPS * float(np.abs(pop_matrix).max())
    ratio = span / floor if floor > 0 else np.inf
    return span, floor, ratio, ratio >= RESOLVE_MARGIN


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


def run_condition(make_fn, n_models, dataset, stim_idx, dt, label="", progress_every=5):
    """Return (pop_matrices, rdms_cos, rdms_euc, n_rejected). Same
    resample-to-N-accepted logic as exp4_perturbation_sweep.py -- used ONLY
    for the already-verified geometry numbers (n_models=5) -- plus periodic
    progress/ETA logging, since a single draw can take ~40s and a level with
    high rejection (e.g. sigma=0.128) can run for an hour+ with no other
    output otherwise."""
    import time
    check = dataset[stim_idx[0]]
    if not isinstance(check, torch.Tensor):
        check = torch.tensor(check, dtype=torch.float32)
    check = check.to(DEVICE)

    pops, rc, re_ = [], [], []
    accepted = seed = rejected = 0
    cts = None
    t0 = time.time()
    draw_times = []

    while accepted < n_models and seed < n_models * MAX_ATTEMPTS:
        t_draw = time.time()
        net = make_fn(seed); seed += 1
        if not is_stable(net, check, dt):
            rejected += 1; del net; torch.cuda.empty_cache()
            draw_times.append(time.time() - t_draw)
            if seed % progress_every == 0:
                _log_progress(label, accepted, n_models, rejected, seed,
                              n_models * MAX_ATTEMPTS, draw_times, t0)
            continue
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
            rejected += 1; del net; torch.cuda.empty_cache()
            draw_times.append(time.time() - t_draw)
            if seed % progress_every == 0:
                _log_progress(label, accepted, n_models, rejected, seed,
                              n_models * MAX_ATTEMPTS, draw_times, t0)
            continue

        pm = np.stack(pv_list, 0)
        pops.append(pm)
        rc.append(build_rdm(pm, "cosine"))
        re_.append(build_rdm(pm, "euclidean_normalized"))
        accepted += 1
        del net; torch.cuda.empty_cache()
        draw_times.append(time.time() - t_draw)
        _log_progress(label, accepted, n_models, rejected, seed,
                      n_models * MAX_ATTEMPTS, draw_times, t0)

    if accepted == 0:
        # All draws rejected within the attempt budget. This is itself a
        # result (total instability at this noise level), not an error --
        # return None for the arrays so the caller can report it cleanly
        # instead of crashing on np.stack([]).
        return None, None, None, rejected

    return np.stack(pops, 0), np.stack(rc, 0), np.stack(re_, 0), rejected


def _log_progress(label, accepted, n_models, rejected, draws_so_far, max_draws,
                  draw_times, t0):
    import time
    elapsed = time.time() - t0
    mean_draw_s = float(np.mean(draw_times)) if draw_times else 0.0
    remaining_draws_est = max(0, max_draws - draws_so_far) if accepted < n_models else 0
    accept_rate = accepted / draws_so_far if draws_so_far else 0.0
    if accept_rate > 0 and accepted < n_models:
        draws_needed_est = (n_models - accepted) / accept_rate
        eta_s = draws_needed_est * mean_draw_s
        eta_str = f"  ETA ~{eta_s/60:.0f}min"
    else:
        eta_str = ""
    print(f"    [progress] {label} accepted {accepted}/{n_models}  "
          f"rejected {rejected}  draws {draws_so_far}  "
          f"elapsed {elapsed/60:.1f}min  avg {mean_draw_s:.1f}s/draw{eta_str}",
          flush=True)


# ── new: synapse noise is the swept parameter (bias fixed) ──────────────────
def make_untrained_cc(seed, syn_noise):
    rng = np.random.default_rng(seed)
    net = Network()
    with torch.no_grad():
        net.nodes_bias.data += torch.tensor(
            rng.normal(0, BIAS_NOISE, size=net.nodes_bias.shape), dtype=torch.float32)
        net.nodes_time_const.data = torch.clamp(
            net.nodes_time_const.data + torch.tensor(
                rng.normal(0, TIME_CONST_NOISE, size=net.nodes_time_const.shape),
                dtype=torch.float32), min=0.001)
        raw = net.edge_params.syn_strength.raw_values
        raw.data = torch.clamp(raw.data + torch.tensor(
            rng.normal(0, syn_noise, size=raw.shape),
            dtype=torch.float32), min=0.0)
    return net.to(DEVICE)


def zeroed_fraction(net):
    """Fraction of edges_syn_strength clamped to exactly 0.0 by the
    non-negativity clamp in make_untrained_cc."""
    raw = net.edge_params.syn_strength.raw_values.data.cpu().numpy()
    return float((raw == 0.0).mean())


# ── new: fixed-budget stability + pruning characterization ──────────────────
def measure_stability_and_pruning(syn_noise, n_seeds, dataset, stim_idx, dt,
                                   progress_every=20):
    """
    Draw n_seeds FIXED seeds (no resampling) for the CC condition at this
    synapse-noise level. For each: check is_stable on the single check
    stimulus (cheap), record whether accepted, and record the zeroed-synapse
    fraction regardless of outcome. Returns a dict with the acceptance rate
    and the Mann-Whitney comparison the preprint claims.
    """
    import time
    check = dataset[stim_idx[0]]
    if not isinstance(check, torch.Tensor):
        check = torch.tensor(check, dtype=torch.float32)
    check = check.to(DEVICE)

    accepted_zeroed, rejected_zeroed = [], []
    t0 = time.time()
    for seed in range(n_seeds):
        net = make_untrained_cc(seed, syn_noise)
        zf = zeroed_fraction(net)
        if is_stable(net, check, dt):
            accepted_zeroed.append(zf)
        else:
            rejected_zeroed.append(zf)
        del net
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if (seed + 1) % progress_every == 0 or seed + 1 == n_seeds:
            elapsed = time.time() - t0
            rate = elapsed / (seed + 1)
            remaining = n_seeds - (seed + 1)
            eta_s = remaining * rate
            print(f"    [progress] sigma={syn_noise} stability-probe "
                  f"seed {seed+1}/{n_seeds}  accepted {len(accepted_zeroed)}  "
                  f"rejected {len(rejected_zeroed)}  "
                  f"elapsed {elapsed/60:.1f}min  ETA ~{eta_s/60:.1f}min", flush=True)

    n_accepted = len(accepted_zeroed)
    n_rejected = len(rejected_zeroed)
    acceptance_rate = n_accepted / n_seeds if n_seeds else float("nan")

    mw_u, mw_p = (float("nan"), float("nan"))
    if n_accepted >= 2 and n_rejected >= 2:
        res = mannwhitneyu(accepted_zeroed, rejected_zeroed, alternative="two-sided")
        mw_u, mw_p = float(res.statistic), float(res.pvalue)

    return dict(
        n_seeds=n_seeds, n_accepted=n_accepted, n_rejected=n_rejected,
        acceptance_rate=acceptance_rate,
        accepted_zeroed_mean=(float(np.mean(accepted_zeroed)) if accepted_zeroed else None),
        rejected_zeroed_mean=(float(np.mean(rejected_zeroed)) if rejected_zeroed else None),
        accepted_zeroed_median=(float(np.median(accepted_zeroed)) if accepted_zeroed else None),
        rejected_zeroed_median=(float(np.median(rejected_zeroed)) if rejected_zeroed else None),
        mannwhitney_U=mw_u, mannwhitney_p=mw_p,
    )


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", type=float, nargs="+",
                    default=[0.002, 0.008, 0.032, 0.128],
                    help="SYN_STRENGTH_NOISE values to sweep (matches the "
                         "preprint's Table tab:sweep levels by default)")
    ap.add_argument("--n_models", type=int, default=5,
                    help="accepted models per condition for the GEOMETRY "
                         "measurement (default 5, matches the already-"
                         "verified exp4_sweep.npz numbers -- do not change "
                         "this unless you intend to produce NEW geometry "
                         "numbers, not verify the old ones)")
    ap.add_argument("--n_seeds", type=int, default=100,
                    help="FIXED seed budget per level for the acceptance-"
                         "rate / Mann-Whitney characterization (CC only). "
                         "100 gives a reasonably tight estimate even at 5%% "
                         "acceptance; use fewer for a faster pilot.")
    args = ap.parse_args()

    dataset = MovingEdge(offsets=[-10, 11], intensities=[0, 1], speeds=[19],
                         height=80, post_pad_mode="continue",
                         t_pre=1.0, t_post=1.0, dt=1/200, angles=ANGLES)
    on_idx = [i for i, r in dataset.arg_df.iterrows() if r["intensity"] == 1]
    assert len(on_idx) == N_DIR

    circ = circular_reference()
    rows = []
    ckpt_dir = results_dir / "exp4_synapse_sweep_ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}   n_models(geometry)={args.n_models}   "
          f"n_seeds(stability)={args.n_seeds}")
    print(f"Swept axis: SYN_STRENGTH_NOISE = {args.syn}")
    print(f"Held fixed: BIAS_NOISE = {BIAS_NOISE}, "
          f"TIME_CONST_NOISE = {TIME_CONST_NOISE}\n")

    for sn in args.syn:
        ckpt_path = ckpt_dir / f"level_{sn}.npz"
        if ckpt_path.exists():
            print(f"[resume] sigma={sn} already completed, loading checkpoint "
                  f"from {ckpt_path} instead of rerunning")
            d = np.load(ckpt_path, allow_pickle=True)
            level = d["level"].item()
            stab = d["stab"].item()
            rows.append((sn, level, stab))
            continue

        print("=" * 78)
        print(f"SYN_STRENGTH_NOISE = {sn}   (BIAS_NOISE fixed at {BIAS_NOISE})")
        print("=" * 78)

        # --- (A) geometry, n_models=5, matches already-verified numbers ---
        conds = {
            "CC":        lambda s, sn=sn: make_untrained_cc(s, sn),
            "Rand-syn":  lambda s, sn=sn: apply_syn_shuffle(
                             make_untrained_cc(s, sn), np.random.default_rng(s + 10_000)),
            "Rand-sign": lambda s, sn=sn: apply_sign_shuffle(
                             make_untrained_cc(s, sn), np.random.default_rng(s + 20_000)),
        }
        level = {}
        for name, fn in conds.items():
            pops, rc, re_, rej = run_condition(
                fn, args.n_models, dataset, on_idx, dataset.dt,
                label=f"sigma={sn} {name}")
            if pops is None:
                level[name] = dict(span=float("nan"), floor=float("nan"),
                                   ratio=0.0, ok=False, r_circ=float("nan"),
                                   rej=rej, resp=float("nan"))
                print(f"  [geometry] {name:10s} 0/{args.n_models} accepted in "
                      f"{rej} draws -- TOTAL INSTABILITY at this noise level "
                      f"(not just below the resolvability floor; no model "
                      f"survived the stability check at all)")
                continue
            mean_cos, mean_euc = rc.mean(0), re_.mean(0)
            span, floor, ratio, ok = resolvability(mean_cos, pops.reshape(-1, pops.shape[-1]))
            r_circ = spearmanr(upper(mean_cos), upper(circ)).statistic if ok else np.nan
            level[name] = dict(span=span, floor=floor, ratio=ratio, ok=ok,
                               r_circ=r_circ, rej=rej, resp=float(np.abs(pops).max()))
            flag = "RESOLVABLE" if ok else "below floor"
            circ_s = f"{r_circ:+.4f}" if ok else "  n/a "
            print(f"  [geometry] {name:10s} span {span:.2e}  ratio {ratio:6.2f}x  "
                  f"{flag:11s}  r(circ) {circ_s}  |resp| {level[name]['resp']:.2f}  rej {rej}")

        # --- (B) stability + pruning, fixed n_seeds, CC only ---
        stab = measure_stability_and_pruning(sn, args.n_seeds, dataset, on_idx, dataset.dt)
        print(f"\n  [stability] CC: {stab['n_accepted']}/{stab['n_seeds']} accepted "
              f"({100*stab['acceptance_rate']:.1f}%)")
        if stab['accepted_zeroed_mean'] is not None:
            print(f"              accepted zeroed-fraction: mean={stab['accepted_zeroed_mean']:.3f} "
                  f"median={stab['accepted_zeroed_median']:.3f}")
        if stab['rejected_zeroed_mean'] is not None:
            print(f"              rejected zeroed-fraction: mean={stab['rejected_zeroed_mean']:.3f} "
                  f"median={stab['rejected_zeroed_median']:.3f}")
        if not np.isnan(stab['mannwhitney_p']):
            print(f"              Mann-Whitney (accepted vs rejected zeroed-fraction): "
                  f"U={stab['mannwhitney_U']:.1f}  p={stab['mannwhitney_p']:.4f}")
        else:
            print(f"              Mann-Whitney: not computable "
                  f"(need >=2 in both groups; got {stab['n_accepted']} accepted, "
                  f"{stab['n_rejected']} rejected)")

        rows.append((sn, level, stab))
        np.savez(ckpt_path, level=level, stab=stab)
        print(f"[checkpoint] saved {ckpt_path}")
        print()

    # ── summary ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'sigma':>7} {'CC ratio':>9} {'Rsign ratio':>12} {'Rsign r(circ)':>14} "
          f"{'accept%':>9} {'n_seeds':>8} {'MW p':>8}")
    for sn, lv, stab in rows:
        rsign_ratio = lv['Rand-sign']['ratio']
        rsign_circ = (f"{lv['Rand-sign']['r_circ']:+.3f}" if lv['Rand-sign']['ok'] else "   n/a")
        mw = f"{stab['mannwhitney_p']:.3f}" if not np.isnan(stab['mannwhitney_p']) else "  n/a"
        print(f"  {sn:>7.3f} {lv['CC']['ratio']:>8.2f}x {rsign_ratio:>11.2f}x "
              f"{rsign_circ:>14} {100*stab['acceptance_rate']:>8.1f}% "
              f"{stab['n_seeds']:>8d} {mw:>8}")

    print("\n  Check against the preprint's claims:")
    print("  1. 'Rand-sign is the only resolvable point, at sigma=0.032' -- ")
    print("     verify no other (sigma, condition) pair shows RESOLVABLE above.")
    print("  2. Acceptance-rate curve -- compare the accept%% column to the")
    print("     preprint's claimed 100%, 100%, 68%, 5% (preprint's n was 25/25/40/60;")
    print("     this run uses a FIXED n_seeds per level instead, so exact seed")
    print("     counts won't match -- only the rate and its trend should be compared).")
    print("  3. Mann-Whitney p -- compare to the preprint's p=0.38 at sigma=0.032.")
    print("     A different p-value here does not mean the original was wrong; it")
    print("     means this is a NEW measurement from new draws, not a reproduction.")

    np.savez(results_dir / "exp4_synapse_sweep.npz",
             syn_noise=np.array([r[0] for r in rows]),
             **{f"{c}_{k}": np.array([r[1][c][k] for r in rows])
                for c in ("CC", "Rand-syn", "Rand-sign")
                for k in ("span", "floor", "ratio", "r_circ", "resp", "rej")},
             acceptance_rate=np.array([r[2]['acceptance_rate'] for r in rows]),
             n_accepted=np.array([r[2]['n_accepted'] for r in rows]),
             n_rejected=np.array([r[2]['n_rejected'] for r in rows]),
             mannwhitney_p=np.array([r[2]['mannwhitney_p'] for r in rows]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    syn = np.array([r[0] for r in rows])
    for c, col in [("CC", "k"), ("Rand-syn", "tab:blue"), ("Rand-sign", "tab:red")]:
        ax[0].plot(syn, [r[1][c]["ratio"] for r in rows], "o-", color=col, label=c)
    ax[0].axhline(RESOLVE_MARGIN, ls="--", c="gray", label=f"threshold ({RESOLVE_MARGIN:g}x)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("SYN_STRENGTH_NOISE"); ax[0].set_ylabel("RDM span / float32 floor")
    ax[0].set_title("Resolvability"); ax[0].legend(fontsize=8)

    ax[1].plot(syn, [100*r[2]['acceptance_rate'] for r in rows], "o-", color="k")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("SYN_STRENGTH_NOISE"); ax[1].set_ylabel("CC acceptance rate (%)")
    ax[1].set_title(f"Stability acceptance rate (n_seeds={args.n_seeds}/level)")

    plt.tight_layout()
    plt.savefig(figures_dir / "exp4_synapse_sweep.png", dpi=150, bbox_inches="tight")
    print("\nSaved results/exp4_synapse_sweep.npz and figures/exp4_synapse_sweep.png")


if __name__ == "__main__":
    main()
