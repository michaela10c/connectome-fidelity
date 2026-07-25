"""
step7_ei_cond_fit_weights.py -- conductance-based counterpart to
step7_ei_fit_weights.py's fitting search. The bisection algorithm itself
(direction detection, feasibility floor, fixed-point acceptance,
multi-start) is proven and copied over UNCHANGED -- it's generic to "fit
two coupled rate axes," not specific to current-based dynamics. Only two
things differ: which simulation gets called, and the weight bracket,
which needs to be narrowed based on what the conductance-based validation
sweep just found.

WHY THE BRACKET IS DIFFERENT, AND WHY THIS IS THE RISKY PART
---------------------------------------------------------------
validate_conductance_infrastructure.py found: w_e=0.05 leaves I
completely silent (hz_i=0.00, confirmed via bit-identical output across
every tested w_i -- I never fires, so its own weight has zero effect).
w_e=0.2 and w_e=1.0 both keep I alive, but produce E rates of 50-88 Hz --
far above the E=12.035 Hz target this model needs to hit to be comparable
to the current-based model's fitted result. This means the true solution
almost certainly requires w_e SMALLER than anything confirmed viable so
far (0.2), while staying far enough above the confirmed-dead 0.05 to keep
I alive. The feasibility floor is not a theoretical safeguard here -- the
achievable target may sit uncomfortably close to the dead-zone boundary,
and this search needs to find that boundary directly rather than assume
it's far away.

USAGE: run_simulation_ei_cond-based conditions only. Same call pattern
as step7_ei_fit_weights.py's fit_weights_ei(), swap the import.
"""

import os
import shutil

import h5py
import numpy as np

import step2_build_and_simulate_microns as step2
import step2_ei_recurrent as step2_ei
from step2_ei_conductance_recurrent import build_v1_recurrent_ei_cond

# Targets carried over UNCHANGED from the current-based model -- these are
# the self-consistent anchors measured from the real network's own
# converged current-based fit. Using the same targets is what makes a
# future fidelity comparison between the two model classes meaningful.
TARGET_HZ_E = 12.035
TARGET_HZ_I = 1.889

RATE_TOLERANCE = 0.05
MAX_ITERS = 12
MAX_ROUNDS = 4

# Narrowed based on the validation sweep just run, NOT copied from the
# current-based model's [2, 15] bracket -- conductance weights (nS) are a
# different physical quantity, confirmed non-comparable in magnitude.
# Deliberately set BELOW the confirmed-dead 0.05, trusting the same
# pull-out-of-dead-zone mechanism already proven on the current-based
# model to escape it if the search wanders there, rather than assuming
# the true solution is comfortably clear of the boundary.
WEIGHT_LO_INIT = 0.01
WEIGHT_HI_INIT = 0.5

MIN_STABLE_HZ = 0.1
MAX_STABLE_HZ = 200.0

GRAY_SCREEN_MS = 500.0
ANALYSIS_MS = 3000.0 - GRAY_SCREEN_MS


def is_stable_ei(hz):
    return bool(MIN_STABLE_HZ <= hz <= MAX_STABLE_HZ)


def fitted_dir_for(tag):
    return f"output_pointnet_fitted_ei_cond_{tag}"


def probe_mean_rates_ei_cond(edge_file, tag, weight_e, weight_i, N_neurons,
                             node_types, e_mask, i_mask, theta=0):
    """Conductance-based counterpart to probe_mean_rates_ei() -- identical
    logic, only the build call differs (build_v1_recurrent_ei_cond instead
    of step2_ei.build_v1_recurrent_ei).
    """
    probe_net_dir = f"network_ei_cond_probe_{tag}"
    probe_out_dir = f"output_ei_cond_probe_{tag}"
    if os.path.exists(probe_net_dir):
        shutil.rmtree(probe_net_dir)
    if os.path.exists(probe_out_dir):
        shutil.rmtree(probe_out_dir)

    neurons_df = step2_ei.load_nodes_ei()
    edges_df = step2_ei.load_edges_signed(edge_file)
    build_v1_recurrent_ei_cond(neurons_df, edges_df, probe_net_dir,
                               weight_e, weight_i, node_types=node_types)
    circuit_cfg = step2.make_circuit_config(f"ei_cond_probe_{tag}", probe_net_dir,
                                            step2.LGN_DIR)
    step2.run_pointnet_single(theta, f"ei_cond_probe_{tag}", probe_out_dir, circuit_cfg)

    fpath = os.path.join(probe_out_dir, f"theta{theta:03d}", "spikes.h5")
    with h5py.File(fpath, "r") as h5:
        node_ids = h5["spikes/microns_v1/node_ids"][()]
        timestamps = h5["spikes/microns_v1/timestamps"][()]
    m = timestamps >= GRAY_SCREEN_MS
    node_ids = node_ids[m]

    n_e = e_mask.sum()
    n_i = i_mask.sum()
    e_spikes = np.isin(node_ids, np.where(e_mask)[0]).sum()
    i_spikes = np.isin(node_ids, np.where(i_mask)[0]).sum()
    hz_e = e_spikes / max(n_e, 1) / (ANALYSIS_MS / 1000.0)
    hz_i = i_spikes / max(n_i, 1) / (ANALYSIS_MS / 1000.0)

    shutil.rmtree(probe_net_dir, ignore_errors=True)
    shutil.rmtree(probe_out_dir, ignore_errors=True)

    stable = is_stable_ei(hz_e) and is_stable_ei(hz_i)
    return hz_e, hz_i, stable


def _bisect_one_axis_cond(edge_file, tag, target_hz, fixed_other_weight,
                          is_e_axis, N_neurons, node_types, e_mask, i_mask):
    """Identical algorithm to step7_ei_fit_weights.py's _bisect_one_axis()
    -- direction detected not assumed, feasibility floor enforced exactly
    the same way. Copied rather than imported so this module has zero
    dependency on the current-based fitting module staying unchanged.
    """
    def probe(w):
        we, wi = (w, fixed_other_weight) if is_e_axis else (fixed_other_weight, w)
        hz_e, hz_i, stable = probe_mean_rates_ei_cond(
            edge_file, tag, we, wi, N_neurons, node_types, e_mask, i_mask)
        axis_rate = hz_e if is_e_axis else hz_i
        other_rate = hz_i if is_e_axis else hz_e
        other_dead = other_rate < MIN_STABLE_HZ
        return axis_rate, other_dead, stable

    lo, hi = WEIGHT_LO_INIT, WEIGHT_HI_INIT
    rate_lo, dead_lo, stable_lo = probe(lo)
    rate_hi, dead_hi, stable_hi = probe(hi)
    increasing = rate_hi >= rate_lo

    if dead_lo and dead_hi:
        raise RuntimeError(
            f"the OTHER axis is dead (silent) across the entire bracket "
            f"[{lo}, {hi}] for this weight -- the conductance-based "
            f"bracket may need to be widened, this is a real finding "
            f"about where the E/I-viability boundary sits, not a search bug")

    pull_iters = 0
    while dead_lo and pull_iters < 8:
        lo = (lo + hi) / 2
        rate_lo, dead_lo, stable_lo = probe(lo)
        pull_iters += 1
    while dead_hi and pull_iters < 8:
        hi = (lo + hi) / 2
        rate_hi, dead_hi, stable_hi = probe(hi)
        pull_iters += 1

    def bracketed(rl, rh):
        return (rl <= target_hz <= rh) if increasing else (rh <= target_hz <= rl)

    expand_iters = 0
    while not bracketed(rate_lo, rate_hi) and expand_iters < 6:
        rate_at_lo_is_too_far = (target_hz < rate_lo) if increasing else (target_hz > rate_lo)
        if rate_at_lo_is_too_far:
            lo /= 2
            rate_lo, dead_lo, stable_lo = probe(lo)
            if dead_lo:
                lo = (lo + hi) / 2
                rate_lo, dead_lo, stable_lo = probe(lo)
        else:
            hi *= 2
            rate_hi, dead_hi, stable_hi = probe(hi)
        expand_iters += 1

    if not bracketed(rate_lo, rate_hi):
        best = lo if abs(rate_lo - target_hz) < abs(rate_hi - target_hz) else hi
        best_rate = rate_lo if best == lo else rate_hi
        best_stable = stable_lo if best == lo else stable_hi
        return best, best_rate, best_stable

    stable = stable_lo and stable_hi
    for _ in range(MAX_ITERS):
        mid = (lo + hi) / 2
        rate_mid, dead_mid, stable_mid = probe(mid)
        stable = stable_mid
        if dead_mid:
            if is_e_axis:
                lo = mid
            else:
                hi = mid
            continue
        if abs(rate_mid - target_hz) <= RATE_TOLERANCE * target_hz:
            return mid, rate_mid, stable_mid
        rate_too_low = rate_mid < target_hz
        if rate_too_low == increasing:
            lo = mid
        else:
            hi = mid
    return mid, rate_mid, stable_mid


def _fit_weights_ei_cond_single_start(edge_file, tag, target_hz_e, target_hz_i,
                                      N_neurons, node_types, e_mask, i_mask,
                                      start_e, start_i, verbose=True):
    """Identical structure to _fit_weights_ei_single_start()."""
    w_e, w_i = start_e, start_i
    last_stable = False
    prev_w_e, prev_w_i = None, None
    for rnd in range(MAX_ROUNDS):
        w_e, hz_e, stable_e = _bisect_one_axis_cond(
            edge_file, tag, target_hz_e, w_i, True, N_neurons,
            node_types, e_mask, i_mask)
        w_i, hz_i, stable_i = _bisect_one_axis_cond(
            edge_file, tag, target_hz_i, w_e, False, N_neurons,
            node_types, e_mask, i_mask)
        last_stable = stable_e and stable_i
        if verbose:
            print(f"    [{tag}] (start E={start_e:.4f},I={start_i:.4f}) "
                  f"round {rnd+1}: E->w={w_e:.4f} hz={hz_e:.2f} "
                  f"(target {target_hz_e:.2f})  I->w={w_i:.4f} hz={hz_i:.2f} "
                  f"(target {target_hz_i:.2f})  stable={last_stable}")
        e_ok = abs(hz_e - target_hz_e) <= RATE_TOLERANCE * target_hz_e
        i_ok = abs(hz_i - target_hz_i) <= RATE_TOLERANCE * target_hz_i
        if e_ok and i_ok and last_stable:
            return w_e, w_i, hz_e, hz_i, True, "full"

        if (prev_w_e is not None and last_stable
                and abs(w_e - prev_w_e) < 1e-6 and abs(w_i - prev_w_i) < 1e-6):
            missed = []
            if not e_ok:
                missed.append(f"E (hz={hz_e:.2f}/target={target_hz_e:.2f})")
            if not i_ok:
                missed.append(f"I (hz={hz_i:.2f}/target={target_hz_i:.2f})")
            if verbose:
                print(f"    [{tag}] fixed point reached from this start "
                      f"(weights unchanged, stable). Outside tolerance: "
                      f"{', '.join(missed)}")
            return w_e, w_i, hz_e, hz_i, True, "fixed_point"
        prev_w_e, prev_w_i = w_e, w_i

    at_bracket_edge = (w_i >= WEIGHT_HI_INIT * 0.999 or w_i <= WEIGHT_LO_INIT * 1.001
                       or w_e >= WEIGHT_HI_INIT * 0.999 or w_e <= WEIGHT_LO_INIT * 1.001)
    raise RuntimeError(
        f"did not converge within {MAX_ROUNDS} rounds from start "
        f"E={start_e:.4f},I={start_i:.4f} (final: E hz={hz_e:.2f}/"
        f"target={target_hz_e:.2f}, I hz={hz_i:.2f}/target={target_hz_i:.2f}, "
        f"stable={last_stable})"
        + (" -- an axis is stuck at its search bracket edge" if at_bracket_edge else ""))


# Alternate starts, rescaled into the new [0.01, 0.5] bracket -- same
# spread-across-the-bracket logic as the current-based model's
# _ALT_STARTS, not theory-derived, just a pragmatic hedge against one
# arbitrary default landing on a fixed point.
_ALT_STARTS = [(0.35, 0.35), (0.3, 0.1), (0.1, 0.3)]


def fit_weights_ei_cond(edge_file, tag, target_hz_e=TARGET_HZ_E,
                        target_hz_i=TARGET_HZ_I, N_neurons=2196,
                        node_types=None, e_mask=None, i_mask=None, verbose=True):
    """Multi-start wrapper, identical structure to fit_weights_ei(). Default
    start is (0.15, 0.15) -- roughly midway in log-space between the
    confirmed-dead 0.05 and the confirmed-viable-but-overshooting 0.2,
    not copied from the current-based model's (5,5).
    """
    if node_types is None:
        node_types = step2_ei.load_node_types()
    if e_mask is None or i_mask is None:
        raise ValueError("e_mask and i_mask must be provided explicitly")

    attempts = []
    starts = [(0.15, 0.15)] + _ALT_STARTS
    for start_e, start_i in starts:
        try:
            result = _fit_weights_ei_cond_single_start(
                edge_file, tag, target_hz_e, target_hz_i, N_neurons,
                node_types, e_mask, i_mask, start_e, start_i, verbose=verbose)
        except RuntimeError as e:
            if verbose:
                print(f"    [{tag}] start E={start_e:.4f},I={start_i:.4f} "
                      f"did not settle: {e}")
            continue
        w_e, w_i, hz_e, hz_i, stable, ctype = result
        if ctype == "full":
            if verbose:
                print(f"    [{tag}] full convergence from start "
                      f"E={start_e:.4f},I={start_i:.4f} -- stopping search")
            return result
        dist = (abs(hz_e - target_hz_e) / target_hz_e
               + abs(hz_i - target_hz_i) / target_hz_i)
        attempts.append(result + (dist,))

    if not attempts:
        raise RuntimeError(
            f"no starting point (of {len(starts)} tried) reached either "
            f"full tolerance or a stable fixed point -- worth checking "
            f"whether the [0.01, 0.5] bracket genuinely contains a viable "
            f"solution before widening it blindly")

    best = min(attempts, key=lambda a: a[-1])
    if verbose:
        print(f"    [{tag}] no start reached full tolerance; using the "
              f"closest fixed point found across {len(attempts)} start(s) "
              f"tried: E={best[0]:.4f} I={best[1]:.4f}")
    return best[:6]


def _joint_objective(hz_e, hz_i, target_hz_e, target_hz_i,
                     e_weight=1.0, i_weight=1.0):
    """Weighted combined rate error, used by the joint-fit search below
    instead of two independent per-axis targets. e_weight/i_weight let E's
    error dominate the search once I is already close to acceptable --
    the whole point is allowing I to sit away from ITS OWN target if
    that's what jointly minimizing both errors requires, which strict
    per-axis alternation (fit_weights_ei_cond above) cannot do by
    construction.
    """
    e_err = abs(hz_e - target_hz_e) / target_hz_e
    i_err = abs(hz_i - target_hz_i) / target_hz_i
    return e_weight * e_err + i_weight * i_err


def fit_weights_ei_cond_joint(edge_file, tag, target_hz_e=TARGET_HZ_E,
                              target_hz_i=TARGET_HZ_I, N_neurons=2196,
                              node_types=None, e_mask=None, i_mask=None,
                              n_grid=12, i_weight_grid=(0.5, 1.0, 2.0, 4.0),
                              verbose=True):
    """CONFIRMED NEEDED: fit_weights_ei_cond() above, run to completion
    from 4 independent starting points, converged to the IDENTICAL fixed
    point every time (w_e=0.2550, w_i=0.8762, hz_e=76.62, hz_i=1.91) --
    not a search failure, a genuine structural finding. Strict per-axis
    alternation fits I to ITS OWN isolated target every round; the I
    weight that satisfies I's target in isolation is not enough
    recurrent inhibition to bring E anywhere near its own target. This
    function tests directly whether that's a hard ceiling of the model
    or an artifact of forcing I to always sit at its own isolated
    optimum: it lets I trade away from its own target, scored by
    i_weight_grid (how much I's own error is allowed to cost against the
    combined objective), and searches E and a scaled I together rather
    than alternating strict single-axis fits.

    Strategy: for each i_weight in i_weight_grid (lower i_weight = more
    permission for I to overshoot), do a 2D grid search over
    (w_e, w_i_scale) where w_i_scale multiplies the I weight found by
    the E-starved single-axis fit above -- i.e. explore ABOVE 0.8762,
    not just at it, since more inhibition than I's own isolated optimum
    is the direction untested so far. For each i_weight, report the
    best (lowest combined-objective) point found. This is a coarse grid,
    not another bisection -- appropriate for a first pass confirming
    whether ANY point in this expanded region beats the confirmed fixed
    point, before investing in a finer search.
    """
    if node_types is None:
        node_types = step2_ei.load_node_types()
    if e_mask is None or i_mask is None:
        raise ValueError("e_mask and i_mask must be provided explicitly")

    # E range: same bracket as the single-axis search, since E's own
    # viability floor (I-starving E) is unrelated to this question
    e_values = np.linspace(WEIGHT_LO_INIT, WEIGHT_HI_INIT, n_grid)
    # I range: START at the confirmed single-axis optimum (0.8762) and
    # go UP from there -- this is the untested direction. Going below it
    # only returns toward I's own under-target region, not useful here.
    i_base = 0.8762
    i_values = i_base * np.linspace(1.0, 4.0, n_grid)

    print(f"=== Joint-objective search: does MORE inhibition than I's own "
          f"isolated optimum ({i_base:.4f}) bring E toward target, even "
          f"at the cost of overshooting I's own target? ===")
    print(f"E grid: {n_grid} points in [{WEIGHT_LO_INIT}, {WEIGHT_HI_INIT}]")
    print(f"I grid: {n_grid} points in [{i_values[0]:.4f}, {i_values[-1]:.4f}]")
    print(f"Total probes: up to {len(e_values) * len(i_values)}\n")

    results = []
    for w_i in i_values:
        for w_e in e_values:
            hz_e, hz_i, stable = probe_mean_rates_ei_cond(
                edge_file, tag, w_e, w_i, N_neurons, node_types, e_mask, i_mask)
            results.append((w_e, w_i, hz_e, hz_i, stable))
            if verbose:
                e_ok = abs(hz_e - target_hz_e) <= RATE_TOLERANCE * target_hz_e
                i_ok = abs(hz_i - target_hz_i) <= RATE_TOLERANCE * target_hz_i
                flag = " <-- BOTH IN TOLERANCE" if (e_ok and i_ok and stable) else ""
                print(f"  w_e={w_e:.4f}  w_i={w_i:.4f}  hz_e={hz_e:6.2f} "
                      f"(target {target_hz_e:.2f})  hz_i={hz_i:6.2f} "
                      f"(target {target_hz_i:.2f})  stable={stable}{flag}")
            if (abs(hz_e - target_hz_e) <= RATE_TOLERANCE * target_hz_e
                    and abs(hz_i - target_hz_i) <= RATE_TOLERANCE * target_hz_i
                    and stable):
                print(f"\n=== FOUND a point in full tolerance on both axes "
                      f"-- the fixed point was an artifact of strict "
                      f"alternation, not a hard ceiling ===")
                return w_e, w_i, hz_e, hz_i, True, "joint_full"

    print(f"\n=== No grid point reached full tolerance on both axes. "
          f"Reporting the best (lowest combined error) point found, and "
          f"whether it beats the single-axis fixed point on E specifically "
          f"(hz_e=76.62), which is the axis that was stuck ===")
    for i_weight in i_weight_grid:
        scored = [(r, _joint_objective(r[2], r[3], target_hz_e, target_hz_i,
                                       e_weight=1.0, i_weight=i_weight))
                 for r in results if r[4]]  # stable only
        if not scored:
            continue
        best_r, best_score = min(scored, key=lambda x: x[1])
        w_e, w_i, hz_e, hz_i, stable = best_r
        print(f"  i_weight={i_weight}: best point w_e={w_e:.4f} w_i={w_i:.4f} "
              f"hz_e={hz_e:.2f} hz_i={hz_i:.2f}  (combined score={best_score:.4f})")

    best_overall = min(
        [(r, abs(r[2] - target_hz_e)) for r in results if r[4]],
        key=lambda x: x[1])
    w_e, w_i, hz_e, hz_i, stable = best_overall[0]
    print(f"\nBest E-rate specifically (regardless of I): w_e={w_e:.4f} "
          f"w_i={w_i:.4f} hz_e={hz_e:.2f} hz_i={hz_i:.2f}")
    if hz_e < 76.62:
        print(f"IMPROVEMENT over the single-axis fixed point (76.62 -> "
              f"{hz_e:.2f}) -- more inhibition than I's own isolated "
              f"optimum DOES help E, confirming this was an artifact of "
              f"strict per-axis alternation, not a hard ceiling.")
    else:
        print(f"NO improvement over the single-axis fixed point -- this "
              f"grid did not find a better point either. Worth extending "
              f"the I range further, or reconsidering whether these "
              f"targets (carried over from the current-based model) are "
              f"the right anchors for this architecture at all.")
    return w_e, w_i, hz_e, hz_i, stable, "joint_best_effort"


def line_search_extension(edge_file, tag, target_hz_e=TARGET_HZ_E,
                          target_hz_i=TARGET_HZ_I, N_neurons=2196,
                          node_types=None, e_mask=None, i_mask=None,
                          verbose=True):
    """CHEAP DIAGNOSTIC, run BEFORE any wider grid: the previous joint grid
    (fit_weights_ei_cond_joint) hit its ceiling on BOTH axes simultaneously
    at w_e=0.5, w_i=3.5048, while STILL improving (hz_e: 76.62 -> 60.36) --
    so this is confirmed not a converged answer, but blindly doubling both
    grid ranges risks wasting a large grid on the wrong direction, because
    the E-axis result at the boundary was non-monotonic in a way worth
    understanding first: at fixed high w_i, INCREASING w_e DECREASED hz_e
    (0.3664->64.89, 0.4109->63.14, 0.5->60.36) -- the opposite of what a
    purely excitatory weight would do in isolation. Since w_e scales BOTH
    E->E and E->I synapses (same weight, no separate parameter), raising
    it plausibly recruits more indirect inhibition via stronger E->I
    drive, net-suppressing E once the recurrent loop settles -- a real
    coupled effect, not noise, and it means "more E, more I" might not be
    the same productive direction on both axes.

    Runs two SEPARATE 1D line searches, each cheap (aim for ~8 probes,
    not a 12x12 grid):
      (1) fix w_i at the confirmed-best 3.5048, extend w_e PAST 0.5
      (2) fix w_e at the confirmed-best 0.5, extend w_i PAST 3.5048
    Reports whether hz_e keeps improving, plateaus, or reverses in each
    direction -- this determines where a wider 2D grid should actually
    focus, rather than guessing.
    """
    if node_types is None:
        node_types = step2_ei.load_node_types()
    if e_mask is None or i_mask is None:
        raise ValueError("e_mask and i_mask must be provided explicitly")

    w_i_fixed = 3.5048   # confirmed best from the previous grid
    w_e_fixed = 0.5000   # confirmed best from the previous grid

    print(f"=== Line search 1: fix w_i={w_i_fixed}, extend w_e PAST the "
          f"previous grid's ceiling (0.5) ===")
    e_extension = [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0]
    e_results = []
    for w_e in e_extension:
        hz_e, hz_i, stable = probe_mean_rates_ei_cond(
            edge_file, tag, w_e, w_i_fixed, N_neurons, node_types, e_mask, i_mask)
        e_results.append((w_e, hz_e, hz_i, stable))
        if verbose:
            print(f"  w_e={w_e:.4f}  hz_e={hz_e:6.2f} (target {target_hz_e:.2f})  "
                  f"hz_i={hz_i:6.2f} (target {target_hz_i:.2f})  stable={stable}")

    print(f"\n=== Line search 2: fix w_e={w_e_fixed}, extend w_i PAST the "
          f"previous grid's ceiling (3.5048) ===")
    i_base = 0.8762
    i_extension = [i_base * m for m in [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]]
    i_results = []
    for w_i in i_extension:
        hz_e, hz_i, stable = probe_mean_rates_ei_cond(
            edge_file, tag, w_e_fixed, w_i, N_neurons, node_types, e_mask, i_mask)
        i_results.append((w_i, hz_e, hz_i, stable))
        if verbose:
            print(f"  w_i={w_i:.4f}  hz_e={hz_e:6.2f} (target {target_hz_e:.2f})  "
                  f"hz_i={hz_i:6.2f} (target {target_hz_i:.2f})  stable={stable}")

    print(f"\n=== Interpretation ===")
    e_hz_trend = [r[1] for r in e_results]
    i_hz_trend = [r[1] for r in i_results]
    e_still_improving = e_hz_trend[-1] < e_hz_trend[0] - 1.0
    i_still_improving = i_hz_trend[-1] < i_hz_trend[0] - 1.0
    e_best = min(e_results, key=lambda r: abs(r[1] - target_hz_e))
    i_best = min(i_results, key=lambda r: abs(r[1] - target_hz_e))

    print(f"  E-direction (fixed w_i={w_i_fixed}): hz_e went from "
          f"{e_hz_trend[0]:.2f} to {e_hz_trend[-1]:.2f} across the extension. "
          f"{'STILL IMPROVING -- extend w_e further' if e_still_improving else 'PLATEAUING OR REVERSING -- this direction is close to exhausted'}")
    print(f"  best point in this line: w_e={e_best[0]:.4f}  hz_e={e_best[1]:.2f}  "
          f"hz_i={e_best[2]:.2f}")
    print(f"  I-direction (fixed w_e={w_e_fixed}): hz_e went from "
          f"{i_hz_trend[0]:.2f} to {i_hz_trend[-1]:.2f} across the extension. "
          f"{'STILL IMPROVING -- extend w_i further' if i_still_improving else 'PLATEAUING OR REVERSING -- this direction is close to exhausted'}")
    print(f"  best point in this line: w_i={i_best[0]:.4f}  hz_e={i_best[1]:.2f}  "
          f"hz_i={i_best[2]:.2f}")

    print(f"\n  Recommended next step: ", end="")
    if e_still_improving and i_still_improving:
        print("BOTH directions still productive -- run a wider 2D grid "
              "spanning both extended ranges, not just one.")
    elif e_still_improving:
        print("only the E direction is still productive -- extend w_e "
              "further at high w_i, no need to widen the w_i range further.")
    elif i_still_improving:
        print("only the I direction is still productive -- extend w_i "
              "further at high w_e, no need to widen the w_e range further.")
    else:
        print("NEITHER direction is meaningfully improving anymore -- this "
              "may be close to the actual best achievable region for this "
              "weight structure. Worth checking whether hz_e can plausibly "
              "reach 12.04 Hz AT ALL under uniform-per-class conductance "
              "weights, or whether that requires per-cell-type or "
              "per-synapse variation, similar to what was already tested "
              "(and found unhelpful) for the current-based model.")

    return {"e_line": e_results, "i_line": i_results,
           "e_best": e_best, "i_best": i_best}


def refined_asymmetric_grid(edge_file, tag, target_hz_e=TARGET_HZ_E,
                            target_hz_i=TARGET_HZ_I, N_neurons=2196,
                            node_types=None, e_mask=None, i_mask=None,
                            verbose=True):
    """NOT a naive "widen both ranges" grid, deliberately -- the line
    search extension showed the two directions are NOT equally
    productive, contrary to what a surface read of "both still
    improving" suggests. The E-direction's improvement (hz_e: 60.36 ->
    47.12 at w_e=2.0) came at the cost of hz_i blowing out to 12.78 --
    nearly 7x over target, almost as high as E's own target rate. Since
    w_e drives both E->E and E->I synapses, pushing it up recruits more
    indirect inhibition faster than it can be tolerated. The I-direction
    (hz_e: 60.36 -> 53.66 at w_i=14.02, hz_i=2.16, only 14% over target)
    is smaller in absolute hz_e terms but doesn't sacrifice the other
    axis -- the genuinely productive direction.

    This grid reflects that asymmetry directly: w_e kept NARROW (staying
    clear of the runaway-inhibition-recruitment zone above ~0.8), w_i
    extended FAR (since that's where clean progress was actually found).
    """
    if node_types is None:
        node_types = step2_ei.load_node_types()
    if e_mask is None or i_mask is None:
        raise ValueError("e_mask and i_mask must be provided explicitly")

    e_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    i_base = 0.8762
    i_values = [i_base * m for m in [4, 6, 8, 10, 14, 18, 24, 30]]

    print(f"=== Refined asymmetric grid: narrow w_e (avoiding the runaway-"
          f"inhibition zone), wide w_i (the confirmed clean direction) ===")
    print(f"w_e: {e_values}")
    print(f"w_i: {[round(v, 3) for v in i_values]}")
    print(f"Total probes: {len(e_values) * len(i_values)}\n")

    results = []
    for w_i in i_values:
        for w_e in e_values:
            hz_e, hz_i, stable = probe_mean_rates_ei_cond(
                edge_file, tag, w_e, w_i, N_neurons, node_types, e_mask, i_mask)
            results.append((w_e, w_i, hz_e, hz_i, stable))
            if verbose:
                e_ok = abs(hz_e - target_hz_e) <= RATE_TOLERANCE * target_hz_e
                i_ok = abs(hz_i - target_hz_i) <= RATE_TOLERANCE * target_hz_i
                flag = " <-- BOTH IN TOLERANCE" if (e_ok and i_ok and stable) else ""
                print(f"  w_e={w_e:.4f}  w_i={w_i:.4f}  hz_e={hz_e:6.2f} "
                      f"(target {target_hz_e:.2f})  hz_i={hz_i:6.2f} "
                      f"(target {target_hz_i:.2f})  stable={stable}{flag}")
            if (abs(hz_e - target_hz_e) <= RATE_TOLERANCE * target_hz_e
                    and abs(hz_i - target_hz_i) <= RATE_TOLERANCE * target_hz_i
                    and stable):
                print(f"\n=== FOUND a point in full tolerance on both axes ===")
                return w_e, w_i, hz_e, hz_i, True, "asymmetric_full"

    print(f"\n=== No point reached full tolerance. Reporting the best "
          f"point that keeps hz_i within a reasonable band of target "
          f"(not just the lowest hz_e regardless of I, which is how the "
          f"E-direction's misleading 'improvement' arose last time) ===")
    # only consider points where hz_i isn't wildly overshot -- same
    # discipline that caught the E-direction problem
    i_reasonable = [r for r in results if r[4] and r[3] <= target_hz_i * 2.0]
    if i_reasonable:
        best = min(i_reasonable, key=lambda r: abs(r[2] - target_hz_e))
        w_e, w_i, hz_e, hz_i, stable = best
        print(f"Best point with hz_i within 2x target: w_e={w_e:.4f} "
              f"w_i={w_i:.4f}  hz_e={hz_e:.2f}  hz_i={hz_i:.2f}")
    else:
        print(f"No point kept hz_i within 2x target anywhere in this grid "
              f"-- worth knowing directly, not papering over.")
        best = min([r for r in results if r[4]], key=lambda r: abs(r[2] - target_hz_e))
        w_e, w_i, hz_e, hz_i, stable = best
        print(f"Best hz_e regardless of I (same caveat as before applies): "
              f"w_e={w_e:.4f} w_i={w_i:.4f}  hz_e={hz_e:.2f}  hz_i={hz_i:.2f}")

    return w_e, w_i, hz_e, hz_i, stable, "asymmetric_best_effort"


def confirm_ceiling_line_search(edge_file, tag, target_hz_e=TARGET_HZ_E,
                                target_hz_i=TARGET_HZ_I, N_neurons=2196,
                                node_types=None, e_mask=None, i_mask=None,
                                verbose=True):
    """Cheap, targeted follow-up to the asymmetric grid, which hit its
    ceiling at w_i=26.286 (hz_e=48.33) while showing shrinking returns:
    hz_e dropped 16.26 (w_i 0.88->3.50), then 6.70 (3.50->14.02), then
    5.33 (14.02->26.29) -- each ~4x jump in w_i buying a smaller
    reduction. This extends w_i further at the confirmed-good w_e=0.6,
    with smaller, evenly-spaced steps (not another big jump) specifically
    to read the SHAPE of the remaining curve: does the delta-per-step
    keep shrinking toward zero (real ceiling, stop searching) or stay
    substantial (worth another real grid)?
    """
    if node_types is None:
        node_types = step2_ei.load_node_types()
    if e_mask is None or i_mask is None:
        raise ValueError("e_mask and i_mask must be provided explicitly")

    w_e_fixed = 0.6000   # confirmed best point from the asymmetric grid
    w_i_values = [26.286, 35.0, 45.0, 60.0, 80.0]

    print(f"=== Ceiling-confirmation line search: fix w_e={w_e_fixed}, "
          f"extend w_i past the asymmetric grid's ceiling (26.286) with "
          f"smaller steps to read the curve's shape ===\n")

    results = []
    for w_i in w_i_values:
        hz_e, hz_i, stable = probe_mean_rates_ei_cond(
            edge_file, tag, w_e_fixed, w_i, N_neurons, node_types, e_mask, i_mask)
        results.append((w_i, hz_e, hz_i, stable))
        if verbose:
            print(f"  w_i={w_i:7.3f}  hz_e={hz_e:6.2f} (target {target_hz_e:.2f})  "
                  f"hz_i={hz_i:6.2f} (target {target_hz_i:.2f})  stable={stable}")

    print(f"\n=== Delta analysis: is hz_e's decrease per step shrinking "
          f"toward zero (ceiling) or staying substantial (still productive)? ===")
    deltas = []
    for i in range(1, len(results)):
        prev_hz_e = results[i-1][1]
        curr_hz_e = results[i][1]
        delta = prev_hz_e - curr_hz_e
        deltas.append(delta)
        print(f"  w_i {results[i-1][0]:.1f} -> {results[i][0]:.1f}: "
              f"hz_e {prev_hz_e:.2f} -> {curr_hz_e:.2f}  (delta={delta:+.2f})")

    print(f"\n=== Verdict ===")
    if all(d < 1.0 for d in deltas):
        print(f"  All deltas under 1.0 Hz -- this is a real ceiling, not a "
              f"search artifact. Further w_i increases are very unlikely "
              f"to meaningfully close the gap to {target_hz_e} Hz. "
              f"Recommend treating this as a genuine negative result: "
              f"uniform-per-class conductance weights cannot jointly hit "
              f"both targets in this region, the conductance-based analog "
              f"of the cell-type pilot's negative finding on the "
              f"current-based model.")
    elif deltas[-1] < deltas[0] * 0.3:
        print(f"  Deltas are shrinking substantially (from {deltas[0]:.2f} "
              f"to {deltas[-1]:.2f}) but not yet under 1.0 -- approaching "
              f"a ceiling but not fully there. One more, larger step "
              f"(e.g. w_i=120-150) would likely confirm whether it fully "
              f"flattens.")
    else:
        print(f"  Deltas are NOT shrinking substantially (from "
              f"{deltas[0]:.2f} to {deltas[-1]:.2f}) -- still genuinely "
              f"productive, contrary to the ceiling hypothesis. Worth "
              f"another real grid extending w_i further, this direction "
              f"has not exhausted itself.")

    final_hz_e = results[-1][1]
    final_hz_i = results[-1][2]
    gap_e = final_hz_e - target_hz_e
    print(f"\n  Remaining gap at w_i={w_i_values[-1]}: hz_e is "
          f"{gap_e:+.2f} Hz from target ({final_hz_e:.2f} vs {target_hz_e:.2f}), "
          f"hz_i is {'within' if abs(final_hz_i - target_hz_i) <= 0.2*target_hz_i else 'outside'} "
          f"a reasonable band of its own target ({final_hz_i:.2f} vs {target_hz_i:.2f}).")

    return results


if __name__ == "__main__":
    import pandas as pd
    from step7_ei_fit_weights import NODE_CSV

    node_df = pd.read_csv(NODE_CSV, index_col="node_id")
    node_types = step2_ei.load_node_types()
    N_neurons = len(node_df)
    node_id_to_nucleus = node_df['nucleus_id'].to_dict()
    is_I = np.array([node_types.get(node_id_to_nucleus[i]) == 'I'
                     for i in range(N_neurons)])

    import sys
    if "--confirm-ceiling" in sys.argv:
        print("=== Ceiling-confirmation line search (reading the shape of "
              "the remaining curve past the asymmetric grid's ceiling) ===\n")
        confirm_ceiling_line_search(
            "signed_real_edges.json", "real_cond_ceiling", N_neurons=N_neurons,
            node_types=node_types, e_mask=~is_I, i_mask=is_I)
    elif "--asymmetric" in sys.argv:
        print("=== Refined asymmetric grid (narrow w_e, wide w_i, based on "
              "the line-search finding that only the I-direction is "
              "genuinely productive) ===\n")
        result = refined_asymmetric_grid(
            "signed_real_edges.json", "real_cond_asym", N_neurons=N_neurons,
            node_types=node_types, e_mask=~is_I, i_mask=is_I)
        w_e, w_i, hz_e, hz_i, stable, ctype = result
        print(f"\n=== Final result ===")
        print(f"w_e={w_e:.4f}  w_i={w_i:.4f}  hz_e={hz_e:.2f}  hz_i={hz_i:.2f}  "
              f"stable={stable}  converged_type={ctype}")
    elif "--extend" in sys.argv:
        print("=== Line-search extension (tracing each axis's real shape "
              "before committing to a wider 2D grid) ===\n")
        line_search_extension(
            "signed_real_edges.json", "real_cond_extend", N_neurons=N_neurons,
            node_types=node_types, e_mask=~is_I, i_mask=is_I)
    elif "--joint" in sys.argv:
        print("=== Joint-objective search (confirming/refuting the "
              "single-axis fixed point) ===\n")
        result = fit_weights_ei_cond_joint(
            "signed_real_edges.json", "real_cond_joint", N_neurons=N_neurons,
            node_types=node_types, e_mask=~is_I, i_mask=is_I)
        w_e, w_i, hz_e, hz_i, stable, ctype = result
        print(f"\n=== Final result ===")
        print(f"w_e={w_e:.4f}  w_i={w_i:.4f}  hz_e={hz_e:.2f}  hz_i={hz_i:.2f}  "
              f"stable={stable}  converged_type={ctype}")
    else:
        print("=== Fitting conductance-based model to real connectome ===")
        print(f"Targets: E={TARGET_HZ_E} Hz, I={TARGET_HZ_I} Hz  "
              f"(same anchors as the current-based model)")
        print(f"Bracket: [{WEIGHT_LO_INIT}, {WEIGHT_HI_INIT}] nS "
              f"(narrowed from the validation sweep, not the current-based "
              f"model's [2, 15] -- different physical units)\n")

        result = fit_weights_ei_cond(
            "signed_real_edges.json", "real_cond", N_neurons=N_neurons,
            node_types=node_types, e_mask=~is_I, i_mask=is_I)
        w_e, w_i, hz_e, hz_i, stable, ctype = result
        print(f"\n=== Result ===")
        print(f"w_e={w_e:.4f}  w_i={w_i:.4f}  hz_e={hz_e:.2f}  hz_i={hz_i:.2f}  "
              f"stable={stable}  converged_type={ctype}")
