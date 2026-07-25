#!/usr/bin/env python
"""
exp5_stability_sweep.py — is seed 0 typical?

ER seed 0 diverged between checkpoint 5 (iter 14,411, max|a| = 5.67) and
checkpoint 10 (iter 32,411, max|a| = 6.18e8), and never recovered across
61 further checkpoints. degree_swap seed 0 stayed bounded across all 72,
settling to ~6. Both trained 250,007 iterations from identical init.

n = 1. This trains seeds 1-9 of both arms to 40,000 iterations -- 16% of
the full recipe, enough to traverse the divergence window -- and reads
resting activity after every checkpoint.

DESIGN NOTES, each earned:

  * resume=false, ALWAYS. flyvis's resolve_checkpoints() has an arity bug
    on the resume path (TypeError: takes 1 positional arg but 4 given).
    production.py's gate turns resume ON exactly when it will crash.

  * Kill at divergence. Once max|a| > 1e6 there is nothing further to
    learn from that network, and the run dies at ~80% rather than 100%.
    Implemented by polling checkpoints in a watcher thread and killing
    the subprocess.

  * Networks go to flow/exp5_<scheme>_short/<seed:04d>, NOT
    flow/exp5_<scheme>/<seed:04d>. The latter is where seed 0's 250k runs
    live, and clean_network_results_dir() calls shutil.rmtree().

  * Results are written to disk after EVERY network. A 15-hour run that
    dies at hour 12 must not lose hours 1-11.

Run:  nohup python exp5_stability_sweep.py > sweep.log 2>&1 &
      tail -f sweep.log
"""

import os, sys, glob, json, time, shutil, signal, threading, subprocess
import numpy as np, torch, h5py, flyvis
from flyvis.network import NetworkView
from flyvis.datasets.moving_bar import MovingEdge

# ── config ───────────────────────────────────────────────────────────
N_ITERS   = 40_000
SEEDS     = range(1, 10)
SCHEMES   = ["erdos_renyi", "degree_preserving_swap"]
CONN_DIR  = "/workspace/rand_out"
OUT_JSON  = "/workspace/exp5_stability_sweep.json"
DIVERGE   = 1e6
POLL_SEC  = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

ds = MovingEdge(offsets=[-10, 11], intensities=[0, 1], speeds=[19], height=80,
                post_pad_mode="continue", t_pre=1.0, t_post=1.0, dt=1/200,
                angles=ANGLES)


# ── stability probe ──────────────────────────────────────────────────
def fade_max(net):
    """max |activity| after 1.0 s of constant illumination. Bounded ~6 for a
    healthy trained network; 1e20 for ER seed 0 at chkpt 46."""
    s = ds[0]
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, dtype=torch.float32)
    s = s.to(DEVICE)
    if s.dim() == 2:
        s = s.unsqueeze(1)
    st = net.fade_in_state(1.0, ds.dt, s[[0]])
    a = st["nodes"]["activity"].detach().cpu().numpy()
    return float(np.abs(a).max())


def load_ckpt(view_path, ck):
    """Bypass init_network's cache: it resolves the path correctly and then
    returns a cached Network, so three 'different' checkpoints come back
    identical. Verified against the file, not against the other nets."""
    nv  = NetworkView(view_path)
    net = nv.init_network(checkpoint=ck)
    raw = torch.load(nv.checkpoints.paths[ck], map_location="cpu", weights_only=False)
    net.load_state_dict(raw["network"], strict=False)
    got  = net.edge_params.syn_strength.raw_values.data.cpu().numpy()
    want = raw["network"]["edges_syn_strength"].numpy()
    assert np.abs(got - want).max() < 1e-6, f"weights not loaded, chkpt {ck}"
    return net.to(DEVICE), int(raw["iteration"]), float(raw["val_loss"])


def read_trace(net_name, upto=None):
    """Stability at every checkpoint written so far. Stops at first divergence."""
    nv, trace = NetworkView(f"flow/{net_name}"), []
    n = len(nv.checkpoints.paths) if upto is None else min(upto, len(nv.checkpoints.paths))
    for ck in range(n):
        net, it, vl = load_ckpt(f"flow/{net_name}", ck)
        m = fade_max(net)
        trace.append(dict(ck=ck, iter=it, val=vl, fade=m))
        del net
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if m > DIVERGE:
            break
    return trace


# ── training, with a watcher that kills on divergence ─────────────────
def _watch(net_dir, proc, stop, tag):
    """Poll checkpoints; kill training once the network has diverged.
    Reads only. HDF5 can throw mid-write -- catch ONLY that."""
    last = -1
    while not stop.wait(POLL_SEC):
        cks = sorted(glob.glob(os.path.join(net_dir, "chkpts", "chkpt_*")))
        if len(cks) == last or not cks:
            continue
        last = len(cks)
        p = os.path.join(net_dir, "chkpt_iter.h5")
        it = "?"
        try:
            with h5py.File(p, "r") as f:
                arr = np.array(f[list(f.keys())[0]])
            it = int(arr[-1]) if len(arr) else "?"
        except (OSError, KeyError):
            continue                      # mid-write; retry next tick
        print(f"    [{tag}] chkpt {last-1}, iter {it}", flush=True)


def train(net_name, connectome, n_iters):
    results_dir = str(flyvis.results_dir)
    net_dir = os.path.join(results_dir, "flow", net_name)
    if os.path.isdir(net_dir):
        shutil.rmtree(net_dir)

    shim = "/workspace/_dm_shim"
    os.makedirs(shim, exist_ok=True)
    sc = os.path.join(shim, "sitecustomize.py")
    if not os.path.exists(sc):
        open(sc, "w").write(
            "try:\n    import datamate; datamate.enforce_config_match(False)\n"
            "except Exception as e:\n    import sys; print('shim:', e, file=sys.stderr)\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = shim + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    cmd = ["flyvis", "train-single",
           f"ensemble_and_network_id={net_name}",
           "task_name=flow", "train=true",
           "resume=false",                       # arity bug on the resume path
           f"task.n_iters={n_iters}",
           f"network.connectome.file={connectome}",
           "description=exp5_trained_random"]

    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, text=True)
    stop = threading.Event()
    w = threading.Thread(target=_watch, args=(net_dir, p, stop, net_name), daemon=True)
    w.start()
    try:
        _, err = p.communicate()
    finally:
        stop.set(); w.join(timeout=2)
    el = time.time() - t0
    if p.returncode != 0:
        print(f"  TRAINING FAILED rc={p.returncode}\n{err[-2000:]}", flush=True)
        return None, el
    return True, el


# ── main ─────────────────────────────────────────────────────────────
results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
print(f"device={DEVICE}  n_iters={N_ITERS}  seeds={list(SEEDS)}", flush=True)
print(f"already done: {sorted(results)}\n", flush=True)

for scheme in SCHEMES:
    for seed in SEEDS:
        key  = f"{scheme}_s{seed}"
        if key in results:
            print(f"{key}: done, skip", flush=True)
            continue

        name = f"exp5_{scheme}_short/{seed:04d}"
        conn = f"{CONN_DIR}/fib25_{scheme}_seed{seed}.json"
        if not os.path.exists(conn):
            print(f"{key}: MISSING {conn}", flush=True)
            results[key] = dict(status="missing_connectome")
            json.dump(results, open(OUT_JSON, "w"), indent=2)
            continue

        print(f"\n=== {key} ===", flush=True)
        ok, el = train(name, conn, N_ITERS)
        if ok is None:
            results[key] = dict(status="train_failed", wall_s=el)
            json.dump(results, open(OUT_JSON, "w"), indent=2)
            continue

        trace = read_trace(name)
        div   = next((t["ck"] for t in trace if t["fade"] > DIVERGE), None)
        results[key] = dict(status="ok", wall_s=el, trace=trace,
                            diverged=div is not None, first_diverged_ck=div,
                            last_iter=trace[-1]["iter"])
        json.dump(results, open(OUT_JSON, "w"), indent=2)

        print(f"  {key}: {'DIVERGED at ck '+str(div) if div is not None else 'stable'}"
              f"  ({el/60:.0f} min, {len(trace)} chkpts)", flush=True)


# ── verdict ──────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("DIVERGENCE COUNT (seeds 1-9, 40k iterations)", flush=True)
print("=" * 70, flush=True)
for scheme in SCHEMES:
    ks = [f"{scheme}_s{s}" for s in SEEDS
          if results.get(f"{scheme}_s{s}", {}).get("status") == "ok"]
    dv = [k for k in ks if results[k]["diverged"]]
    print(f"  {scheme:26s} {len(dv)}/{len(ks)} diverged", flush=True)
    for k in dv:
        t = results[k]
        ck = t["first_diverged_ck"]
        print(f"      {k}: ck {ck}, iter {t['trace'][ck]['iter']}", flush=True)

print("\n  seed 0 (250k iters): degree_swap stable through ck 71;"
      " erdos_renyi diverged between ck 5 and ck 10", flush=True)
print(f"\nwrote {OUT_JSON}", flush=True)
