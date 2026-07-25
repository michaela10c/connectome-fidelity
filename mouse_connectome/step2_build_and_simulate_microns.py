"""
Step 2 (parameterized, LGN-shared): build a connectome-constrained PointNet
network and run it across 8 orientations -- with the LGN input scaffold built
ONCE and shared identically across the real connectome and all null connectomes.

WHY THE LGN SPLIT (read before running null comparisons):
    The Option-A experiment compares the simulated geometry of the REAL
    connectome against the simulated geometry of NULL connectomes. For that
    comparison to be valid, the ONLY thing that may differ between conditions
    is the V1->V1 recurrent connectome. Everything feeding the network -- the
    LGN nodes, the precomputed LGN spike trains (step1), and crucially the
    LGN->V1 wiring -- must be byte-identical across real and null runs.

    The original step2 rebuilt the LGN network (and its LGN->V1 edges, via an
    UNSEEDED np.random.randint) on every run. Across real vs null that would
    make the LGN input drive differ between conditions -- an uncontrolled
    confound that defeats the experiment. This version fixes that:

        build_lgn_scaffold()           # run ONCE; seeded; shared by all tags
        run_simulation(edge_file, tag) # builds ONLY V1->V1 recurrent edges,
                                        # reuses the shared LGN scaffold

    Result: real and every null see the IDENTICAL LGN drive; only the recurrent
    connectome changes. That is the control the comparison requires.

REPRODUCTION CHECK (do this first):
    python step2_build_and_simulate_microns.py            # builds LGN + real
    python step3_extract_rates_and_rsa.py                 # expect 0.082 / 0.055
    With USE_LEGACY_DIRS=True the real run writes to network/ + output_pointnet/
    so the existing step3 runs unchanged.

Weighting (unchanged): linear synapse-count-proportional, weight = nsyns *
SYN_WEIGHT_EE via BMTK's native nsyns path.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from bmtk.builder import NetworkBuilder
from bmtk.simulator import pointnet

# -- parameters ----------------------------------------------------------------

ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
INPUT_DIR    = "inputs"
TSTOP        = 3000.0
DT           = 0.1

V1_MODEL_TEMPLATE  = "nest:iaf_psc_alpha"
V1_DYNAMICS_PARAMS = "IntFire1_exc_point.json"

SYN_WEIGHT_EE  = 5.0
SYN_WEIGHT_LGN = 20.0
N_LGN          = 100

WEIGHT_MODE = "linear"      # "linear" (nsyns*SYN_WEIGHT_EE) or "uniform"
USE_LEGACY_DIRS = True      # tag=="real" -> network/ + output_pointnet/

NODE_CSV  = "node_data_v1.csv"

# Shared LGN scaffold location (built once, reused by every tag).
LGN_DIR   = "network_lgn_shared"
# Seed for the LGN->V1 wiring so it is reproducible AND identical across tags.
LGN_SEED  = 42


# -- dir helpers ---------------------------------------------------------------

def network_dir_for(tag):
    if USE_LEGACY_DIRS and tag == "real":
        return "network"
    return f"network_{tag}"

def output_dir_for(tag):
    if USE_LEGACY_DIRS and tag == "real":
        return "output_pointnet"
    return f"output_pointnet_{tag}"


# -- load neurons --------------------------------------------------------------

def load_nodes():
    print(f"Loading neuron data from {NODE_CSV}...")
    node_data = pd.read_csv(NODE_CSV, index_col="node_id")
    node_data = node_data.rename(columns={"pt_root_id": "root_id"})
    print(f"  Loaded {len(node_data)} neurons")
    return node_data


def load_edges(edge_file):
    print(f"Loading edges from {edge_file}...")
    with open(edge_file, "r") as f:
        edges_raw = json.load(f)
    if len(edges_raw) > 0 and len(edges_raw[0]) == 3:
        edges_df = pd.DataFrame(edges_raw,
                                columns=['pre_node_id', 'post_node_id', 'syn_count'])
        print(f"  Edge file includes synapse counts "
              f"(median={int(edges_df.syn_count.median())}, "
              f"max={int(edges_df.syn_count.max())})")
    else:
        edges_df = pd.DataFrame(edges_raw, columns=['pre_node_id', 'post_node_id'])
        edges_df['syn_count'] = 1
        print("  WARNING: edge file has no synapse counts -- weights uniform.")
    print(f"  Loaded {len(edges_df)} directed edges")
    return edges_df


# -- SHARED LGN scaffold (build once) ------------------------------------------

def build_lgn_scaffold(neurons_df, force=False):
    """Build the LGN virtual network and its LGN->V1 wiring ONCE, seeded, and
    save to LGN_DIR. Every tag's circuit config references this same LGN. The
    LGN->V1 connection rule is seeded so the wiring is identical across runs.

    Note: the LGN->V1 edges target V1 node_ids 0..N-1, which are stable because
    V1 nodes are always built from the same node_data_v1.csv in the same order.
    """
    lgn_nodes = os.path.join(LGN_DIR, "lgn_nodes.h5")
    lgn_edges = os.path.join(LGN_DIR, "lgn_microns_v1_edges.h5")
    if (not force) and os.path.exists(lgn_nodes) and os.path.exists(lgn_edges):
        print(f"  Shared LGN scaffold already present in {LGN_DIR}/ "
              f"(use force=True to rebuild).")
        return LGN_DIR

    N = len(neurons_df)
    os.makedirs(LGN_DIR, exist_ok=True)
    print(f"\nBuilding SHARED LGN scaffold -> {LGN_DIR}/  (seed={LGN_SEED})")

    # A V1 node placeholder so LGN->V1 edges have a target population with the
    # right size/order. We rebuild V1 recurrent edges per tag separately; here
    # we only need V1 nodes to anchor the LGN->V1 targets.
    v1 = NetworkBuilder('microns_v1')
    v1.add_nodes(
        N=N, model_type='point_neuron', ei='e',
        model_template=V1_MODEL_TEMPLATE,
        dynamics_params=V1_DYNAMICS_PARAMS,
        x=neurons_df['x'].values,
        y=neurons_df['y'].values,
        z=neurons_df['z'].values,
    )
    # Build/save V1 nodes into LGN_DIR so the LGN->V1 edge file has a consistent
    # companion node file; per-tag runs will re-save identical V1 nodes too.
    v1.build()
    v1.save_nodes(output_dir=LGN_DIR)

    rng = np.random.RandomState(LGN_SEED)
    def lgn_to_v1_rule(src, trg):
        return int(rng.randint(0, 5))

    lgn = NetworkBuilder('lgn')
    lgn.add_nodes(N=N_LGN, model_type='virtual', ei='e')
    lgn.add_edges(
        source=lgn.nodes(), target=v1.nodes(ei='e'),
        connection_rule=lgn_to_v1_rule,
        model_template='static_synapse',
        dynamics_params='LGN_to_LIF.json',
        syn_weight=SYN_WEIGHT_LGN, delay=2.0
    )
    lgn.build()
    lgn.save(output_dir=LGN_DIR)
    print(f"  Shared LGN scaffold saved to {LGN_DIR}/ "
          f"(lgn nodes + LGN->V1 edges, seeded).")
    return LGN_DIR


# -- per-tag V1 recurrent build ------------------------------------------------

def build_v1_recurrent(neurons_df, edges_df, network_dir):
    """Build ONLY the V1 nodes + V1->V1 recurrent edges for one connectome.
    LGN is NOT built here -- it comes from the shared scaffold."""
    N = len(neurons_df)
    os.makedirs(network_dir, exist_ok=True)
    print(f"\nBuilding V1 recurrent network: {N} neurons, {len(edges_df)} edges "
          f"-> {network_dir}/")

    nucleus_to_node = {int(row.nucleus_id): idx
                       for idx, row in neurons_df.iterrows()}

    conn_lookup = {}
    skipped = 0
    for row in edges_df.itertuples(index=False):
        src = nucleus_to_node.get(int(row.pre_node_id))
        trg = nucleus_to_node.get(int(row.post_node_id))
        if src is not None and trg is not None:
            n_syn = int(row.syn_count) if WEIGHT_MODE == "linear" else 1
            conn_lookup[(src, trg)] = max(n_syn, 1)
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} edges with nucleus IDs not in node set")
    print(f"  Edges within neuron set: {len(conn_lookup)}")

    counts = np.array(list(conn_lookup.values()), dtype=float)
    if len(counts):
        eff = counts * SYN_WEIGHT_EE
        med = float(np.median(eff))
        pct = np.percentile(eff, [50, 90, 99, 100])
        print(f"  [weights] mode={WEIGHT_MODE}  nsyns median={np.median(counts):.0f} "
              f"max={counts.max():.0f}  (weight = nsyns * {SYN_WEIGHT_EE})")
        print(f"  [weights] EE weight: median={pct[0]:.2f} p90={pct[1]:.2f} "
              f"p99={pct[2]:.2f} max={pct[3]:.2f} (max/median={pct[3]/med:.1f}x)")

    v1 = NetworkBuilder('microns_v1')
    v1.add_nodes(
        N=N, model_type='point_neuron', ei='e',
        model_template=V1_MODEL_TEMPLATE,
        dynamics_params=V1_DYNAMICS_PARAMS,
        x=neurons_df['x'].values,
        y=neurons_df['y'].values,
        z=neurons_df['z'].values,
    )

    def microns_connection_rule(src, trg):
        return conn_lookup.get((src.node_id, trg.node_id), 0)

    v1.add_edges(
        source={'ei': 'e'}, target={'ei': 'e'},
        connection_rule=microns_connection_rule,
        model_template='static_synapse',
        dynamics_params='instantaneousExc.json',
        syn_weight=SYN_WEIGHT_EE, delay=2.0
    )
    print(f"  Building V1 recurrent ({N}x{N} = {N*N:,} pairs)...")
    v1.build()
    v1.save(output_dir=network_dir)
    print("  V1 recurrent saved.")
    return v1


# -- circuit config: V1 (per-tag) + shared LGN ---------------------------------

def make_circuit_config(tag, network_dir, lgn_dir):
    """Circuit references the per-tag V1 network AND the shared LGN scaffold."""
    config = {
        "manifest": {
            "$BASE_DIR":    ".",
            "$NETWORK_DIR": network_dir,
            "$LGN_DIR":     lgn_dir,
            "$MODELS_DIR":  "../bmtk/examples/point_components"
        },
        "components": {
            "point_neuron_models_dir": "$MODELS_DIR/cell_models",
            "synaptic_models_dir":     "$MODELS_DIR/synaptic_models"
        },
        "networks": {
            "nodes": [
                {"nodes_file":      "$NETWORK_DIR/microns_v1_nodes.h5",
                 "node_types_file": "$NETWORK_DIR/microns_v1_node_types.csv"},
                {"nodes_file":      "$LGN_DIR/lgn_nodes.h5",
                 "node_types_file": "$LGN_DIR/lgn_node_types.csv"}
            ],
            "edges": [
                {"edges_file":      "$NETWORK_DIR/microns_v1_microns_v1_edges.h5",
                 "edge_types_file": "$NETWORK_DIR/microns_v1_microns_v1_edge_types.csv"},
                {"edges_file":      "$LGN_DIR/lgn_microns_v1_edges.h5",
                 "edge_types_file": "$LGN_DIR/lgn_microns_v1_edge_types.csv"}
            ]
        }
    }
    path = f"config.circuit.{tag}.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  {path} written (V1={network_dir}, LGN={lgn_dir})")
    return path


def make_simulation_config_for_theta(theta, tag, output_dir, circuit_config_path):
    theta_output_dir = os.path.join(output_dir, f"theta{theta:03d}")
    os.makedirs(theta_output_dir, exist_ok=True)
    config = {
        "manifest": {"$BASE_DIR": ".", "$OUTPUT_DIR": theta_output_dir,
                     "$INPUT_DIR": INPUT_DIR},
        "target_simulator": "NEST",
        "run": {"tstop": TSTOP, "dt": DT},
        "inputs": {
            "LGN_spikes": {
                "input_type": "spikes", "module": "sonata",
                "input_file": os.path.join(INPUT_DIR, f"lgn_spikes_theta{theta:03d}.h5"),
                "node_set": "lgn"
            }
        },
        "output": {
            "log_file": "log.txt", "output_dir": theta_output_dir,
            "overwrite_output_dir": True,
            "spikes_file": "spikes.h5", "spikes_file_csv": "spikes.csv"
        },
        "network": circuit_config_path
    }
    config_path = f"config_simulation_{tag}_theta{theta:03d}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# -- run PointNet (subprocess per orientation) ---------------------------------

def run_pointnet_single(theta, tag, output_dir, circuit_config_path):
    import subprocess
    config_path = make_simulation_config_for_theta(theta, tag, output_dir,
                                                   circuit_config_path)
    run_script = f"run_pointnet_{tag}_theta{theta:03d}.py"
    with open(run_script, "w") as f:
        f.write(
            "import sys\n"
            "from bmtk.simulator import pointnet\n"
            f"configure = pointnet.Config.from_json(\"{config_path}\")\n"
            "configure.build_env()\n"
            "graph = pointnet.PointNetwork.from_config(configure)\n"
            "sim = pointnet.PointSimulator.from_config(configure, graph)\n"
            "sim.run()\n"
        )
    result = subprocess.run([sys.executable, run_script], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"PointNet failed for tag={tag} theta={theta}")


def run_all_orientations(tag, output_dir, circuit_config_path):
    for theta in ORIENTATIONS:
        print(f"\n--- [{tag}] theta={theta}deg (fresh NEST kernel) ---")
        run_pointnet_single(theta, tag, output_dir, circuit_config_path)
        print(f"    [{tag}] theta={theta}deg done.")


# -- sanity check --------------------------------------------------------------

def check_outputs(N, output_dir):
    import h5py
    print(f"\nSanity check [{output_dir}] -- mean firing rates per orientation:")
    print(f"{'Theta':>6}  {'N_spikes':>9}  {'Mean Hz':>9}")
    print("-" * 32)
    for theta in ORIENTATIONS:
        fpath = os.path.join(output_dir, f"theta{theta:03d}", "spikes.h5")
        if not os.path.exists(fpath):
            print(f"  {theta:>5}deg  NOT FOUND")
            continue
        with h5py.File(fpath, 'r') as h5:
            keys = list(h5['spikes'].keys())
            if not keys:
                print(f"  {theta:>5}deg  {'0':>9}  {'0.00':>9}  (no spikes)")
                continue
            grp = h5[f'spikes/{keys[0]}']
            n_spikes = len(grp['timestamps'][()]) if 'timestamps' in grp \
                       else len(grp[()])
        mean_hz = n_spikes / (N * TSTOP / 1000.0)
        print(f"  {theta:>5}deg  {n_spikes:>9d}  {mean_hz:>8.2f}")
    print("\nIf mean Hz is 0 or >100, adjust SYN_WEIGHT_EE / SYN_WEIGHT_LGN.")


# -- public entry point --------------------------------------------------------

def run_simulation(edge_file, tag, build_lgn=True, force_lgn=False):
    """Build (V1 recurrent for this connectome) + simulate, reusing the shared
    LGN scaffold. The LGN scaffold is built once on the first call (build_lgn=
    True) and reused thereafter.

    edge_file : [pre, post, n_syn] JSON in nucleus-id space.
    tag       : short name controlling V1/output dirs.
    build_lgn : if True, ensure the shared LGN scaffold exists (builds if absent
                or if force_lgn=True). For null runs after the real run, the
                scaffold already exists and will be reused unchanged.
    """
    network_dir = network_dir_for(tag)
    output_dir  = output_dir_for(tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*72}\nrun_simulation(tag={tag!r}, edge_file={edge_file!r})\n{'='*72}")
    neurons_df = load_nodes()

    if build_lgn:
        build_lgn_scaffold(neurons_df, force=force_lgn)
    else:
        if not os.path.exists(os.path.join(LGN_DIR, "lgn_nodes.h5")):
            raise RuntimeError(
                f"Shared LGN scaffold missing in {LGN_DIR}/. Run once with "
                f"build_lgn=True first.")

    edges_df = load_edges(edge_file)
    build_v1_recurrent(neurons_df, edges_df, network_dir)
    circuit_cfg = make_circuit_config(tag, network_dir, LGN_DIR)
    run_all_orientations(tag, output_dir, circuit_cfg)
    check_outputs(len(neurons_df), output_dir)
    print(f"\n[{tag}] done -> {output_dir}/  (LGN drive: shared {LGN_DIR}/)")
    return output_dir


# -- CLI -----------------------------------------------------------------------

if __name__ == '__main__':
    # python step2_build_and_simulate_microns.py                 # LGN + real
    # python step2_build_and_simulate_microns.py <edge_file> <tag>
    if len(sys.argv) == 3:
        edge_file, tag = sys.argv[1], sys.argv[2]
    else:
        edge_file, tag = "candidate1_real_edges.json", "real"
        print("No args -- defaulting to real connectome "
              "(candidate1_real_edges.json, tag='real'), building shared LGN.")
    run_simulation(edge_file, tag, build_lgn=True)
