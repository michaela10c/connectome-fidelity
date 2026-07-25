"""
step2_ei_recurrent.py — E/I-aware V1 recurrent connectivity, built on top of
step2_build_and_simulate_microns.py rather than replacing it.

WHY THIS EXISTS
---------------
Everything in step2 except build_v1_recurrent() is already E/I-agnostic: the
shared LGN scaffold, circuit config, per-orientation simulation running, and
output sanity check don't care about synapse weight signs, only node/edge
counts and file paths. Only connectivity building needs Dale's law.

BMTK's actual mechanism for signed synapses (confirmed against BMTK's own
tutorial, not assumed): a NEGATIVE syn_weight on the same static_synapse
model_template is how inhibitory connections are implemented -- there is no
separate "inhibitory" synapse model needed. For a linear, no-kernel synapse
(the existing instantaneousExc.json this project already uses -- "instant",
i.e. no temporal filtering), sign alone fully determines excitatory vs
inhibitory effect on this class of point neuron. So the fix is: split the
single add_edges() call into two, one per source type, each with its own
signed weight scale -- same dynamics_params for both.

TWO INDEPENDENT FREE PARAMETERS, replacing step2's single SYN_WEIGHT_EE:
    SYN_WEIGHT_E : magnitude for excitatory-sourced edges (sign applied: +)
    SYN_WEIGHT_I : magnitude for inhibitory-sourced edges (sign applied: -)

INPUTS: node_data_v1_ei.csv (full E+I population, from
        01b_save_node_positions_ei.py -- NOT node_data_v1.csv, which is
        E-only), PLUS node_types.json (nucleus_id -> E/I) to know which
        nodes are inhibitory. Edge files are the same
        [pre, post, n_syn] format written by signed_null_c1.py /
        unrestricted_null_c1.py / a real signed-graph edge file -- no format
        change needed, sign is applied here based on node_types.json, not
        carried in the edge file itself.

USAGE: import this module and call run_simulation_ei() in place of
step2.run_simulation() -- same signature plus syn_weight_e/syn_weight_i.
Everything else (LGN scaffold sharing, orientation loop, output checking)
is inherited from step2 unchanged.
"""

import os
import json
import numpy as np
import pandas as pd

import step2_build_and_simulate_microns as step2

NODE_TYPES = "node_types.json"
NODE_CSV_EI = "node_data_v1_ei.csv"   # the full 2,196-neuron E+I file from
                                        # 01b_save_node_positions_ei.py --
                                        # NOT node_data_v1.csv, which is
                                        # E-only and silently drops every
                                        # edge touching an I neuron if used
                                        # here (confirmed empirically)


def load_nodes_ei(path=NODE_CSV_EI):
    """Load the full E+I node population. Distinct from step2.load_nodes(),
    which reads node_data_v1.csv (E-only, 1,762 rows) -- using that file
    here would silently drop all inhibitory-touching edges again."""
    print(f"Loading neuron data from {path}...")
    node_data = pd.read_csv(path, index_col="node_id")
    node_data = node_data.rename(columns={"pt_root_id": "root_id"})
    print(f"  Loaded {len(node_data)} neurons "
          f"({(node_data.cell_class=='E').sum()} E + "
          f"{(node_data.cell_class=='I').sum()} I)")
    return node_data

# Defaults matching step2's original SYN_WEIGHT_EE=5.0 as a starting point
# for the excitatory side; the inhibitory default is a placeholder, NOT a
# fitted value -- step7_ei_fit_weights.py exists specifically because this
# cannot be guessed and must be fit per condition, same as the E-only case.
SYN_WEIGHT_E_DEFAULT = 5.0
SYN_WEIGHT_I_DEFAULT = 5.0


def load_node_types(path=NODE_TYPES):
    """Return {nucleus_id (int): 'E'|'I'}."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_v1_recurrent_ei(neurons_df, edges_df, network_dir,
                          syn_weight_e, syn_weight_i,
                          node_types=None):
    """E/I-aware replacement for step2.build_v1_recurrent(). Splits edges by
    SOURCE type (Dale's law: sign lives on the presynaptic neuron) and issues
    two add_edges() calls with independently-scaled, oppositely-signed
    weights. Everything else (node placement, dynamics_params, delay) is
    identical to step2's original.
    """
    from bmtk.builder import NetworkBuilder

    N = len(neurons_df)
    os.makedirs(network_dir, exist_ok=True)

    if node_types is None:
        node_types = load_node_types()

    nucleus_to_node = {int(row.nucleus_id): idx
                       for idx, row in neurons_df.iterrows()}
    node_to_nucleus = {v: k for k, v in nucleus_to_node.items()}

    def is_inhibitory_node(node_id):
        nuc = node_to_nucleus.get(node_id)
        return node_types.get(nuc) == 'I'

    conn_lookup_e = {}   # (src, trg) -> n_syn, for E-sourced edges
    conn_lookup_i = {}   # (src, trg) -> n_syn, for I-sourced edges
    skipped = 0
    sign_mismatch = 0
    for row in edges_df.itertuples(index=False):
        src = nucleus_to_node.get(int(row.pre_node_id))
        trg = nucleus_to_node.get(int(row.post_node_id))
        if src is None or trg is None:
            skipped += 1
            continue
        n_syn = max(int(row.syn_count), 1)
        if is_inhibitory_node(src):
            conn_lookup_i[(src, trg)] = n_syn
        else:
            conn_lookup_e[(src, trg)] = n_syn
    if skipped:
        print(f"  Skipped {skipped} edges with nucleus IDs not in node set")
    print(f"  E-sourced edges: {len(conn_lookup_e)}   "
          f"I-sourced edges: {len(conn_lookup_i)}")

    for label, conn_lookup, weight_scale in [
        ("E", conn_lookup_e, syn_weight_e),
        ("I", conn_lookup_i, syn_weight_i),
    ]:
        if not conn_lookup:
            print(f"  [{label}] no edges -- skipping weight report")
            continue
        counts = np.array(list(conn_lookup.values()), dtype=float)
        eff = counts * weight_scale
        med = float(np.median(eff))
        pct = np.percentile(eff, [50, 90, 99, 100])
        print(f"  [{label} weights] nsyns median={np.median(counts):.0f} "
              f"max={counts.max():.0f}  (weight = nsyns * {weight_scale:.4f})")
        print(f"  [{label} weights] effective weight: median={pct[0]:.2f} "
              f"p90={pct[1]:.2f} p99={pct[2]:.2f} max={pct[3]:.2f} "
              f"(max/median={pct[3]/med:.1f}x)" if med > 0 else "")

    v1 = NetworkBuilder('microns_v1')
    v1.add_nodes(
        N=N, model_type='point_neuron', ei='e',   # 'ei' tag here is a BMTK
                                                    # node metadata field, not
                                                    # what determines synapse
                                                    # sign -- sign is applied
                                                    # explicitly below via
                                                    # syn_weight, per the
                                                    # confirmed BMTK convention
        model_template=step2.V1_MODEL_TEMPLATE,
        dynamics_params=step2.V1_DYNAMICS_PARAMS,
        x=neurons_df['x'].values,
        y=neurons_df['y'].values,
        z=neurons_df['z'].values,
    )

    # BMTK bug workaround: its SONATA edge writer infers an integer dtype for
    # the weight column when computed values look integer-like, then fails
    # with "OverflowError: Python integer -1 out of bounds for uint32" the
    # moment nsyns * syn_weight lands on an exact negative integer (e.g.
    # nsyns=1, syn_weight_i=1.0 -> -1.0 exactly). Confirmed reproducible:
    # every previously-tested weight (0.5, 2.0, 5120.0, etc.) worked fine;
    # only w_i=1 (an untested round number) triggered it. A tiny, fixed
    # perturbation guarantees nsyns*weight is never an exact integer for any
    # small integer nsyns, without meaningfully changing the fitted network's
    # dynamics -- this is a numerical-type workaround, not a modeling choice.
    _EPS = 1e-6
    syn_weight_e = syn_weight_e + _EPS
    syn_weight_i = syn_weight_i + _EPS

    def rule_e(src, trg):
        return conn_lookup_e.get((src.node_id, trg.node_id), 0)

    def rule_i(src, trg):
        return conn_lookup_i.get((src.node_id, trg.node_id), 0)

    if conn_lookup_e:
        v1.add_edges(
            source={'ei': 'e'}, target={'ei': 'e'},
            connection_rule=rule_e,
            model_template='static_synapse',
            dynamics_params='instantaneousExc.json',
            syn_weight=syn_weight_e, delay=2.0,
        )
    if conn_lookup_i:
        v1.add_edges(
            source={'ei': 'e'}, target={'ei': 'e'},
            connection_rule=rule_i,
            model_template='static_synapse',
            dynamics_params='instantaneousExc.json',   # same file: an
                                                          # instantaneous,
                                                          # no-kernel synapse
                                                          # has no separate
                                                          # E/I-specific
                                                          # temporal shape to
                                                          # differ on; sign of
                                                          # syn_weight below
                                                          # is what implements
                                                          # Dale's law
            syn_weight=-syn_weight_i, delay=2.0,   # NEGATIVE: this is the
                                                     # entire mechanism, per
                                                     # BMTK's own tutorial
                                                     # convention
        )

    print(f"  Building V1 recurrent ({N}x{N} = {N*N:,} pairs, "
          f"E+I split)...")
    v1.build()
    v1.save(output_dir=network_dir)
    print("  V1 recurrent (E/I) saved.")
    return v1


def load_edges_signed(edge_file):
    """Load an edge file in EITHER format this pipeline produces:
        [pre, post, n_syn]              -- signed_null_c1.py / unrestricted_null_c1.py
        [pre, post, n_syn, pre_type]    -- the real signed_real_edges.json pull

    step2.load_edges() only branches on len==3 (falls back to an
    unweighted 2-column parse otherwise), so it breaks on the real signed
    file's 4th column. This handles both without touching step2's original
    function, which stays correct and unchanged for the E-only pipeline.
    The 4th column (pre_type), when present, is not used here -- sign is
    determined from node_types.json in build_v1_recurrent_ei(), which is
    already cross-validated against the edge file's own pre_type field
    elsewhere (signed_null_c1.py's load_signed()); carrying it through here
    too would be redundant, not more correct.
    """
    print(f"Loading edges from {edge_file}...")
    with open(edge_file) as f:
        edges_raw = json.load(f)
    if len(edges_raw) == 0:
        raise ValueError(f"{edge_file} is empty")
    n_cols = len(edges_raw[0])
    if n_cols not in (3, 4):
        raise ValueError(
            f"{edge_file}: expected 3 or 4 columns per edge "
            f"([pre,post,n_syn] or [pre,post,n_syn,pre_type]), got {n_cols}")
    edges_df = pd.DataFrame(
        [(int(e[0]), int(e[1]), int(e[2])) for e in edges_raw],
        columns=['pre_node_id', 'post_node_id', 'syn_count'])
    print(f"  Loaded {len(edges_df)} directed edges "
          f"(source format: {n_cols} columns)")
    return edges_df


def run_simulation_ei(edge_file, tag, syn_weight_e, syn_weight_i,
                      build_lgn=True, force_lgn=False, node_types=None):
    """E/I-aware counterpart to step2.run_simulation(). Reuses step2's shared
    LGN scaffold, circuit config, and per-orientation run loop verbatim --
    only the V1 recurrent connectivity step is replaced.
    """
    network_dir = step2.network_dir_for(tag)
    output_dir = step2.output_dir_for(tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*72}\nrun_simulation_ei(tag={tag!r}, edge_file={edge_file!r}, "
          f"E={syn_weight_e:.4f}, I={syn_weight_i:.4f})\n{'='*72}")
    neurons_df = load_nodes_ei()

    if build_lgn:
        step2.build_lgn_scaffold(neurons_df, force=force_lgn)
    else:
        if not os.path.exists(os.path.join(step2.LGN_DIR, "lgn_nodes.h5")):
            raise RuntimeError(
                f"Shared LGN scaffold missing in {step2.LGN_DIR}/. Run once "
                f"with build_lgn=True first.")

    edges_df = load_edges_signed(edge_file)
    build_v1_recurrent_ei(neurons_df, edges_df, network_dir,
                          syn_weight_e, syn_weight_i, node_types=node_types)
    circuit_cfg = step2.make_circuit_config(tag, network_dir, step2.LGN_DIR)
    step2.run_all_orientations(tag, output_dir, circuit_cfg)
    step2.check_outputs(len(neurons_df), output_dir)
    print(f"\n[{tag}] done (E/I) -> {output_dir}/  "
          f"(LGN drive: shared {step2.LGN_DIR}/)")
    return output_dir


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        edge_file, tag = sys.argv[1], sys.argv[2]
    else:
        edge_file, tag = "signed_real_edges.json", "real_ei"
        print("No args -- defaulting to real signed connectome "
              "(signed_real_edges.json, tag='real_ei'), building shared LGN.")
    run_simulation_ei(edge_file, tag,
                      syn_weight_e=SYN_WEIGHT_E_DEFAULT,
                      syn_weight_i=SYN_WEIGHT_I_DEFAULT,
                      build_lgn=True)
