"""
step2_ei_conductance_recurrent.py -- conductance-based counterpart to
step2_ei_recurrent.py. Same E/I split, same Dale's-law sign mechanism, same
everything else (node placement, LGN scaffold, orientation loop) -- the ONLY
change is the neuron model itself: nest:iaf_psc_alpha (current-based) ->
nest:iaf_cond_alpha (conductance-based).

WHY THIS EXISTS
---------------
The geometry-alignment investigation (this project, same session) ruled out
information poverty as the explanation for why real trails the nulls, and
the cell-type-specific weighting pilot (also this session) made things
*worse* along the most defensible direction tested. Del Rosario et al. 2025
(Choi/Haider lab) demonstrates a concrete, mechanistic reason conductance-
based synapses can matter: divisive gain modulation is a structural,
emergent property of conductance dynamics (via the (E-V) driving-force
term) that a current-based synapse cannot produce at all, not just fit
poorly. Whether that same necessity extends to representational geometry
(RSA), rather than gain modulation specifically, is untested -- this
module is the infrastructure needed to actually test it, not an assumption
that it will help.

WHAT CHANGED, PRECISELY, FROM THE CURRENT-BASED MODEL
-------------------------------------------------------
Confirmed directly against the installed NEST 3.10.0 (not assumed from
documentation) before building anything:
  - iaf_cond_alpha has NO separate receptor_types -- excitatory vs
    inhibitory drive is determined purely by the SIGN of the connection
    weight (positive -> g_ex, negative -> g_in), the exact same convention
    already used throughout this pipeline for Dale's law. The existing
    sign-based logic ports over with zero change and zero new risk of an
    E/I mixup.
  - The current model's synapse-side dynamics_params (instantaneousExc.json)
    is genuinely empty ({}) -- confirmed by reading the file directly, not
    assumed. iaf_cond_alpha bakes its conductance kernel (tau_syn_ex,
    tau_syn_in) directly into the NEURON model, so the same empty
    placeholder file is reused here too; nothing synapse-side needs to
    change.
  - Neuron parameter translation (see iaf_cond_alpha_point.json):
    I_e, C_m, t_ref, E_L, V_th, V_reset carry over directly (same units,
    same meaning). tau_m has no direct iaf_cond_alpha equivalent -- its
    effective membrane time constant is C_m/g_L, so g_L was DERIVED
    (g_L = C_m/tau_m = 120.0/24.0 = 5.0 nS) specifically to preserve the
    same passive membrane dynamics as the current-based model, even though
    active (synaptic) dynamics necessarily differ. E_ex, E_in, tau_syn_ex,
    tau_syn_in have no current-based counterpart at all -- left at NEST's
    own iaf_cond_alpha defaults (E_ex=0mV, E_in=-85mV: biologically
    standard AMPA/NMDA and GABA-A/chloride reversal potentials), a
    defensible first-pass choice, not yet independently justified from
    literature for this specific circuit.

WHAT "VALIDATION" MEANS HERE, AND WHY IT'S DIFFERENT FROM THE CELL-TYPE
PILOT'S VALIDATION
------------------------------------------------------------------------
The cell-type pilot could validate by exact bit-for-bit reproduction
(uniform weights through the new pathway == the old pathway's output),
because both used the same current-based dynamics -- only the number of
weight parameters changed. Current-based and conductance-based are
genuinely different dynamical systems; there is no parameter setting that
makes iaf_cond_alpha produce bit-identical output to iaf_psc_alpha.
Validation here instead means: (1) the model runs without crashing,
(2) it produces non-degenerate firing rates when fit to the same targets,
(3) the existing Dale's-law sign convention still produces the correct
FUNCTIONAL effect -- increasing inhibitory weight measurably suppresses
excitatory population rate, not just "doesn't crash." See
validate_conductance_infrastructure.py.

USAGE: import this module and call run_simulation_ei_cond() in place of
step2_ei_recurrent.run_simulation_ei() -- same signature. Everything else
(LGN scaffold sharing, orientation loop, output checking) is inherited
from step2 unchanged, exactly as step2_ei_recurrent.py already does.
"""

import os

import numpy as np

import step2_build_and_simulate_microns as step2
import step2_ei_recurrent as step2_ei

# Deliberately separate from step2.V1_MODEL_TEMPLATE / V1_DYNAMICS_PARAMS --
# never overwrite those constants, so the existing current-based pipeline
# remains completely untouched regardless of what this module does.
V1_MODEL_TEMPLATE_COND = "nest:iaf_cond_alpha"
V1_DYNAMICS_PARAMS_COND = "iaf_cond_alpha_point.json"


def build_v1_recurrent_ei_cond(neurons_df, edges_df, network_dir,
                               syn_weight_e, syn_weight_i, node_types=None):
    """Conductance-based counterpart to
    step2_ei_recurrent.build_v1_recurrent_ei() -- identical structure,
    only the neuron model_template/dynamics_params differ. syn_weight_e/i
    are now CONDUCTANCE weights (nS), not current weights -- the same
    coordinate-descent fitting procedure applies, but the resulting
    fitted values will be on a different numeric scale than the
    current-based model's fitted weights, since they mean something
    physically different. Do not compare raw fitted weight magnitudes
    across the two model classes directly.
    """
    from bmtk.builder import NetworkBuilder

    N = len(neurons_df)
    os.makedirs(network_dir, exist_ok=True)

    if node_types is None:
        node_types = step2_ei.load_node_types()

    nucleus_to_node = {int(row.nucleus_id): idx
                       for idx, row in neurons_df.iterrows()}
    node_to_nucleus = {v: k for k, v in nucleus_to_node.items()}

    def is_inhibitory_node(node_id):
        nuc = node_to_nucleus.get(node_id)
        return node_types.get(nuc) == 'I'

    conn_lookup_e = {}
    conn_lookup_i = {}
    skipped = 0
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

    v1 = NetworkBuilder('microns_v1')
    v1.add_nodes(
        N=N, model_type='point_neuron', ei='e',
        model_template=V1_MODEL_TEMPLATE_COND,
        dynamics_params=V1_DYNAMICS_PARAMS_COND,
        x=neurons_df['x'].values,
        y=neurons_df['y'].values,
        z=neurons_df['z'].values,
    )

    # same BMTK uint32-overflow workaround as the current-based model --
    # the underlying bug (nsyns*weight landing on an exact negative
    # integer) is about BMTK's SONATA writer, not about which neuron
    # model is downstream, so it applies here identically
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
            dynamics_params='instantaneousExc.json',  # confirmed empty ({})
                                                         # -- reused as-is,
                                                         # since iaf_cond_alpha
                                                         # bakes its
                                                         # conductance kernel
                                                         # into the NEURON,
                                                         # not the synapse
            syn_weight=syn_weight_e, delay=2.0,
        )
    if conn_lookup_i:
        v1.add_edges(
            source={'ei': 'e'}, target={'ei': 'e'},
            connection_rule=rule_i,
            model_template='static_synapse',
            dynamics_params='instantaneousExc.json',
            syn_weight=-syn_weight_i, delay=2.0,   # NEGATIVE weight is the
                                                     # entire E/I mechanism,
                                                     # confirmed directly:
                                                     # iaf_cond_alpha has no
                                                     # receptor_types, sign
                                                     # of weight alone routes
                                                     # to g_ex vs g_in
        )

    print(f"  Building V1 recurrent, conductance-based ({N}x{N} = {N*N:,} "
          f"pairs)...")
    v1.build()
    v1.save(output_dir=network_dir)
    print("  V1 recurrent (conductance-based E/I) saved.")


def run_simulation_ei_cond(edge_file, tag, syn_weight_e, syn_weight_i,
                           build_lgn=True, force_lgn=False, node_types=None):
    """Conductance-based counterpart to
    step2_ei_recurrent.run_simulation_ei() -- identical flow, only the
    connectivity-building step differs.
    """
    network_dir = step2.network_dir_for(tag)
    output_dir = step2.output_dir_for(tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*72}\nrun_simulation_ei_cond(tag={tag!r}, "
          f"edge_file={edge_file!r})\n{'='*72}")
    neurons_df = step2_ei.load_nodes_ei()

    if build_lgn:
        step2.build_lgn_scaffold(neurons_df, force=force_lgn)
    else:
        if not os.path.exists(os.path.join(step2.LGN_DIR, "lgn_nodes.h5")):
            raise RuntimeError(
                f"Shared LGN scaffold missing in {step2.LGN_DIR}/. Run once "
                f"with build_lgn=True first.")

    edges_df = step2_ei.load_edges_signed(edge_file)
    build_v1_recurrent_ei_cond(neurons_df, edges_df, network_dir,
                               syn_weight_e, syn_weight_i, node_types=node_types)
    circuit_cfg = step2.make_circuit_config(tag, network_dir, step2.LGN_DIR)
    step2.run_all_orientations(tag, output_dir, circuit_cfg)
    step2.check_outputs(len(neurons_df), output_dir)
    print(f"\n[{tag}] done (conductance-based E/I) -> {output_dir}/  "
          f"(LGN drive: shared {step2.LGN_DIR}/)")
    return output_dir
