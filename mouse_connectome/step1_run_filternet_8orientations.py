"""
Step 1: Run FilterNet for 8 drifting grating orientations (0-315 deg, step 45).
Produces 8 SONATA spike files in inputs/: lgn_spikes_theta000.h5 ... lgn_spikes_theta315.h5

Prerequisites:
    pip install bmtk
    # Clone bmtk repo and build LGN network files first:
    #   cd ~/Documents/GitHub
    #   git clone https://github.com/AllenInstitute/bmtk.git
    #   cd bmtk/docs/examples/filter_graitings && python build_network.py
    #   cp -r network/ ~/Documents/GitHub/connectome-fidelity-microns/network/
    #   (filter_components lives at ~/Documents/GitHub/bmtk/examples/filter_components)

Directory structure expected (relative to connectome-fidelity-microns/):
    ./network/lgn_nodes.h5
    ./network/lgn_node_types.csv
    ../bmtk/examples/filter_components/model_templates/
"""

import os
import json
import shutil
from bmtk.simulator import filternet

# ── parameters ────────────────────────────────────────────────────────────────

ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]

GRATING_PARAMS = {
    "row_size": 120,
    "col_size": 240,
    "gray_screen_dur": 0.5,
    "cpd": 0.04,
    "temporal_f": 4.0,
    "contrast": 0.8,
}

TSTOP = 2000.0
DT    = 0.1

NETWORK_DIR    = "network"
COMPONENT_DIR  = "../bmtk/examples/filter_components"
INPUT_DIR      = "inputs"
OUTPUT_DIR_BASE = "output_filternet"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

# ── helper ────────────────────────────────────────────────────────────────────

def make_filternet_config(theta, config_path, output_dir, spikes_file):
    config = {
        "manifest": {
            "$BASE_DIR": ".",
            "$OUTPUT_DIR": output_dir,
            "$NETWORK_DIR": NETWORK_DIR,
            "$COMPONENT_DIR": COMPONENT_DIR
        },
        "run": {
            "tstop": TSTOP,
            "dt": DT,
            "overwrite_output_dir": True
        },
        "target_simulator": "LGNModel",
        "conditions": {
            "jitter_lower": 0.75,
            "jitter_upper": 1.25
        },
        "components": {
            "filter_models_dir": "$COMPONENT_DIR/model_templates"
        },
        "networks": {
            "nodes": [
                {
                    "nodes_file": "$NETWORK_DIR/lgn_nodes.h5",
                    "node_types_file": "$NETWORK_DIR/lgn_node_types.csv"
                }
            ]
        },
        "inputs": {
            "LGN_spikes": {
                "input_type": "movie",
                "module": "grating",
                "theta": float(theta),
                "evaluation_options": {
                    "downsample": 1,
                    "separable": True
                },
                **GRATING_PARAMS
            }
        },
        "output": {
            "log_file": "log.txt",
            "output_dir": output_dir,
            "rates_csv": "rates.csv",
            "spikes_csv": "spikes.csv",
            "spikes_file": spikes_file,
            "overwrite_output_dir": True
        }
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

# ── main ──────────────────────────────────────────────────────────────────────

spike_files = []

for theta in ORIENTATIONS:
    print(f"\n{'='*60}")
    print(f"FilterNet: theta = {theta}deg")
    print(f"{'='*60}")

    output_dir   = os.path.join(OUTPUT_DIR_BASE, f"theta{theta:03d}")
    config_path  = os.path.join(OUTPUT_DIR_BASE, f"config_theta{theta:03d}.json")
    spikes_fname = f"spikes_theta{theta:03d}.h5"

    os.makedirs(output_dir, exist_ok=True)
    make_filternet_config(theta, config_path, output_dir, spikes_fname)

    config = filternet.Config.from_json(config_path)
    config.build_env()
    net = filternet.FilterNetwork.from_config(config)
    sim = filternet.FilterSimulator.from_config(config, net)
    sim.run()

    src = os.path.join(output_dir, spikes_fname)
    dst = os.path.join(INPUT_DIR, f"lgn_spikes_theta{theta:03d}.h5")
    shutil.copy(src, dst)
    spike_files.append(dst)
    print(f"  Spike file: {dst}")

print("\nDone. LGN spike files:")
for f in spike_files:
    print(f"  {f}")
print("\nRun step2_build_and_simulate_microns.py next.")
