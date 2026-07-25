#!/usr/bin/env python
"""
Regenerates plot_comparison figures from a saved *_raw_for_replotting.npz
file, with NO network evaluation, no flyvis import, no GPU needed at all --
exactly the "plot after the results are in without rerunning the jobs"
workflow. Reuses plot_comparison() verbatim from test_item1_all_null_schemes.py.

Usage:
    python replot_from_saved.py --npz item1_all_null_schemes_results_moving_edge_12dir_on_off_first_raw_for_replotting.npz
"""
import argparse
import numpy as np

# Reused verbatim, from the lightweight, flyvis-independent module -- no
# GPU, torch, or flyvis needed to replot
from plotting_utils import plot_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_prefix", default="replot")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    angles = list(d["angles"])
    polarity = str(d["polarity"])
    checkpoint_label = str(d["checkpoint_label"])

    schemes = set()
    for key in d.files:
        if key.endswith("_cc_rdm"):
            schemes.add(key[:-len("_cc_rdm")])

    print(f"Found {len(schemes)} scheme(s) in {args.npz}: {sorted(schemes)}")
    for scheme in sorted(schemes):
        cc_rdm = d[f"{scheme}_cc_rdm"]
        null_rdm = d[f"{scheme}_null_rdm"]
        null_r_dist = d[f"{scheme}_null_r_dist"]
        obs_r = float(d[f"{scheme}_obs_r"])
        print(f"  Replotting {scheme}...")
        plot_comparison(cc_rdm, null_rdm, scheme, polarity, angles, null_r_dist,
                         obs_r, out_prefix=args.out_prefix, checkpoint_label=checkpoint_label)
    print("Done -- no network evaluation was performed, purely replotted from saved data.")


if __name__ == "__main__":
    main()
