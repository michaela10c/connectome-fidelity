"""
correct_exp5_circularity.py -- applies the SAME correction already validated
for Exp3 (rank-residualize both RDMs against the circular-distance
reference, then correlate residuals) to this Exp5 trained-random result,
since production.py's build_bio_rdm() is explicitly commented "Verbatim
Exp 4" -- this is the identical 97.8%-circular Maisak reference already
known to substantially inflate raw r-vs-bio numbers (Exp3's CC dropped
from 0.930 to 0.145, not significant, after this exact correction).

Reuses build_bio_rdm() directly from production.py rather than
reconstructing it, to guarantee an exact match to the reference already
established as confounded -- no risk of a subtly different circular
reference producing a misleading "different" answer.

USAGE:
    python correct_exp5_circularity.py
"""

import argparse
import glob

import numpy as np
from scipy.stats import rankdata, pearsonr, spearmanr

from production import build_bio_rdm

N_DIRECTIONS = 12


def upper_tri(mat):
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]


def circular_distance_rdm(n=N_DIRECTIONS):
    """circ[i,j] = min(|i-j|, n-|i-j|) -- the same circular-distance
    reference structure Exp3's correction was run against."""
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(diff, n - diff).astype(float)


def rank_residualize(values, against):
    """Rank-residualize `values` on `against` -- Spearman-equivalent
    partial correlation via linear regression on ranks."""
    r_values = rankdata(values)
    r_against = rankdata(against)
    slope, intercept = np.polyfit(r_against, r_values, 1)
    predicted = slope * r_against + intercept
    return r_values - predicted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="e.g. pilot_out_250k/degree_preserving_swap or "
                         "pilot_out_250k/erdos_renyi -- the FIX for the bug "
                         "in the previous version, which was hardcoded to "
                         "degree_preserving_swap regardless of what you "
                         "actually wanted to check")
    args = ap.parse_args()

    print("Loading biological reference via build_bio_rdm() (Verbatim Exp 4, "
          "the same reference already known to be 97.8% circular)...")
    bio_rdm = build_bio_rdm()
    print(f"  bio_rdm shape: {bio_rdm.shape}")

    print(f"\nLoading and averaging the per-network trained-random RDMs from "
          f"{args.dir} (same as production.py's aggregate_from_disk mean_rdm)...")
    rdm_files = sorted(glob.glob(f"{args.dir}/rdm_net*.npy"))
    print(f"  found {len(rdm_files)} files")
    if not rdm_files:
        raise FileNotFoundError(
            f"no rdm_net*.npy files found in {args.dir} -- check the path "
            f"and that this scheme's evaluation has actually completed")
    all_rdms = np.stack([np.load(f) for f in rdm_files])
    mean_rdm = all_rdms.mean(axis=0)
    print(f"  mean_rdm shape: {mean_rdm.shape}")

    circ_rdm = circular_distance_rdm(bio_rdm.shape[0])

    bio_tri = upper_tri(bio_rdm)
    mean_tri = upper_tri(mean_rdm)
    circ_tri = upper_tri(circ_rdm)

    print("\n=== Sanity check: how circular is the bio reference itself? ===")
    r_bio_circ, p_bio_circ = spearmanr(bio_tri, circ_tri)
    print(f"  bio_rdm vs circular-distance: rho={r_bio_circ:.4f}  p={p_bio_circ:.2e}")
    print(f"  (Exp3 established this at r=0.978 -- should match closely if this "
          f"is genuinely the same reference)")

    print("\n=== Raw (uncorrected) result, matching production.py's printed r ===")
    r_raw, p_raw = spearmanr(mean_tri, bio_tri)
    print(f"  trained-random vs bio, raw Spearman: r={r_raw:.4f}  p={p_raw:.2e}")
    print(f"  (production.py printed r=0.832 -- should match this closely)")

    print("\n=== Corrected: partial correlation controlling for circular distance ===")
    resid_mean = rank_residualize(mean_tri, circ_tri)
    resid_bio = rank_residualize(bio_tri, circ_tri)
    r_partial, p_partial = pearsonr(resid_mean, resid_bio)
    print(f"  trained-random vs bio, PARTIAL (circularity removed): "
          f"r={r_partial:.4f}  p={p_partial:.2e}")

    print(f"\n=== Summary ===")
    print(f"  raw r        = {r_raw:.4f}")
    print(f"  partial r    = {r_partial:.4f}")
    print(f"  drop         = {r_raw - r_partial:+.4f}")
    if abs(r_partial) < 0.2 or p_partial > 0.05:
        print(f"\n  Same pattern as Exp3's CC correction (0.930 -> 0.145, n.s.): "
              f"most of the raw r appears to be circular-structure artifact, "
              f"not genuine trained-random-vs-biology fidelity. Do not "
              f"interpret the raw r=0.832 as a real result without this context.")
    else:
        print(f"\n  Unlike Exp3's CC correction, meaningful signal survives "
              f"circularity control here. Worth double-checking this is not "
              f"driven by a different confound before trusting it either.")


if __name__ == "__main__":
    main()
