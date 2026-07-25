"""
MICrONS orientation script — Step 8
Goal: updated RSA comparison including Candidate 3 (cell-type-block-shuffled).

Interpretive hierarchy:
  real > C3 > C2 > C1: sub-cell-type wiring specificity predicts functional
    geometry beyond cell-type statistics, proximity, and degree
  real > C2 ≈ C3 > C1: non-spatial biological specificity matters, but
    cell-type-block shuffling explains most of it
  real ≈ C3 > C2 > C1: cell-type statistics explain the signal
  real ≈ C3 ≈ C2 ≈ C1: degree statistics explain everything

Requires:
  structural_rdm_real.npy
  structural_rdm_candidate1.npy
  structural_rdm_candidate2.npy
  structural_rdm_candidate3.npy
  structural_rdm_nucleus_ids.npy
  signal_corr_matrix.npy
  signal_corr_nucleus_ids.npy

Outputs:
  rsa_results_full.txt     — all four Spearman r values and permutation p
  rsa_comparison_full.png  — updated bar chart with all four conditions
"""

import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Step 1: Load all RDMs and align
# -------------------------------------------------------------------
print("Loading RDMs...")

rdm_real = np.load('structural_rdm_real.npy')
rdm_c1   = np.load('structural_rdm_candidate1.npy')
rdm_c2   = np.load('structural_rdm_candidate2.npy')
rdm_c3   = np.load('structural_rdm_candidate3.npy')
struct_ids = np.load('structural_rdm_nucleus_ids.npy')

sig_corr_matrix = np.load('signal_corr_matrix.npy')
sig_corr_ids    = np.load('signal_corr_nucleus_ids.npy')

sig_id_to_idx = {nid: i for i, nid in enumerate(sig_corr_ids.tolist())}
shared_ids = [nid for nid in struct_ids if nid in sig_id_to_idx]
N = len(shared_ids)

struct_sel = [i for i, nid in enumerate(struct_ids) if nid in sig_id_to_idx]
sig_sel    = [sig_id_to_idx[nid] for nid in shared_ids]

rdm_real_a = rdm_real[np.ix_(struct_sel, struct_sel)]
rdm_c1_a   = rdm_c1[np.ix_(struct_sel, struct_sel)]
rdm_c2_a   = rdm_c2[np.ix_(struct_sel, struct_sel)]
rdm_c3_a   = rdm_c3[np.ix_(struct_sel, struct_sel)]

sig_corr_a = sig_corr_matrix[np.ix_(sig_sel, sig_sel)]
rdm_bio    = 1 - sig_corr_a
np.fill_diagonal(rdm_bio, 0)

print(f"Aligned RDM shape: {rdm_real_a.shape}, N pairs: {N*(N-1)//2:,}")

# -------------------------------------------------------------------
# Step 2: Spearman r for all four conditions
# -------------------------------------------------------------------
triu_idx = np.triu_indices(N, k=1)
vec_bio  = rdm_bio[triu_idx]
vec_real = rdm_real_a[triu_idx]
vec_c1   = rdm_c1_a[triu_idx]
vec_c2   = rdm_c2_a[triu_idx]
vec_c3   = rdm_c3_a[triu_idx]

print("\nComputing Spearman correlations...")
r_real, p_real = spearmanr(vec_real, vec_bio)
r_c1,   p_c1   = spearmanr(vec_c1,   vec_bio)
r_c2,   p_c2   = spearmanr(vec_c2,   vec_bio)
r_c3,   p_c3   = spearmanr(vec_c3,   vec_bio)

print(f"Real connectome:  r = {r_real:.4f}, p = {p_real:.4e}")
print(f"Candidate 1:      r = {r_c1:.4f},   p = {p_c1:.4e}")
print(f"Candidate 2:      r = {r_c2:.4f},   p = {p_c2:.4e}")
print(f"Candidate 3:      r = {r_c3:.4f},   p = {p_c3:.4e}")

# -------------------------------------------------------------------
# Step 3: Permutation test (n=1000)
# -------------------------------------------------------------------
print("\nRunning permutation tests (n=1000)...")
np.random.seed(42)
n_perms = 1000

perm_r = {k: np.zeros(n_perms) for k in ['real', 'c1', 'c2', 'c3']}

for i in range(n_perms):
    perm_idx = np.random.permutation(N)
    rdm_bio_perm = rdm_bio[np.ix_(perm_idx, perm_idx)]
    vec_bio_perm = rdm_bio_perm[triu_idx]
    perm_r['real'][i] = spearmanr(vec_real, vec_bio_perm)[0]
    perm_r['c1'][i]   = spearmanr(vec_c1,   vec_bio_perm)[0]
    perm_r['c2'][i]   = spearmanr(vec_c2,   vec_bio_perm)[0]
    perm_r['c3'][i]   = spearmanr(vec_c3,   vec_bio_perm)[0]
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_perms} done...")

perm_p_real = (np.abs(perm_r['real']) >= np.abs(r_real)).mean()
perm_p_c1   = (np.abs(perm_r['c1'])   >= np.abs(r_c1)).mean()
perm_p_c2   = (np.abs(perm_r['c2'])   >= np.abs(r_c2)).mean()
perm_p_c3   = (np.abs(perm_r['c3'])   >= np.abs(r_c3)).mean()

print(f"\nPermutation p-values:")
print(f"Real connectome:  p_perm = {perm_p_real:.4f}")
print(f"Candidate 1:      p_perm = {perm_p_c1:.4f}")
print(f"Candidate 2:      p_perm = {perm_p_c2:.4f}")
print(f"Candidate 3:      p_perm = {perm_p_c3:.4f}")

# -------------------------------------------------------------------
# Step 4: Save results
# -------------------------------------------------------------------
results_text = f"""RSA Comparison Results (Full) — MICrONS Connectome Fidelity
=============================================================
N neurons: {N}
N pairs:   {N*(N-1)//2:,}
Biological reference: in vivo signal correlation RDM (Ding et al. 2025)
Structural RDMs: cosine distance on binary connectivity profiles
RSA metric: Spearman r between upper triangles

Results
-------
Real connectome:
  Spearman r = {r_real:.4f}, analytical p = {p_real:.4e}, perm p = {perm_p_real:.4f}

Candidate 1 (degree-preserving rewiring):
  Spearman r = {r_c1:.4f}, analytical p = {p_c1:.4e}, perm p = {perm_p_c1:.4f}

Candidate 2 (distance-constrained rewiring):
  Spearman r = {r_c2:.4f}, analytical p = {p_c2:.4e}, perm p = {perm_p_c2:.4f}

Candidate 3 (cell-type-block-shuffled rewiring):
  Spearman r = {r_c3:.4f}, analytical p = {p_c3:.4e}, perm p = {perm_p_c3:.4f}

Interpretive hierarchy
----------------------
real > C3 > C2 > C1: sub-cell-type wiring specificity matters beyond
  cell-type statistics, proximity, and degree.
real > C2 ≈ C3 > C1: non-spatial biological specificity matters, but
  most of it is explained by cell-type statistics.
real ≈ C3 > C2 > C1: cell-type statistics explain the signal.
real ≈ C3 ≈ C2 ≈ C1: degree statistics explain everything.
"""

with open('rsa_results_full.txt', 'w') as f:
    f.write(results_text)
print("\nSaved rsa_results_full.txt")

# -------------------------------------------------------------------
# Step 5: Plot
# -------------------------------------------------------------------
labels = [
    'Real\nconnectome',
    'Candidate 1\n(degree)',
    'Candidate 2\n(distance)',
    'Candidate 3\n(cell type)'
]
r_vals   = [r_real, r_c1, r_c2, r_c3]
colors   = ['steelblue', 'lightgray', 'tomato', 'mediumpurple']
perm_ps  = [perm_p_real, perm_p_c1, perm_p_c2, perm_p_c3]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, r_vals, color=colors, edgecolor='black',
              linewidth=0.8, width=0.5)

for i, (r, pp) in enumerate(zip(r_vals, perm_ps)):
    marker = '***' if pp < 0.001 else '**' if pp < 0.01 else \
             '*' if pp < 0.05 else 'n.s.'
    ypos = r + 0.0005 if r >= 0 else r - 0.003
    ax.text(i, ypos, marker, ha='center', va='bottom', fontsize=11)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel("Spearman r\n(structural RDM vs. signal correlation RDM)",
              fontsize=11)
ax.set_title(
    "RSA: Connectome structural geometry vs. functional similarity\n"
    "MICrONS proofread excitatory neurons (N=899)",
    fontsize=11
)
ax.set_ylim(min(r_vals) - 0.01, max(r_vals) + 0.015)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('rsa_comparison_full.png', dpi=150, bbox_inches='tight')
print("Saved rsa_comparison_full.png")

print("\n--- STEP 8 COMPLETE ---")
print(f"Real:        r = {r_real:.4f} (perm p = {perm_p_real:.4f})")
print(f"Candidate 1: r = {r_c1:.4f} (perm p = {perm_p_c1:.4f})")
print(f"Candidate 2: r = {r_c2:.4f} (perm p = {perm_p_c2:.4f})")
print(f"Candidate 3: r = {r_c3:.4f} (perm p = {perm_p_c3:.4f})")
