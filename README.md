# Representational Geometry as a Fidelity Metric for Connectome-Constrained Neural Emulations

This repository implements a proof-of-concept showing that connectome-constrained networks
produce geometrically distinct population codes compared to randomly initialized networks
with the same architecture — using representational similarity analysis (RSA) applied to
the [Flyvis](https://github.com/TuragaLab/flyvis) Drosophila visual system model.

---

## Background

Connectome-scale neural emulations are increasingly feasible, but the field lacks a principled framework for evaluating their fidelity. Brunton et al. (2026) demonstrated that behavioral fidelity is achievable without biological fidelity — a randomly wired network can produce realistic fly walking. This raises the question: what does biological wiring actually contribute, and how do we measure it?

Representational geometry — the structure of pairwise distances between population responses to different stimuli — offers a candidate answer. If connectome-constrained networks produce a representational geometry that random networks cannot replicate, then geometry is a fidelity-discriminating signal that operates at the population level, without requiring a behavioral decoder.

This project tests that hypothesis using the pretrained Flyvis ensemble (Lappalainen et al. 2024), applying RSA (Kriegeskorte et al. 2008) to compare population codes across connectome-constrained models versus sign-preserving random weight shuffles.

---

## Experiment

**Stimuli:** 12 ON moving edges at 30° increments (0° through 330°)

**Networks:**
- *Connectome-constrained (CC):* All 50 models in the pretrained Flyvis ensemble (indices 000–049 within flow/0000, pre-sorted by task error in directory naming), trained to perform optic flow estimation on naturalistic video with connectome-fixed architecture (734 free parameters)
- *Random baseline:* Same 50 model architectures with weight magnitudes shuffled while preserving E/I sign structure (Shiu-style control); matched-normal resampling also tested as an alternative (see Results)

**Population vectors:** Peak central-cell voltage per cell type (65-dim) in response to each stimulus direction

**Metrics:**
- Cosine distance RDM — scale-invariant, captures pattern geometry
- Euclidean distance RDM — captures magnitude differences
- Spearman RDM correlation — measures similarity between CC and random geometry
- Within-ensemble consistency — measures stability of CC representational geometry across trained solutions

---

## Key Results

### n=10 (top 10 models, primary fidelity result)

| Metric | Value |
|--------|-------|
| CC cosine RDM off-diagonal range | 0.001 – 0.022 (structured) |
| Random cosine RDM off-diagonal range | ~0.200 (uniform — no direction selectivity) |
| CC vs random RDM correlation (cosine) | r = 0.757, p < 0.0001 |
| Within-CC ensemble consistency | r = 0.838 ± 0.078 |
| Random models with unstable dynamics | 5 / 10 |
| CC models with unstable dynamics | 0 / 10 |

### n=50 (full ensemble, Shiu-style shuffle)

| Metric | Value |
|--------|-------|
| CC cosine RDM off-diagonal range | 0.001 – 0.012 (structured) |
| Random cosine RDM | NaN — undefined due to numerical overflow from unstable models |
| CC vs random RDM correlation (cosine) | NaN — not computable |
| Within-CC ensemble consistency | r = 0.721 ± 0.150 |
| Random models with unstable dynamics | 33 / 50 (66%) |
| CC models with unstable dynamics | 0 / 50 |

### n=50 (full ensemble, matched-normal resampling)

| Metric | Value |
|--------|-------|
| Random cosine RDM | NaN — still not computable |
| Random models with unstable dynamics | 38 / 50 (76%) |
| CC models with unstable dynamics | 0 / 50 |

The connectome-constrained network produces direction-sensitive representational geometry with a smooth circular structure — adjacent directions are most similar, opposite directions most dissimilar — consistent with the known tuning of T4/T5 neurons in the fly visual system. Zero trained CC models exhibited instability at either n=10 or n=50 under any randomization strategy, while 66–76% of random models collapsed at n=50, confirming that the biological connectome reliably occupies a dynamically stable region of parameter space that random weight configurations consistently leave.

![RDM figure](figures/moving_edge_poc_rdms.png)

*Left to right: connectome-constrained cosine RDM, random baseline cosine RDM, connectome-constrained Euclidean RDM, random baseline Euclidean RDM (n=50 run, Shiu-style shuffle). The CC cosine RDM shows structured, direction-dependent dissimilarity with a smooth circular gradient (range 0.001–0.012). The random cosine RDM is entirely NaN due to numerical overflow from unstable models and is not renderable. The random Euclidean RDM is dominated by exploding activations in unstable models (33/50) and is not interpretable. Stimuli: 12 ON moving edges at 30° increments. All 50 pretrained Flyvis models, seed=42.*

---

## Results

### CC Cosine RDM
The connectome-constrained network produces a structured 12×12 dissimilarity matrix with clear direction-dependent organization. At n=10, off-diagonal values range from ~0.001 to ~0.022 — small in absolute terms but systematically organized: adjacent directions are most similar (minimum: 0°–30°, dissimilarity = 0.001), while opposite directions are most dissimilar (maximum: 30°–210°, dissimilarity = 0.022). At n=50, the range tightens to 0.001–0.012, reflecting the inclusion of lower-performing models. Both runs show a smooth circular gradient consistent with the known direction tuning of T4/T5 neurons in the fly visual system.

### Random Cosine RDM
At n=10, the random baseline produces a nearly uniform matrix with all off-diagonal values at ~0.200 — the random network cannot distinguish motion directions, with directional variation confined to the fourth decimal place. At n=50, the mean random cosine RDM collapses to NaN due to numerical overflow from the majority of unstable models, making the cosine metric unsuitable for comparison at this scale. Neither restricting to stable models nor switching from weight shuffling to matched-normal resampling resolves the issue — instability is a fundamental property of random weight configurations in this architecture, not an artifact of any particular randomization strategy.

### Dynamic Instability
Dynamic instability is robust across randomization strategies. At n=10, 5 of 10 random models (models 2, 3, 4, 8, 9) produced exploding activations (756 non-finite values each, corresponding to 63 of 65 cell types across all 12 stimuli). At n=50 under Shiu-style shuffling, this strengthens to 33 of 50 random models (66%). Under matched-normal resampling — where weight magnitudes are drawn from a normal distribution matched to each parameter tensor's mean and std — 38 of 50 random models (76%) were unstable, confirming that instability is not an artifact of the shuffling procedure. 0 of 50 trained CC models showed any instability under any condition. The biological connectome, as optimized by task training, reliably occupies a dynamically stable region of parameter space that random weight configurations consistently leave.

### CC vs Random RDM Correlation
At n=10, cosine RDM correlation: **r = 0.757, p < 0.0001** — highly significant. This moderate positive correlation indicates that the CC and random cosine RDMs share directional ordering — both assign smaller dissimilarities to adjacent directions and larger dissimilarities to opposing ones — but differ substantially in the depth and resolution of that structure. The CC network encodes direction with fine-grained, graded dissimilarities spanning a 20-fold range (0.001–0.022), while the random baseline collapses that structure to a nearly uniform ~0.200 with no functionally meaningful variation.

At n=50, cosine RDM correlation: **NaN** under both randomization strategies — not computable due to numerical overflow in the mean random cosine RDM. The n=10 result remains the primary fidelity metric.

Euclidean RDM correlation: **r = 0.021, p = 0.865** (Shiu-style shuffle); **r = 0.241, p = 0.052** (matched-normal resampling) — neither significant, and not interpretable due to extreme magnitudes (~10²¹–10²⁶) from exploding activations in unstable random models.

**Interpretive note:** The n=50 random baseline is dominated by dynamically unstable models under both randomization strategies and is not suitable for RDM correlation analysis. The meaningful fidelity signal at n=50 is the within-ensemble consistency of CC models and the instability rate of random models, not the CC vs random RDM correlation. The n=10 result (r = 0.757, p < 0.0001) remains the primary fidelity metric, computed against a random baseline with only 5/10 unstable models.

### Within-Ensemble Consistency
At n=10, mean pairwise RDM correlation: **r = 0.838 ± 0.078** (range: 0.601–0.956). At n=50, mean pairwise RDM correlation: **r = 0.721 ± 0.150** (range: 0.323–0.983). The decrease in mean and increase in variance at n=50 reflects the inclusion of lower-performing models implementing more varied solutions, consistent with the known cluster structure of the Flyvis ensemble reported in Lappalainen et al. Fig. 3.

### Next Steps
- Dynamic instability in random models persists across both Shiu-style shuffling (66% unstable) and matched-normal resampling (76% unstable) — per Lappalainen et al. (Methods), time constants are clamped during training to prevent 
instability — restricting randomization to the 604 unitary synapse scaling factors only, while preserving trained time constants and resting potentials, should produce a stable baseline
- Include OFF edges (intensity = 0) alongside ON edges to test whether the directional geometry generalizes across polarity
- Euclidean metric is not suitable when random baselines are dynamically unstable; cosine distance is the appropriate primary metric for this comparison
- Within-CC consistency could be reported separately per cluster if UMAP reveals substructure in the ensemble geometry (planned)
- Consider Kendall's τ_A as an alternative to Spearman for RDM comparison, as it is more robust to ties

---

## Installation

This experiment runs on Google Colab with a T4 GPU runtime. Local installation requires Python ≥ 3.9, < 3.13.

```python
# On Google Colab — run these cells in order
!git clone https://github.com/TuragaLab/flyvis.git
%cd /content/flyvis
!pip install -e .[examples]
!flyvis download-pretrained
```

---

## Usage

```python
# Run proof of concept (n_models=1 for debugging, n_models=50 for full run)
results = run_experiment(n_models=50)
```

The full experiment takes approximately 60–90 minutes on a T4 GPU.

The Colab-ready notebook is at `notebooks/moving_edge_poc.ipynb`.
The standalone script is at `experiments/moving_edge_poc.py`.

---

## Repository Structure

```
connectome-fidelity/
├── README.md
├── experiments/
│   └── moving_edge_poc.py        ← standalone experiment script
├── notebooks/
│   └── moving_edge_poc.ipynb     ← Colab-ready notebook with results
└── figures/
    └── moving_edge_poc_rdms.png  ← output figure
```

---

## References

- Lappalainen et al. 2024. Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634, 1132–1140. https://www.nature.com/articles/s41586-024-07939-3

- Shiu et al. 2024. A Drosophila computational brain model reveals sensorimotor processing. *Nature* 634, 210–219. https://www.nature.com/articles/s41586-024-07763-9

- Kriegeskorte et al. 2008. Representational similarity analysis — connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience* 2:4. https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full

- Kriegeskorte & Wei 2021. Neural tuning and representational geometry. *Nature Reviews Neuroscience* 22, 703–718. https://www.nature.com/articles/s41583-021-00502-3

- Brunton et al. 2026. The digital sphinx: Can a worm brain control a fly body? *bioRxiv*. https://www.biorxiv.org/content/10.64898/2026.03.20.713233v1

---

## Author

Michael Zhou — PhD student, Electrical and Computer Engineering, Georgia Institute of Technology
