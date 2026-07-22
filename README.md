# Representational Geometry as a Fidelity Metric for Connectome-Constrained Neural Emulations

**Evidence from *Drosophila* and Mouse Visual Systems.** Testing whether the pattern of similarity between a population's responses to different stimuli, representational geometry, can distinguish real connectome wiring from random wiring where behavior alone cannot, directly extending Brunton et al. (2026)'s finding that behavioral fidelity doesn't require biological fidelity.

**Author:** Michael Zhou · **Advisor:** Prof. Jennifer Hasler

## Paper Details

**Preprint:** [doi.org/10.64898/2026.06.10.731214](https://doi.org/10.64898/2026.06.10.731214)

⚠️ The published preprint predates most of the content in this README (item 1's untrained-null-scheme extension, item 3's Method B, the OFF-polarity Erdős–Rényi finding, item 5's full K=8 result). This README is the current, up-to-date account; treat the DOI'd version as the last formally published snapshot, not the latest results.

```bibtex
@article{zhou2026representational,
  title     = {Representational geometry as a fidelity metric for connectome-constrained networks: evidence from the Drosophila visual system},
  author    = {Zhou, Michael G. and Hasler, Jennifer O.},
  journal   = {bioRxiv},
  year      = {2026},
  doi       = {10.64898/2026.06.10.731214},
  url       = {https://doi.org/10.64898/2026.06.10.731214}
}
```

## Table of Contents

- [Paper Details](#paper-details)
- [Setup Requirements](#setup-requirements)
- [Quick Start](#quick-start)
- [Data & Reproducibility](#data--reproducibility)
- [Key Findings](#key-findings)
- [Background](#background)
- [Full Methodology & Results](#full-methodology--results)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Setup Requirements

- **Python 3.12**
- **[Flyvis](https://github.com/TuragaLab/flyvis) 1.1.3** (`pip install -e .[examples]`), the connectome-constrained *Drosophila* visual system model this project is built on
- **PyTorch with CUDA** (developed against CUDA 12.8)
- **GPU**: all training and evaluation in this project ran on a single Quadro RTX 8000 (46GB); any CUDA-capable GPU with comparable memory should work, but per-checkpoint evaluation is CPU/single-GPU-bound, not something that benefits from multi-GPU parallelism (tested directly, found slower, not faster)
- `scipy`, `numpy`, `matplotlib` for the analysis and plotting scripts

## Quick Start

Once Flyvis is installed and the required data is in place (see [Data & Reproducibility](#data--reproducibility) below), the untrained real-vs-random comparison from item 1 can be reproduced with:

```bash
python test_item1_all_null_schemes.py \
    --n_models 10 --n_permutations 10000 \
    --results_root /path/to/trained/null/networks \
    --polarity on_off --checkpoint first
```

This is a real, verified command from this project's own analysis pipeline, not a simplified illustration. Swap `--checkpoint first` for `--checkpoint last` to reproduce the trained comparison, or `--stimulus_set henning_8dir` for the Henning condition.

## Data & Reproducibility

Reproducing this work end to end requires two things that aren't bundled in this repository:

1. **The pretrained Flyvis ensemble.** `flyvis download-pretrained` does **not** provide the `flow/0000/000...049` ensemble structure this project's scripts expect, this was confirmed directly during this project (the download succeeds and passes its checksum, but unpacks connectome/rendering data, not trained model checkpoints). Where this real ensemble came from originally is not fully resolved; the practical workaround used throughout this project is the saved population-vector files below, which don't require re-deriving the ensemble at all.
2. **The trained null-scheme networks** (degree-preserving swap, Erdős–Rényi), custom-trained for this project through Flyvis's own training pipeline. These are not currently packaged for external distribution.

What *is* directly reusable without either of the above: the saved real-CC population matrices (`results_exp1_50models_full_shiu.npz`, `results_exp2_50models_full_shiu.npz`, `results_exp1_8dir_50models_full_shiu.npz`), each containing pre-computed per-model population vectors and RDMs for the pretrained ensemble under a specific stimulus condition. Any of item 1 or item 3's real-CC-side comparisons can be rerun directly against these files.

## Key Findings

- **Untrained real wiring is clearly distinguishable from random wiring** by representational geometry, for every null scheme and stimulus condition tested (Table 0).
- **Once both are actually trained**, that distinction narrows substantially, and in one specific case (Erdős–Rényi, OFF-polarity structure) closes completely to statistical indistinguishability, the direct test of Brunton et al.'s untested prediction (item 3).
- **Convergence with training is abrupt, not gradual**: nearly the entire shift happens within the first few percent of a network's training run, then plateaus for the rest (Figure 6).
- **What determines a network's individual fidelity trend remains genuinely unresolved**: no significant evidence on the more trustworthy reference that wiring realization drives the direction, which leans toward training-process randomness, but real heterogeneity across networks (some cluster tightly, others scatter or flip) means this could equally reflect the test remaining underpowered at n=8, not a clean resolution either way (item 5).
- **The original biological reference was invalidated and replaced**: Maisak et al. (2013) turned out to be dominated by circular stimulus structure rather than real tuning signal; the Henning et al. (2022) dataset provides a validated non-circular replacement (item 2).
- **On mouse visual cortex (MICrONS)**, wiring reliably predicts functional geometry under fixed, non-trained simulation. Real connectivity beats distance-constrained and cell-type-shuffled null wiring decisively, but is not distinguishable from a degree-preserving null, suggesting degree structure specifically, not finer wiring detail, accounts for the connectome's resemblance to real neural activity. Whether excitatory/inhibitory structure carries additional signal beyond degree remains an open, unresolved question.

## Background

Brunton et al. (2026) demonstrated that a connectome, taken from one species (*C. elegans*) and used to control the body of another (*Drosophila*), can produce realistic behavior even when only a downstream interface is trained. The connectome's own synaptic weights and cellular parameters were never optimized; behavior emerged entirely from training the decoder mapping its outputs to the target body. The authors note this role could be fulfilled equally well by a randomly connected network, since all the learning happens in the decoder, this shows behavioral fidelity is achievable without biological fidelity: a model can look right without its underlying structure being correct. That raises a direct question: **if behavior alone can't verify fidelity, what can?**

This project tests one candidate, **representational geometry**, asking whether the structure of a population's response patterns can distinguish a real connectome from a random one, in a way behavior alone cannot.

## Full Methodology & Results

The complete, item-by-item methodology, tables, and figures behind the findings above live in **[METHODS.md](METHODS.md)**: five experiments building on each other (real-vs-random wiring untrained and trained, the biological-reference validation, the training-vs-wiring analysis, and the retraining-based instability test), plus the parallel mouse-connectome work and the full answer to the Brunton question.

## Acknowledgments

- Built on [Flyvis](https://github.com/TuragaLab/flyvis) (Lappalainen et al. 2024), the connectome-constrained *Drosophila* visual system model this project's entire fly-side analysis depends on.
- The mouse-side work uses the MICrONS connectome and co-registered functional recordings.
- Thanks to Prof. Jennifer Hasler for ongoing advising, and to Prof. Hannah Choi for feedback that shaped several of the methodological corrections in this document.
- This project exists because of Brunton et al. (2026)'s Digital Sphinx finding, which raised the question this whole line of work tries to answer.

## References

Brunton, B. W., Abe, E. T. T., Hu, L. J., & Tuthill, J. C. (2026). The digital sphinx: Can a worm brain control a fly body? *bioRxiv*. https://doi.org/10.64898/2026.03.20.713233

Ding, Z., Fahey, P. G., Papadopoulos, S., et al. (2025). Functional connectomics reveals general wiring rule in mouse visual cortex. *Nature*, 640, 459. https://doi.org/10.1038/s41586-025-08840-3

Frankle, J., Dziugaite, G. K., Roy, D., & Carbin, M. (2020). Linear mode connectivity and the lottery ticket hypothesis. *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 3259–3269. https://proceedings.mlr.press/v119/frankle20a.html

Henning, M., Ramos-Traslosheros, G., Gür, B., & Silies, M. (2022). Populations of local direction-selective cells encode global motion patterns generated by self-motion. *Science Advances*, 8, eabi7112. https://doi.org/10.1126/sciadv.abi7112

Kim, S. H., & Choi, H. (2025). Inhibitory cell type heterogeneity in a spatially structured mean-field model of V1. *bioRxiv*, 2025.03.13.643046. https://doi.org/10.1101/2025.03.13.643046

Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4. https://doi.org/10.3389/neuro.06.004.2008

Lappalainen, J. K., Tschopp, F. D., Prakhya, S., et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature*, 634, 1132–1140. https://doi.org/10.1038/s41586-024-07939-3

Maisak, M. S., Haag, J., Ammer, G., Serbe, E., Meier, M., Leonhardt, A., Schilling, T., Bahl, A., Rubin, G. M., Nern, A., Dickson, B. J., Reiff, D. F., Hopp, E., & Borst, A. (2013). A directional tuning map of *Drosophila* elementary motion detectors. *Nature*, 500(7461), 212–216. https://doi.org/10.1038/nature12320

The MICrONS Consortium. (2025). Functional connectomics spanning multiple areas of mouse visual cortex. *Nature*, 640, 435–447. https://doi.org/10.1038/s41586-025-08790-w
