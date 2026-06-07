# ESM Variant-Effect Tools

A small, teaching-oriented toolkit for predicting and analyzing the effects of
protein mutations using **ESM (Evolutionary Scale Modeling)** protein language
models. It was developed for an *Experimental Systems Biology* lab and provides
three command-line scripts that take you from raw protein sequences to scored,
visualized, and clustered variant-effect predictions.

> **Educational resource.** These tools are intended for teaching and research.
> ESM predictions are useful approximations — not ground truth — and are most
> valuable for prioritizing and exploring large sets of variants.

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kouroshSA/ESM-variant-effect-tools/blob/master/colab/ESM_variant_effect_tools.ipynb)

Click the badge to open a ready-to-run notebook — **no local installation
required**. It clones this repository, installs the dependencies, runs the full
demo workflow on a free GPU, and lets you upload your own sequences. Just set
**Runtime → Change runtime type → GPU** first. The notebook lives in
[`colab/ESM_variant_effect_tools.ipynb`](./colab/ESM_variant_effect_tools.ipynb).

## What are ESM models?

**ESM (Evolutionary Scale Modeling)** is a family of open-source *protein
language models* from Meta AI / FAIR. Just as a natural-language model is
trained on text, ESM models are trained on hundreds of millions of natural
protein sequences with a masked-language-modeling objective: residues are hidden
and the model learns to predict them from their context. In doing so the models
absorb statistical patterns of evolution, structure, and function directly from
sequence.

A trained ESM model can therefore be used, *without any task-specific
fine-tuning* ("zero-shot"), to:

- **Score mutations** — by comparing the model's predicted probability of the
  wild-type residue versus a mutant residue at a given position (a *delta
  log-probability*). Strongly negative scores flag substitutions the model finds
  unlikely/disruptive.
- **Generate** plausible novel sequences, and
- **Predict structure** (e.g., ESMFold), depending on the model variant.

Two model lines are relevant here:

- **ESM-1v** (`esm1v_t33_650M_UR90S_1` … `_5`) — 650M-parameter models trained on
  UniRef90 and **specialized for variant-effect prediction**. These are usually
  the best choice for the mutation-scoring tasks in this repo.
- **ESM-2** (`esm2_t6_8M` … `esm2_t36_3B`) — general-purpose models spanning a
  wide range of sizes; larger models capture more but cost much more to run.

For mutation-effect work, ESM-1v is the recommended default.

## The tools

The repository contains three scripts that form a pipeline:

### 1. `ESM_predict_mutation_effects.py` — mutation-effect predictor
Takes a FASTA file, enumerates every single amino-acid substitution (optionally
restricted to chosen positions), runs an **ESM-1v** model, and writes a CSV of
*delta log-probability* scores for each substitution. Supports CUDA and
multi-GPU execution.

> `ESM_predict_mutation_effects_any_model.py` is a sibling of this script that
> lets you load **any** ESM model from the ESM hub by name (`--model_name`)
> instead of being limited to the five ESM-1v models. Use it if you want to
> experiment with ESM-2 or other variants; for standard variant-effect scoring,
> the ESM-1v predictor above is recommended.

### 2. `visualize_mutation_effects_indel_adjust_logo2.py` — analyzer & visualizer
Reads the prediction CSV and produces, per sequence, a **heatmap**, a
**per-position boxplot**, an **average-effect bar plot by amino acid**, and a
**sequence logo**. It aligns each variant to the wild type, so it correctly
handles insertions and deletions (indels).

### 3. `calculate_variant_scores_with_proxy_PCA-annoy.py` — group variant analyzer
Aligns a set of variants to a wild-type (or WT-proxy) sequence, computes a
**cumulative detrimental score** per variant, and optionally runs **PCA** and an
**Annoy** approximate-nearest-neighbor search to find the variants most similar
to a chosen query in mutation-effect space. Tip: use a closely related sequence
as a WT proxy so the reference's own effect vector is not all zeros.

## Installation

The tools target Python 3.10. The pretrained ESM weights are downloaded
automatically on first use.

```bash
conda create -n esm python=3.10
conda activate esm
# Install PyTorch (CUDA build shown; see pytorch.org for your platform):
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

A GPU is strongly recommended — exhaustively scoring every substitution in a
protein is compute-intensive.

## Quick start

A short end-to-end walkthrough is in [`esm_demo.txt`](./esm_demo.txt). In brief:

```bash
# 1. Predict per-substitution effects for the demo variants:
python ESM_predict_mutation_effects.py LDH_variants.fasta LDH_variants.csv --cuda

# 2. Score a group of variants against a WT reference, with PCA + nearest-neighbors:
python calculate_variant_scores_with_proxy_PCA-annoy.py \
    WT_LDHA.fasta LDH_variants.fasta WT_LDHA.csv cumulative_scores.csv --perform_pca

# 3. Visualize the predicted mutation effects:
python visualize_mutation_effects_indel_adjust_logo2.py \
    output_predictions.csv out_dir \
    --wt_sequence WT_LDHA.fasta --variant_sequences LDH_variants.fasta \
    --positions 33 66 --ranges 24-33
```

### Included demo data

- `input_sequences.fasta` — 5 example protein sequences.
- `WT_LDHA.fasta` / `LDH_variants.fasta` — a wild-type LDHA sequence and a small
  set of variants for the demo workflow.

## Acknowledgments & citation

These tools are wrappers around the **ESM** models and the `fair-esm` library by
Meta AI / FAIR (https://github.com/facebookresearch/esm), distributed under the
MIT License. All credit for the underlying models belongs to their authors. If
you use this toolkit in academic work, please cite the relevant ESM papers:

- Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A. (2021).
  *Language models enable zero-shot prediction of the effects of mutations on
  protein function.* Advances in Neural Information Processing Systems (NeurIPS).
  bioRxiv: https://doi.org/10.1101/2021.07.09.450648  *(ESM-1v)*
- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., et al. (2023).
  *Evolutionary-scale prediction of atomic-level protein structure with a
  language model.* Science, 379(6637), 1123–1130.
  https://doi.org/10.1126/science.ade2574  *(ESM-2 / ESMFold)*
- Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., et al. (2021).
  *Biological structure and function emerge from scaling unsupervised learning
  to 250 million protein sequences.* PNAS, 118(15), e2016239118.
  https://doi.org/10.1073/pnas.2016239118  *(ESM-1b)*

## License

Original code in this repository is released under the [MIT License](LICENSE).
The ESM models and `fair-esm` are third-party dependencies under their own MIT
License and are not redistributed here (see the third-party notice in `LICENSE`).

## Credits

Developed by Kourosh Salehi-Ashtiani, with assistance from AI tools
(ChatGPT and Claude).
