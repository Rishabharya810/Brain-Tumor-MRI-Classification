# Brain Tumor MRI: Why 99% Is a Leak — measured on three exams

Four-class brain-tumor MRI classification (glioma / meningioma / no tumor / pituitary) on the most-used public dataset for this task — built as an honest pipeline that audits its own benchmark, then scores every model **three ways**: the official benchmark, its strictly de-contaminated remnant, and a **family-disjoint re-split** (re-drawn so the same patient's scans can never appear on both sides) with every model retrained from zero.

**The short version:** eight models trained under one honest protocol land within **3 points** of each other (92.9–96.1%) on the official 1,600-image test — including **the exact recipe behind the famous "99% accuracy" notebook** (ImageNet-pretrained Xception, fully fine-tuned), which scores **95.88% here, not 99**. The audit shows why: **73% of the test images share a perceptual hash with the training split** (same patients on both sides). On the 438-image uncontaminated remnant every model drops 8–10 points (best: 88.1%); on the fair family-disjoint exam most models score *at or above* their official marks (94.9–97.1%), matching published patient-disjoint results. The leak's real damage is not inflating good models — it's letting weak protocols fake 99 while hiding the benchmark's hard core. An ensemble of five models gains nothing anywhere (more saturation evidence), and the best model is stable across three seeds (95.98 ± 0.07). **In an independent earlier run, the best model was TinyViT-5M (96.12%) — the flip to EfficientNet-B0 (96.06%) across runs is itself a saturation finding.**

> **Headline: the famous 99% is a protocol artifact. Under an honest protocol the same recipe scores 95.88. Models genuinely generalize at 95–97% — but only the audit can tell you which part of any score is memory.**

---

## Why this project looks the way it does

This is a rebuild of an earlier course project that reported 99.31%. A forensic re-audit of that original run found it had trained on CPU while being presented as a cloud-GPU run, executed with a **broken scikit-learn** (its `roc_auc_score` literally returned 0.5 — every metric was a homemade approximation), and duplicated rows across sessions. Rather than patch the numbers, I rebuilt the pipeline from scratch with the discipline the original lacked: stage-gated execution, atomic artifacts, real metrics, seeded everything, and a data-quality audit *before* trusting any accuracy. The full original-flaws → fixes table is in the [notebook](notebooks/brain_tumor_mri.ipynb) and [CHANGELOG](CHANGELOG.md).

The rebuild's headline came out lower than the original's — which prompted the question this project actually answers: why do all projects on this dataset hover near 99%? The audit below is the response.

## The data — and its leakage problem

**Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (CC BY 4.0), Version 2 (Feb 2026), 5,600 training / 1,600 testing images, balanced 1,400/400 per class. It combines three public collections — [Figshare (Cheng)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), [SARTAJ](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri), and [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) — scans of ~233 patients total, slice by slice. The exact snapshot is pinned in this repo with SHA-256 checksums of all 7,200 files ([data/DATASET_SNAPSHOT.md](data/DATASET_SNAPSHOT.md)).

Version 2 already did real cleaning: duplicates removed, classes balanced, **exact train↔test copies eliminated** (our MD5 audit confirms: zero). But the deeper problem survives, because slice-level scans of the same patient sit on both sides of the split:

| Audit finding (this run, recomputed in the notebook) | Count |
|---|---|
| Exact train↔test copies (the flaw Version 2 fixed) | **0** |
| Test images with a near-duplicate (pHash ≤ 8) in the raw Training folder | 1,248 / 1,600 (78%) |
| Test images with a near-duplicate in the **actual 4,480-image training split** | **1,162 / 1,600 (73%)** |
| Exact duplicates within train / within test | 171 / 16 |
| Dual-evidence near-duplicate links (pHash ≤ 8 **and** deep-feature cosine ≥ 0.95) | 4,121 |
| Families (connected components of dual links) / largest / cross-class | 5,279 / 26 / 4 |

This independently replicates what 2025–26 medical-imaging literature reports for this dataset family (see *Related work*). One rule this project enforces: accuracies are never compared across dataset editions — the original 99.31% was scored on the 2021 Version 1 with a different split.

## What the pipeline does

Every stage writes its artifacts and marks itself complete in a manifest; re-running skips finished work, interrupted training resumes mid-epoch, and the final summary is assembled *only* from the stage records (a contradictory number is structurally impossible):

| Stage | What it does |
|---|---|
| 1–4 | Environment record, dataset integrity, duplicate/near-duplicate audit, seeded split + EDA |
| 5–6 | One honest training loop (AdamW, cosine, early stop on validation macro-F1) applied to **eight models**: a 90k-parameter true-from-scratch baseline; five lightweight ImageNet-initialized architectures; ResNet-50 (the literature's anchor); **Xception-Reference — the "99% club" recipe itself** (ImageNet-pretrained, fully fine-tuned at its native 299×299) |
| 7 | Comparison (rebuilt fresh from per-model JSONs) |
| 8 | Grad-CAM on the best model — correct + misclassified cases, with the failure profile |
| 9 | ONNX export + parity verification (logits must match PyTorch to ~1e-5) |
| 10 | Inference speed on the T4 (warmup, medians, synchronized timing) |
| 11 | **Clean-subset evaluation** — re-score all 8 models on the 438 uncontaminated test images, per class |
| 12 | **Seed stability** — best model × 3 seeds, mean ± std |
| 13 | **Family-disjoint re-split** — dual-evidence families, whole-family stratified re-split, **all 8 models retrained from zero**, scored on the 1,440-image exam where leakage is structurally impossible |
| 14 | **Ensemble** of the five lightweights (softmax-vote), scored on all three exams |
| 15 | SUMMARY.md assembled from the records only |

## Results — one project, three exams

**Tier 1 — the official benchmark** (1,600 images, as shipped):

| Model | Params | Official acc | Macro-F1 |
|---|---|---|---|
| **EfficientNet-B0** (best; 3-seed: **95.98 ± 0.07**) | 4.0M | **96.06%** | 0.9601 |
| **Xception-Reference (the "99%" recipe)** | 20.8M | **95.88%** | 0.9582 |
| DeiT-Tiny | 5.5M | 95.50% | 0.9543 |
| TinyViT-5M | 5.1M | 95.19% | 0.9510 |
| MobileViT-S | 4.9M | 94.88% | 0.9480 |
| MobileNetV3-Small | 1.5M | 94.31% | 0.9423 |
| ResNet-50 | 23.5M | 92.94% | 0.9285 |
| Baseline-CNN (true from-scratch) | 0.09M | 65.25% | 0.6460 |

All seven real models inside **3.1 points**. The famous recipe does not reproduce its 99 — that number requires the weak protocol (minimal augmentation, 10 epochs, no per-class analysis) *plus* the leak.

**Tiers 2 and 3 — the same eight models, honest exams:**

| Model | Official | Clean remnant (438) | Family-disjoint (1,440) |
|---|---|---|---|
| EfficientNet-B0 | 96.06% | **88.13%** | **97.08%** |
| Xception-Reference | 95.88% | 87.90% | 96.60% |
| DeiT-Tiny | 95.50% | 86.53% | 95.56% |
| TinyViT-5M | 95.19% | 86.99% | 96.74% |
| MobileViT-S | 94.88% | 86.07% | 95.21% |
| MobileNetV3-Small | 94.31% | 84.47% | 94.86% |
| ResNet-50 | 92.94% | 84.02% | 95.49% |
| Baseline-CNN | 65.25% | 56.62% | 66.46% |
| *Ensemble of five* | *95.81%* | *87.90%* | *97.50%* |

**How to read the three tiers — the project's central finding:**

1. **The fair family-disjoint exam (tier 3) shows the models genuinely generalize at ~95–97%** — most score *at or above* their official marks, in line with published patient-disjoint results (SSRN 97.5, Bassi 95.2, WICA-Net 95.25). For an honest protocol the leak adds little.
2. **The strictly de-contaminated remnant (tier 2) is systematically *harder* — 8–10 points lower for everyone.** This is not contradiction; it is selection bias made visible: the benchmark's contaminated images are precisely the ones covering its hard core (the notumor test class was 90% contaminated — only 40 of 400 images survive de-contamination). Naive de-contamination therefore *underestimates* skill; the family-disjoint re-split is the fair way to measure.
3. **The leak's real victims are the claims, not the models:** it converts weak protocols into 99% headlines (memorization), masks the benchmark's hard core, and compresses architecture differences into noise (below).

**Per-class honesty (best model):** glioma recall falls from 0.855 (official) to **0.705** on the clean remnant — the glioma advantage was partly memorization — while notumor stays at 1.00 (only 40 survive) and meningioma/pituitary hold at 0.97–0.99. The Grad-CAM failure profile agrees: errors concentrate in glioma→meningioma (34) and glioma→notumor (18), the distinctions clinicians also find hardest.

**The ensemble buys nothing — saturation quantified:** softmax-voting five models together scores 95.81% official (**below** the best single model at 96.06%), 87.90% clean (**below** the best single at 88.13%), 97.50% disjoint (within the band). When five diverse architectures plus their vote all land in the same band, the benchmark is saturated: what's left is not reachable by model choice.

**The best model flips between runs — saturation from yet another angle.** In an independent earlier run (identical code, identical seed, same GPU type), TinyViT-5M won at 96.12% while EfficientNet-B0 scored 95.19%. In this run, EfficientNet-B0 wins at 96.06% and TinyViT-5M scores 95.19%. The ranking is not stable — the models are statistically indistinguishable on this benchmark.

**Inference speed (T4, batch-1 median):** Baseline 0.96 ms; MobileNetV3-Small 5.1 ms; DeiT-Tiny 5.1 ms; ResNet-50 6.0 ms; EfficientNet-B0 7.9 ms; TinyViT-5M 8.3 ms; MobileViT-S and Xception-Reference 8.4 ms. ONNX exports verified against PyTorch to ≤7.3e-5 max logit difference (PASS).

## Honest limitations

- **Research prototype — not for clinical use.** No external, multi-institution, or prospective validation.
- The family-disjoint tier is leak-proof w.r.t. **dual-evidence families** (pHash ≤ 8 *and* embedding cosine ≥ 0.95 — the published standard for this dataset); weaker same-patient similarity could in principle survive it. The clean remnant is the strict bound, the fair exam the best estimate.
- The clean remnant is selection-biased (documented above) — it is a *pessimistic* bound, not an estimate.
- Per-image accuracy ≠ per-patient accuracy; patient IDs do not exist in this dataset, so a true patient-level split is impossible — the family clustering is the best available proxy and is disclosed in full.
- Single dataset edition (Version 2, Feb 2026); results are not comparable to numbers produced on Version 1 (2021) — including the original project's 99.31%.
- Run-to-run GPU nondeterminism moves individual scores by ~±0.3 pp (measured: 3 seeds); the *conclusions* are stable across two independent full runs — but the best-model *label* flipped from TinyViT-5M to EfficientNet-B0 between them (see the results section).

## Related work

- Zhang, Xu, Qin & Li — *Near-Duplicate Leakage in Brain Tumor MRI Classification: Cluster-Disjoint Re-Evaluation* ([SSRN, DOI 10.2139/ssrn.7493561](https://doi.org/10.2139/ssrn.7493561)): 68.8% test-side near-duplicates; ResNet-18 scores 99.73% on matched vs 85.40% on unmatched test images; cluster-disjoint best 0.9748. **This project independently replicates their leakage rate (73% vs 68.8%) and their disjoint tier (95–97%).**
- Bassi — *Brain Tumor MRI Benchmark Audit* ([Zenodo 22721030](https://zenodo.org/records/22721030), [code](https://github.com/imjbassi/brain-tumor-mri-classification)): patient mapping of this dataset's Figshare lineage; patient-disjoint CV 99.22% → 95.19%; exact-dedup alone changes nothing.
- Khan et al. — WICA-Net-M, *Computers* 15(9):586 ([DOI](https://doi.org/10.3390/computers15090586)): 99.42% standard vs 95.25% deduplicated; "patient-level leakage remains unresolved."
- The 99% club on the older Version-1 split: 99.86% ([arXiv 2509.24149](https://arxiv.org/abs/2509.24149)), 99.62% ([arXiv 2411.15596](https://arxiv.org/abs/2411.15596)), 98.73% (*Cancers* 2025, [DOI](https://doi.org/10.3390/cancers17010121)).
- The audit tradition this project follows: Recht et al., *Do ImageNet Classifiers Generalize to ImageNet?* ([arXiv 1902.10811](https://arxiv.org/abs/1902.10811)); Barz & Denzler, *Purging CIFAR of Near-Duplicates* ([arXiv 1902.00423](https://arxiv.org/abs/1902.00423)); Northcutt et al., *Pervasive Label Errors* ([arXiv 2103.14749](https://arxiv.org/abs/2103.14749)).

## How to reproduce

New to deep learning? Every concept and formula used here — splits, cross-entropy, AdamW, perceptual hashes, the metrics, all of it — is explained from zero in **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

Everything ran on a single Kaggle Tesla T4 in **~8 hours**, executed top-to-bottom with GPU enabled and the dataset attached ([notebooks/brain_tumor_mri.ipynb](notebooks/brain_tumor_mri.ipynb) is the actual executed run, outputs included — also [live on Kaggle](https://www.kaggle.com/code/rishabarya810/brain-tumor-mri-why-99-is-a-leak-73-audit), where **Copy & Edit** runs it on your own GPU quota).

**Run it yourself, on your own PC:** clear the outputs of [notebooks/brain_tumor_mri.ipynb](notebooks/brain_tumor_mri.ipynb) (Jupyter: *Edit → Clear all outputs* — or **Copy & Edit** on Kaggle for a fresh copy that runs on your own GPU quota). `pip install -r requirements.txt`, unzip the pinned dataset snapshot (from the [Release](../../releases) — its checksums are in [data/](data/DATASET_SNAPSHOT.md)) next to the notebook, and run top-to-bottom: the prep cell auto-links the dataset (a directory junction on Windows), installs the extra packages, and every stage is idempotent and resumable — interrupt anywhere, re-run, and it continues instead of restarting. The stage-gated design makes the whole notebook re-runnable without recomputing anything. Environment: [requirements.txt](requirements.txt), recorded from the run (Python 3.13.15, torch 2.11.0+cu128, timm 1.0.29). **Trained weights** (all 18 trainings: 8 official + 2 seed-stability + 8 family-disjoint) and the **pinned dataset snapshot** are attached to the GitHub *Release*; the repo's checksums verify both.

## Repository map

```
├── README.md                     this file
├── CHANGELOG.md                  the project's evolution: v1 → v8 → this run
├── docs/
│   └── METHODOLOGY.md            the zero-background reader's layer (every concept + formula)
├── requirements.txt              pinned environment recorded from the actual run
├── notebooks/
│   └── brain_tumor_mri.ipynb        the full pipeline — as executed on the T4 (all outputs included); clear outputs to run it yourself
├── data/                         pinned dataset snapshot: card + SHA-256 manifest
│   └── LICENSES/                 CC BY 4.0 text + TASL attribution
└── results/                      every artifact the run produced
    ├── SUMMARY.md                the run's self-written report
    ├── model_comparison.csv · benchmarks.csv · clean_subset_report.json
    ├── cluster_report.json · ensemble_report.json · seed_stability.json
    ├── data_quality.json · environment.json · split.json
    └── plots/                    EDA, confusion matrices, ROC, training curves, Grad-CAM
```

## Credits & license

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar — **CC BY 4.0** (full attribution in [data/LICENSES/THIRD-PARTY-NOTICES.md](data/LICENSES/THIRD-PARTY-NOTICES.md)); underlying sources: Figshare (Cheng et al.), SARTAJ, Br35H.
- Code: MIT (see [LICENSE](LICENSE)).
- Built by Rishab Arya · M.Tech CSE (Data Science), Bennett University.

### Authorship note

The code in this project was developed with an LLM coding agent (ZCode, running GLM-5.3) under my direction. The research questions, the audit design (near-duplicate thresholds, the three-exam evaluation, the family-disjoint re-split), and every modeling decision are mine; I reviewed the code and results stage by stage and verified every number against the pipeline's own artifacts. I can walk through any stage of this project without the assistant.
