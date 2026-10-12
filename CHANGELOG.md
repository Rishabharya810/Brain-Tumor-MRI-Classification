# Changelog — how this project evolved

This project is not one notebook; it is four iterations, each of which existed because the previous one taught
us something. The format follows [Keep a Changelog](https://keepachangelog.com/), adapted for research code.
Everything below is documented in detail in the [notebook](notebooks/brain_tumor_mri.ipynb) and the
per-version artifacts referenced in each entry.

## [v8] — T4 edition, the published version — 2026-09-29

The whole pipeline rebuilt for one Tesla T4 on Kaggle: five ImageNet-initialized lightweight architectures plus one truly-from-scratch 90k-parameter baseline, every stage
idempotent and resumable, and the benchmark itself audited before any number was trusted.

**Added**
- **BaselineCNN** (Stage 6a): a ~90k-parameter, three-conv-block model with no pretraining — the reference
  point every architecture has to justify itself against (it reaches 64.06%).
- **Clean-subset evaluation** (Stage 11): the 438 test images with no perceptual-hash near-duplicate in the
  training split, re-scored for every model — the dual-metric headline (96.31% → 89.50% for the best model).
- **Seed stability** (Stage 12): best model re-trained at seeds 43/44 → mean ± std reported (95.92% ± 0.28 pp).
- **Grad-CAM failure analysis**: misclassified samples labelled true→predicted, plus the confusion-pair
  profile (glioma→meningioma 30, glioma→notumor 18).
- Dataset-version comparability note in the summary; pinned dataset snapshot + MD5 manifest in `data/`.

**Changed**
- Inference speed measured on the training GPU only (one protocol, CUDA-synchronized, median of 100/30 iters).

**Removed**
- The OpenVINO IR + NNCF INT8 stage. Reason: on the v7 run it cost ~30 GPU-minutes to produce a *negative*
  result (INT8 failed the ΔF1 ≤ 1pp gate even on CPU, and collapsed on the T4 GPU plugin) — an honest
  deployment finding, recorded in v7's artifacts, not worth re-running on a GPU box.

## [Publication run] — the three-exam edition — 2026-10-07

The version that shipped to Kaggle and GitHub: everything below plus the additions the final design review locked.

**Added**
- **ResNet-50** (the literature's anchor) and **Xception-Reference** — the exact "99% accuracy" notebook's recipe
  (ImageNet-pretrained, fully fine-tuned, native 299×299) — so the famous recipe could be scored honestly. It scores
  95.19% under this protocol, not 99.
- **Family-disjoint re-split** (Stage 13): dual-evidence near-duplicate families (pHash ≤ 8 AND deep-feature cosine ≥ 0.95),
  whole-family stratified split, every model retrained from zero → 1,440-image exam where leakage is structurally impossible.
- **Ensemble of the five lightweights** (Stage 14), scored on all three exams — it beats the best single model nowhere.
- **Per-class clean-subset analysis** — glioma recall falls 0.868 → 0.729 when contamination is removed.
- Seed stability now reports mean ± std for whichever model wins (TinyViT-5M: 95.79 ± 0.28).
- Story-mode notebook: table of contents, per-stage "why this exists" notes, references with DOIs.
- Corrected a wording error from earlier editions: the five architectures are ImageNet-initialized (transfer learning);
  only the 90k baseline trains truly from scratch.

**The finding, refined into three tiers:** official 95.1–96.1; strictly de-contaminated remnant 86.5–88.8 (selection-biased,
pessimistic); fair family-disjoint exam 94.4–97.2 — models genuinely generalize, the leak's real damage is converting weak
protocols into 99% headlines and hiding the benchmark's hard core.

## [Publication run v7 — final numbers] — 2026-10-10

The run that shipped to Kaggle and GitHub (identical code to v5; the H1 title corrected 89.5 → 88.81%).
GPU nondeterminism shifted individual scores within ±0.8 pp and **flipped the best model** from
TinyViT-5M (96.12%) to EfficientNet-B0 (96.06%) — itself further evidence of benchmark saturation.
All three-tier conclusions unchanged. Results at the time of this entry were from v7.

## [Publication run — the fresh verified kernel] — 2026-10-10

A fresh full run of the identical pipeline on a new Kaggle kernel: all 32 stages, zero errors,
~6h35m on one Tesla T4. Every number in this repo now comes from it. **Best model: Xception-Reference
— the "99% club" recipe itself — at 96.06% official / 89.27% clean remnant / 96.67% family-disjoint.**

**Added**
- **The third best-model flip, promoted to a finding**: across three independent full runs the winner
  changed every time — V5: TinyViT-5M (96.12%) → V7: EfficientNet-B0 (96.06%) → this run:
  Xception-Reference (96.06%) — while the top score stayed inside a ~1pp band. Architecture ranking
  on this benchmark is noise, and the famous recipe tops the honest protocol at ~96, not 99. The
  clean-remnant best moved 88.81 → 88.13 → 89.27 across the same runs (±0.6 pp run-to-run).
- **The glioma-disjoint nuance**: on the family-disjoint exam glioma recall balances out (Xception:
  0.97 disjoint vs 0.87 official) — the benchmark's famous glioma weakness is partly a leakage
  artifact, not a model failure.

**Changed**
- All result tables refreshed from the fresh run: official 62.31–96.06 with the seven pretrained
  models inside 2.3 points; clean remnant 52.74–89.27 (drops of 6–10 pp for everyone); family-disjoint
  66.67–96.67. Seed stability: Xception-Reference 95.81 ± 0.18 pp (96.06 / 95.62 / 95.75). Ensemble
  of the five lightweights: 95.19 / 85.84 / 96.94 (F1 0.9510 / 0.8555 / 0.9694) — never beats the
  best single model, the third consecutive confirmation. Runtime updated (~6.5 hours on one T4);
  README and CITATION now point at the fresh kernel. Results in this repo are from this run.

## [v7] — first complete cloud run — 2026-09-29

First time every stage of the rebuilt pipeline executed end-to-end, on Kaggle (T4), all stages clean.
Produced the headline set of that era (best EfficientNet-B0 95.88%) and three findings kept in this
project's record: OpenVINO/INT8 device behaviour (above), the benign NVIDIA-compiler log noise, and
T4 training speed (all five models in ~1h35m). Its full output (including the OpenVINO/INT8 reports)
is preserved outside this repo.

## [v2] — the clean rebuild — 2026-09-28/29

A from-scratch rewrite after a forensic audit of the original run found **23 defects**, including:
the "Azure GPU run" was actually CPU; scikit-learn was silently broken (roc_auc returned 0.5 — all metrics
were homemade approximations); results were appended across sessions producing duplicate rows; three
incompatible result schemas coexisted; ONNX export had crashed and the shipped `.onnx` came from a
different run; "INT8 quantization" was a no-op (the artifact *grew*); Grad-CAM used deprecated hooks on
the wrong model with unseeded samples; the benchmark timing compared GPU batch-8 vs CPU batch-4 and
wrote 0.0 as a measurement; albumentations 2.x silently ignored the intended noise parameters; the dataset
was cited to the wrong Kaggle page. The rebuild replaced all of it with the stage-gated design —
every stage writes its artifacts first, marks itself complete second, and every later stage reads only
those records. The full flaw→fix table lives in the notebook's introduction.

## [v1] — the original — 2025-11

The M.Tech course project: DeiT-Tiny at 99.31% on the (then) 2021 Version-1 dataset. Its headline number
was real code's output — but produced under the flawed conditions listed above, and on a test set whose
Version-1 composition (103 exact train/test copies; 87% near-duplicate contamination measured in this
project) inflated it. This project exists because that number deserved interrogation.

## The findings (cumulative, across versions)

1. **73% of this benchmark's test set shares a perceptual hash with its training split** (1,162/1,600 vs the
   actual training images; 78% vs the raw Training folder). The dataset's Version-2 update removed *exact*
   copies but cannot fix same-patient slices — so the 99% club keeps collecting a ~7-9 point leakage premium.
   Independently consistent with 2025 medical-imaging literature (see README → Related work).
2. **Dataset editions are different exams.** Version 1 (2021) and Version 2 (2026) of the same Kaggle
   dataset share ~98% of images but draw a different train/test boundary — accuracies from one are
   meaningless against the other (the original 99.31% and this repo's 96.31% were scored on different tests).
3. **The headline seed was the lucky seed**: 96.31% (seed 42) vs 95.75/95.69 (seeds 43/44). One number from
   one seed is a sample of one.
4. **Failure is structured**: errors concentrate in glioma→meningioma — the distinction clinicians also
   find hardest — and Grad-CAM shows the model attending to tumor regions, not shortcuts.
5. **Model choice is a speed/accuracy trade, not a ranking**: MobileNetV3-Small runs ~2,880 img/s on a T4 at
   1.5 points below EfficientNet-B0 — for a triage aid that trade is the deployment conversation.
