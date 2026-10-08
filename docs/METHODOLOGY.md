# METHODOLOGY — the zero-background reader's layer

This document explains every concept and formula the notebook uses, in plain English, in the order you meet them. Nothing here requires prior machine-learning knowledge; every section ends with a pointer to the notebook stage that implements it. For what we found, read the [README](../README.md); for how the machinery works, keep reading.

---

## 1 · The problem, in plain English

A brain MRI machine takes **many thin cross-sectional pictures ("slices")** of one patient's brain. A radiologist looking at one slice wants to know which of four things it shows: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. We train a computer model to look at the image's raw pixel numbers and output a probability for each of the four.

A model like this would only ever be a **triage aid** — a fast first pass to help prioritize scans — never a diagnosis. (Notebook Stages 0–6.)

## 2 · The three-way split — textbook, practice quizzes, final exam

| Set | Role | How often used |
|---|---|---|
| **Training** (~4,480 images) | the textbook — the model *learns* from these | every epoch |
| **Validation** (~1,120) | practice quizzes — steer decisions: when to stop, which epoch's weights were best | checked after every epoch, never learned from |
| **Test** (1,600) | the sealed final exam | opened **once** per model |

Why validation must exist: training decisions need a measuring stick, and the test set must stay virgin — using it to steer training is a smaller version of the very leak this project is about. Our split is **seeded and built from sorted file lists**, so it is identical on any machine, forever. *(Stage 3)*

## 3 · Preprocessing, augmentation, and normalization

Every image passes one shared function:

1. **Black-margin removal** — MRI scans sit inside a black frame; we detect the bright region (the brain) and crop to it.
2. **Resize** — to $224 \times 224$ pixels ($299 \times 299$ for the Xception reference model, its native size).
3. **Normalization** — pixels (originally $0$–$255$) become small, centered numbers:

$$x_{\text{norm}} = \frac{x/255 - \mu_c}{\sigma_c} \quad \text{per color channel } c \in \{R, G, B\}$$

with the standard **ImageNet statistics** $\mu = (0.485,\ 0.456,\ 0.406)$, $\sigma = (0.229,\ 0.224,\ 0.225)$ — a universal convention. The result: values mostly between $-2$ and $+2$, so every input influences the model on a comparable scale, which makes learning stable. Train, validation, and test are normalized **identically** — one shared function.

**Augmentation** (training images only): random rotations, flips, small shifts, brightness/contrast changes, mild noise and blur. It multiplies variety for free and teaches "a tumor is a tumor at any angle." Validation and test get **zero** augmentation — they must look like real future data. *(Stage 3)*

## 4 · What "training" physically is

A model is millions of adjustable knobs (**weights**). Training repeats five steps:

1. **Guess** — feed a batch of 16 images through; the weights produce 16 predictions.
2. **Score — the loss.** Our loss function is **cross-entropy**. For one image with true class $y$ and predicted probabilities $p_1 \ldots p_4$:

$$\mathcal{L} = -\log p_y$$

Its personality: a humble near-miss costs a little; a **confidently wrong** answer costs enormously — exactly the teacher we want.
3. **Assign blame — backpropagation.** Walk the network's math backwards and compute, for every weight, how much it contributed and which direction reduces the loss (its **gradient**).
4. **Nudge — the optimizer (AdamW).** Turn every weight slightly downhill.
5. **Repeat** — one pass over all training data = one **epoch** (max 30 for us).

**Learning rate** = the size of the nudge. Too big and the model thrashes; too small and it crawls. *(Stage 5)*

## 5 · The optimizer's safety systems

**AdamW** improves the plain nudge with two behaviors and one discipline:

- **Momentum** — each weight remembers its recent direction and keeps rolling that way, smoothing out noisy batches.
- **Per-weight adaptive step sizes** — weights with wild gradients get careful small steps; weights with small consistent gradients get bigger ones.
- **Weight decay** — every weight is pulled a tiny bit toward zero each step, keeping the model humble (huge weights = brittle overconfidence).

Two safety systems around it:

- **Cosine learning-rate schedule** — the rate starts at $10^{-4}$ and glides down a cosine curve to $\approx 0$ by epoch 30:

$$\eta_t = \tfrac{1}{2}\,\eta_{\max}\left(1 + \cos\left(\tfrac{t}{T}\,\pi\right)\right)$$

Bold steps early while far from a solution, tiny steps late while polishing.
- **Gradient clipping** (cap = 1.0) — one weird batch can scream "turn all knobs 50 units!"; clipping rescales any oversized update to the cap. A speed limiter.

**Early stopping:** after each epoch the validation **macro-F1** (see §7) is measured; if it hasn't improved for 7 straight epochs, training stops. The best-so-far weights are saved **atomically** (never half-written), and interrupted training **resumes** mid-epoch. *(Stage 5)*

## 6 · The models — three eras of design, one shared recipe

- **CNN (Convolutional Neural Network):** learned stencils slide across the image detecting patterns — edges in early layers, textures next, object parts deep. Efficient and fast. (Baseline-CNN, EfficientNet-B0, MobileNetV3-Small, ResNet-50, Xception-Reference.)
- **Vision Transformer (ViT):** the image is cut into 196 patches; in **attention** blocks every patch looks at every other patch — global context in one step, where a CNN needs many layers. (DeiT-Tiny, TinyViT-5M.)
- **Hybrid:** convolution early (local detail), attention later (global context). (MobileViT-S.)

All five lightweight architectures start from **ImageNet-initialized weights** (standard transfer learning — a head start from a model that learned on everyday photos) and fine-tune under one identical recipe; the 90k-parameter **BaselineCNN trains truly from scratch** as the no-pretraining floor. **Xception-Reference** reproduces the famous "99% accuracy" recipe itself. So the comparison measures *architecture and recipe*, not training budget. *(Stages 5–6, Config in Stage 0)*

## 7 · Reading the scorecard: precision, recall, F1, ROC

For one class (say glioma):

- **Precision** $= \dfrac{TP}{TP + FP}$ — *when the model said glioma, how often was it right?* (quality of alarms)
- **Recall** $= \dfrac{TP}{TP + FN}$ — *of all true gliomas, how many did it catch?* (coverage of disease)
- **F1** $= 2 \cdot \dfrac{P \cdot R}{P + R}$ — their balance (a lopsided score is punished).

Loosening the decision threshold raises recall but lowers precision (the metal-detector knob). **Macro** averaging = the plain average across the four classes — the smallest class counts as much as the biggest, the only ethical averaging in medicine. The **confusion matrix** is the 4×4 truth-vs-prediction grid; the **ROC curve** sweeps every threshold and plots catch-rate vs false-alarm-rate, with **AUC** (1.0 = perfect separation, 0.5 = coin flip) measuring separation independent of any chosen threshold. *(Stage 7)*

## 8 · Grad-CAM — looking inside the model's head

Grad-CAM is an **after-the-fact X-ray**, not part of prediction. For an already-made prediction it measures how much each image region contributed (via gradients pointed at the last feature maps) and paints it: red = "this is why I decided." It verifies the model looks at *tumors*, not shortcuts (skull shape, brightness), and on wrong answers shows whether the mistake was looking at something tumor-like (understandable) or noise (broken). We explain 4 correct + 4 misclassified seeded samples per model, plus the **failure profile** (which true→predicted confusions dominate). *(Stage 8)*

## 9 · The audit: fingerprints for images

- **MD5 hash** — a fingerprint of a file's exact **bytes**; same fingerprint = byte-identical files (exact duplicates).
- **Perceptual hash (pHash)** — a 64-bit fingerprint of what an image *looks like*: the image is shrunk, transformed to frequencies (DCT), and each bit records whether a low-frequency coefficient is above or below the median. Visually similar images get similar fingerprints.
- **Hamming distance** — the number of differing bits between two fingerprints. Ours: distance $\le 8$ (of 64) = near-duplicate — almost certainly the **same patient**.

This is how the audit finds that **73% of the official test images have a near-twin in the training split** — MD5 alone would never show it (near-twins have different bytes). For the family-disjoint split, pairs additionally require **deep-feature cosine similarity $\ge 0.95$** (two independent pieces of evidence), and **families** are built by chaining those links — then whole families, never single images, are assigned to sides. *(Stages 2 and 13)*

## 10 · ONNX export and the parity check

**ONNX** is a universal model format — "PDF for models" — so the trained network can run outside PyTorch (faster runtimes, other languages, edge devices). Exporting can silently break a model, so the same 8 test images are pushed through both engines and the raw outputs (logits) must match:

$$\max_i \left| z_i^{\text{PyTorch}} - z_i^{\text{ONNX}} \right| < 10^{-3}$$

Measured: $\le 1.2 \times 10^{-4}$ — PASS for both exported models. *(Stage 9)*

## 11 · Timing honestly

Speed uses: **warmup 10 runs** (first calls pay one-time costs — measure steady state), **100 timed runs at batch 1** (one doctor, one scan — latency) and **30 at batch 16** (a server chewing a queue — throughput), reporting the **median** (immune to one-off OS hiccups), with **GPU synchronization inside the timer** (GPU work is asynchronous; without syncing you time the request, not the work). A number is only written if it was actually measured — 0.0 is never a placeholder. *(Stage 10)*

## 12 · Reproducibility: seeds and error bars

Computers fake randomness with a generator that has a starting point — the **seed** (42). Same seed + same data + same code = the same result, every time. But one seed's score is still **one lottery draw** (different seeds give different scores), so the best model is retrained at seeds 43 and 44 and reported as **mean ± standard deviation** — the honest error bar. Run-to-run GPU nondeterminism moves scores by ~±0.3 pp; conclusions here are stable across two independent full runs. *(Stages 0 and 12)*

## 13 · The three exams — why one accuracy number is never enough

- **Official benchmark:** the shipped test set. Comparable with the literature — but 73% of it shares patients with training.
- **Clean remnant:** the 438 test images with no near-twin in training. Strict — and *pessimistic*, because the survivors are systematically harder than average (selection bias: the contaminated images were covering the hard cases).
- **Family-disjoint exam:** re-draw the whole split moving families intact, retrain every model from zero, score on 1,440 images where leakage is structurally impossible. The **fair** estimate.

The gap between tiers is the finding — see the [README results](../README.md#results--one-project-three-exams). *(Stages 11, 13, 14)*

## 14 · Glossary

| Term | Meaning |
|---|---|
| Epoch / batch | one full pass over the training data / the chunk of 16 images processed together |
| Loss (cross-entropy) | one number measuring how wrong — and how confidently wrong — the guesses were |
| Gradient | for one weight: the direction (and steepness) that reduces the loss |
| Overfitting | memorizing the textbook instead of learning the subject |
| Parameter | one learnable number inside the model (e.g., 5.1M in TinyViT-5M) |
| Transfer learning / ImageNet-initialized | starting from weights already taught on everyday photos |
| Leakage | any way information about the exam leaks into studying — here, same-patient slices on both sides |
| Family | a group of near-duplicate images (chained with dual evidence) that must never be split across train/test |
| pp | percentage points |
| Idempotent / resumable / atomic | run twice = run once / crashed runs continue / no half-written files |

## 15 · Hyperparameter cheat-sheet (all from the notebook's Config)

| Setting | Value |
|---|---|
| Image size | 224 (299 for Xception-Reference) |
| Batch size | 16 |
| Max epochs | 30 (early stop patience 7) |
| Optimizer | AdamW, lr $10^{-4}$, weight decay $10^{-4}$ |
| LR schedule | cosine to $\approx 0$ over 30 epochs |
| Gradient clip | 1.0 |
| Seed | 42 (43, 44 for stability) |
| Class balancing | WeightedRandomSampler (sampling only) |
| pHash rule | Hamming $\le 8$; family links also require cosine $\ge 0.95$ |
| ONNX | opset 17, dynamic batch axis, parity $< 10^{-3}$ |
