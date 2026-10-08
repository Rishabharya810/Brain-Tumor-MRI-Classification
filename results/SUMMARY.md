# Brain Tumor MRI Classification — v2 Clean Rebuild — Results Summary

Generated: 2026-10-08T14:44:25

## Environment
- Machine: Linux-6.18.48+-x86_64-with-glibc2.39 | training device: cuda
- torch 2.11.0+cu128 | timm 1.0.29 | sklearn 1.6.1 | numpy 2.1.3
- Dataset: Kaggle 'Brain Tumor MRI Dataset' by Masoud Nickparvar (masoudnickparvar/brain-tumor-mri-dataset). Public Kaggle version: 5,600 Training / 1,600 Testing (balanced 1,400/class train, 400/class test). Local v2 copy: 5,712 Training / 1,311 Testing (unbalanced). Combines Figshare (Cheng Jun) + SARTAJ + Br35H. The original notebook's download cell cited sartajbhuvaji/brain-tumor-classification-mri, which is a DIFFERENT dataset (2,870/3,919) - citation corrected in v2.

## Data-quality audit (leakage disclosure)
- exact_duplicates_within_train: 171
- exact_duplicates_within_test: 16
- exact_duplicates_train_test_overlap: 0
- near_duplicates_test_with_near_match_in_train (pHash Hamming<=8): 1248
- near_duplicate_pairs_test_x_train: 4740
- Slice-level MRI scans of the same patient commonly appear on both sides of the split; per-image metrics therefore overestimate patient-level generalisation. Reported honestly, not fixed by surgery on the dataset (that would change the benchmark).

## Model comparison (sealed test set, 1600 images)
```
             Model       Type  Params (M)  Size (MB)  Accuracy (%)  Macro-F1  Macro-P  Macro-R  Best epoch
        TinyViT-5M        ViT        5.07      20.29         96.12    0.9607   0.9630   0.9613          23
         DeiT-Tiny        ViT        5.53      22.10         95.38    0.9533   0.9567   0.9538          24
         ResNet-50        CNN       23.52      94.06         95.25    0.9519   0.9556   0.9525          27
       MobileViT-S Hybrid ViT        4.94      19.76         95.19    0.9513   0.9548   0.9519          26
Xception-Reference        CNN       20.82      83.26         95.19    0.9510   0.9555   0.9519          14
   EfficientNet-B0        CNN        4.01      16.05         95.19    0.9509   0.9555   0.9519          14
 MobileNetV3-Small        CNN        1.52       6.09         95.12    0.9507   0.9521   0.9512          19
      Baseline-CNN        CNN        0.09       0.38         65.81    0.6515   0.6549   0.6581          20
```

Reference run (Azure CPU session, Nov 2025): DeiT-Tiny 99.31%, TinyViT-5M 99.24%, EfficientNet-B0 98.93%, MobileViT-S 98.86%, MobileNetV3-Small 98.78%
Note: the reference run was scored on Version 1 of this dataset (2021 release — 5,712/1,311 images, 103 exact train/test copies, approximate fallback metrics); this run uses the 2026 Version 2 (5,600/1,600, exact overlap removed). The two scores are NOT directly comparable.

## ONNX export parity (max |Δlogit| vs PyTorch, 8-image fixed batch)
- MobileNetV3-Small: 3.00e-05 (PASS)
- TinyViT-5M: 4.77e-06 (PASS)

## Inference speed (median; warmup 10; 100 iters @ batch 1, 30 @ batch 16)
```
             model           runtime   device  batch  median_ms  ms_per_image    fps
      Baseline-CNN PyTorch CUDA fp32 Tesla T4      1   0.963673         0.964 1037.3
      Baseline-CNN PyTorch CUDA fp32 Tesla T4     16  10.075113         0.630 1587.3
         DeiT-Tiny PyTorch CUDA fp32 Tesla T4      1   6.103155         6.103  163.9
         DeiT-Tiny PyTorch CUDA fp32 Tesla T4     16  19.488419         1.218  821.0
   EfficientNet-B0 PyTorch CUDA fp32 Tesla T4      1   9.437719         9.438  106.0
   EfficientNet-B0 PyTorch CUDA fp32 Tesla T4     16  20.082109         1.255  796.8
 MobileNetV3-Small PyTorch CUDA fp32 Tesla T4      1   6.283516         6.284  159.1
 MobileNetV3-Small PyTorch CUDA fp32 Tesla T4     16   6.898368         0.431 2320.2
       MobileViT-S PyTorch CUDA fp32 Tesla T4      1  10.069205        10.069   99.3
       MobileViT-S PyTorch CUDA fp32 Tesla T4     16  40.002454         2.500  400.0
         ResNet-50 PyTorch CUDA fp32 Tesla T4      1   6.701590         6.702  149.2
         ResNet-50 PyTorch CUDA fp32 Tesla T4     16  56.720899         3.545  282.1
        TinyViT-5M PyTorch CUDA fp32 Tesla T4      1  10.074238        10.074   99.3
        TinyViT-5M PyTorch CUDA fp32 Tesla T4     16  31.625251         1.977  505.8
Xception-Reference PyTorch CUDA fp32 Tesla T4      1   8.570751         8.571  116.7
Xception-Reference PyTorch CUDA fp32 Tesla T4     16 182.835698        11.427   87.5
```

## Honest-split evaluation — clean test subset
Of the 1600 test images, 1162 share a perceptual hash (Hamming<=8) with a training image and were removed; the remaining 438 behave like unseen patients. The full-vs-clean gap is the leakage premium of this benchmark:
```
Baseline-CNN: full 65.81% -> clean 57.53% (drop 8.28 pp)
EfficientNet-B0: full 95.19% -> clean 86.76% (drop 8.43 pp)
MobileNetV3-Small: full 95.12% -> clean 86.76% (drop 8.37 pp)
MobileViT-S: full 95.19% -> clean 86.99% (drop 8.2 pp)
DeiT-Tiny: full 95.38% -> clean 86.53% (drop 8.85 pp)
TinyViT-5M: full 96.12% -> clean 88.81% (drop 7.31 pp)
ResNet-50: full 95.25% -> clean 87.21% (drop 8.04 pp)
Xception-Reference: full 95.19% -> clean 86.76% (drop 8.43 pp)
```

## Seed stability
- Best model (TinyViT-5M), 3 seeds: test accuracy 95.79% ± 0.28 pp | macro-F1 0.9572 ± 0.0030

## Notes
- Training device: cuda (CUDA GPU).
- Test metrics are per-image on a dataset containing duplicated and patient-correlated slices (see data-quality audit); the clean-subset and family-disjoint sections above are the honest estimates. Disclosed rather than hidden.

## Family-disjoint exam (leakage structurally impossible)
Families never cross this split (1440 test images). Every model was retrained from zero on it:
```
Baseline-CNN: official 65.81% -> disjoint 65.49%
EfficientNet-B0: official 95.19% -> disjoint 96.18%
MobileNetV3-Small: official 95.12% -> disjoint 95.14%
MobileViT-S: official 95.19% -> disjoint 96.46%
DeiT-Tiny: official 95.38% -> disjoint 94.44%
TinyViT-5M: official 96.12% -> disjoint 95.97%
ResNet-50: official 95.25% -> disjoint 94.51%
Xception-Reference: official 95.19% -> disjoint 97.22%
```

## Ensemble of the five lightweight architectures
- official 95.88% | clean subset 87.44% | family-disjoint 97.29% (macro-F1 0.9581 / 0.8720 / 0.9729)

## Clean-subset class composition
- glioma: 166, meningioma: 114, notumor: 40, pituitary: 118
- Research prototype only — not for clinical use. Single public dataset; no external or multi-institution validation.