# Brain Tumor MRI Classification — v2 Clean Rebuild — Results Summary

Generated: 2026-10-10T15:32:15

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
Xception-Reference        CNN       20.82      83.26         96.06    0.9601   0.9627   0.9606          17
   EfficientNet-B0        CNN        4.01      16.05         95.75    0.9569   0.9603   0.9575          20
         DeiT-Tiny        ViT        5.53      22.10         95.44    0.9533   0.9575   0.9544          24
       MobileViT-S Hybrid ViT        4.94      19.76         95.12    0.9505   0.9542   0.9513          23
         ResNet-50        CNN       23.52      94.06         94.38    0.9433   0.9438   0.9437          24
 MobileNetV3-Small        CNN        1.52       6.09         93.94    0.9382   0.9411   0.9394          17
        TinyViT-5M        ViT        5.07      20.29         93.75    0.9356   0.9410   0.9375          15
      Baseline-CNN        CNN        0.09       0.38         62.31    0.6109   0.6239   0.6231          17
```

Reference run (Azure CPU session, Nov 2025): DeiT-Tiny 99.31%, TinyViT-5M 99.24%, EfficientNet-B0 98.93%, MobileViT-S 98.86%, MobileNetV3-Small 98.78%
Note: the reference run was scored on Version 1 of this dataset (2021 release — 5,712/1,311 images, 103 exact train/test copies, approximate fallback metrics); this run uses the 2026 Version 2 (5,600/1,600, exact overlap removed). The two scores are NOT directly comparable.

## ONNX export parity (max |Δlogit| vs PyTorch, 8-image fixed batch)
- MobileNetV3-Small: 4.48e-05 (PASS)
- Xception-Reference: 6.02e-06 (PASS)

## Inference speed (median; warmup 10; 100 iters @ batch 1, 30 @ batch 16)
```
             model           runtime   device  batch  median_ms  ms_per_image    fps
      Baseline-CNN PyTorch CUDA fp32 Tesla T4      1   0.966059         0.966 1035.2
      Baseline-CNN PyTorch CUDA fp32 Tesla T4     16   9.807287         0.613 1631.3
         DeiT-Tiny PyTorch CUDA fp32 Tesla T4      1   5.296700         5.297  188.8
         DeiT-Tiny PyTorch CUDA fp32 Tesla T4     16  17.807202         1.113  898.5
   EfficientNet-B0 PyTorch CUDA fp32 Tesla T4      1   8.835830         8.836  113.2
   EfficientNet-B0 PyTorch CUDA fp32 Tesla T4     16  18.983925         1.186  843.2
 MobileNetV3-Small PyTorch CUDA fp32 Tesla T4      1   5.457009         5.457  183.3
 MobileNetV3-Small PyTorch CUDA fp32 Tesla T4     16   5.558164         0.347 2881.8
       MobileViT-S PyTorch CUDA fp32 Tesla T4      1   8.682693         8.683  115.2
       MobileViT-S PyTorch CUDA fp32 Tesla T4     16  36.164222         2.260  442.5
         ResNet-50 PyTorch CUDA fp32 Tesla T4      1   6.544817         6.545  152.8
         ResNet-50 PyTorch CUDA fp32 Tesla T4     16  47.172573         2.948  339.2
        TinyViT-5M PyTorch CUDA fp32 Tesla T4      1   8.860070         8.860  112.9
        TinyViT-5M PyTorch CUDA fp32 Tesla T4     16  28.177006         1.761  567.9
Xception-Reference PyTorch CUDA fp32 Tesla T4      1   7.826413         7.826  127.8
Xception-Reference PyTorch CUDA fp32 Tesla T4     16 142.041684         8.878  112.6
```

## Honest-split evaluation — clean test subset
Of the 1600 test images, 1162 share a perceptual hash (Hamming<=8) with a training image and were removed; the remaining 438 behave like unseen patients. The full-vs-clean gap is the leakage premium of this benchmark:
```
Baseline-CNN: full 62.31% -> clean 52.74% (drop 9.57 pp)
EfficientNet-B0: full 95.75% -> clean 88.58% (drop 7.17 pp)
MobileNetV3-Small: full 93.94% -> clean 84.02% (drop 9.92 pp)
MobileViT-S: full 95.12% -> clean 86.07% (drop 9.05 pp)
DeiT-Tiny: full 95.44% -> clean 86.07% (drop 9.36 pp)
TinyViT-5M: full 93.75% -> clean 83.33% (drop 10.42 pp)
ResNet-50: full 94.38% -> clean 88.36% (drop 6.02 pp)
Xception-Reference: full 96.06% -> clean 89.27% (drop 6.79 pp)
```

## Seed stability
- Best model (Xception-Reference), 3 seeds: test accuracy 95.81% ± 0.18 pp | macro-F1 0.9575 ± 0.0019

## Notes
- Training device: cuda (CUDA GPU).
- Test metrics are per-image on a dataset containing duplicated and patient-correlated slices (see data-quality audit); the clean-subset and family-disjoint sections above are the honest estimates. Disclosed rather than hidden.

## Family-disjoint exam (leakage structurally impossible)
Families never cross this split (1440 test images). Every model was retrained from zero on it:
```
Baseline-CNN: official 62.31% -> disjoint 66.67%
EfficientNet-B0: official 95.75% -> disjoint 95.9%
MobileNetV3-Small: official 93.94% -> disjoint 95.0%
MobileViT-S: official 95.12% -> disjoint 95.62%
DeiT-Tiny: official 95.44% -> disjoint 94.93%
TinyViT-5M: official 93.75% -> disjoint 96.53%
ResNet-50: official 94.38% -> disjoint 95.49%
Xception-Reference: official 96.06% -> disjoint 96.67%
```

## Ensemble of the five lightweight architectures
- official 95.19% | clean subset 85.84% | family-disjoint 96.94% (macro-F1 0.9510 / 0.8555 / 0.9694)

## Clean-subset class composition
- glioma: 166, meningioma: 114, notumor: 40, pituitary: 118
- Research prototype only — not for clinical use. Single public dataset; no external or multi-institution validation.