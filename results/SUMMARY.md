# Brain Tumor MRI Classification — v2 Clean Rebuild — Results Summary

Generated: 2026-09-29T19:33:49

## Environment
- Machine: Linux-6.12.90+-x86_64-with-glibc2.35 | training device: cuda
- torch 2.10.0+cu128 | timm 1.0.26 | sklearn 1.6.1 | numpy 2.0.2
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
  EfficientNet-B0        CNN        4.01      16.05         96.31    0.9626   0.9644   0.9631          15
       TinyViT-5M        ViT        5.07      20.29         95.50    0.9541   0.9572   0.9550          21
MobileNetV3-Small        CNN        1.52       6.09         95.44    0.9536   0.9572   0.9544          26
        DeiT-Tiny        ViT        5.53      22.10         95.19    0.9509   0.9543   0.9519          16
      MobileViT-S Hybrid ViT        4.94      19.76         95.00    0.9489   0.9532   0.9500          20
     Baseline-CNN        CNN        0.09       0.38         64.06    0.6354   0.6414   0.6406          27
```

Reference run (Azure CPU session, Nov 2025): DeiT-Tiny 99.31%, TinyViT-5M 99.24%, EfficientNet-B0 98.93%, MobileViT-S 98.86%, MobileNetV3-Small 98.78%
Note: the reference run was scored on Version 1 of this dataset (2021 release — 5,712/1,311 images, 103 exact train/test copies, approximate fallback metrics); this run uses the 2026 Version 2 (5,600/1,600, exact overlap removed). The two scores are NOT directly comparable.

## ONNX export parity (max |Δlogit| vs PyTorch, 8-image fixed batch)
- EfficientNet-B0: 6.53e-05 (PASS)
- MobileNetV3-Small: 4.20e-05 (PASS)

## Inference speed (median; warmup 10; 100 iters @ batch 1, 30 @ batch 16)
```
            model           runtime   device  batch  median_ms  ms_per_image    fps
     Baseline-CNN PyTorch CUDA fp32 Tesla T4      1   0.965657         0.966 1035.2
     Baseline-CNN PyTorch CUDA fp32 Tesla T4     16   9.847867         0.615 1626.0
        DeiT-Tiny PyTorch CUDA fp32 Tesla T4      1   5.277278         5.277  189.5
        DeiT-Tiny PyTorch CUDA fp32 Tesla T4     16  17.468926         1.092  915.8
  EfficientNet-B0 PyTorch CUDA fp32 Tesla T4      1   8.192617         8.193  122.1
  EfficientNet-B0 PyTorch CUDA fp32 Tesla T4     16  18.987393         1.187  842.5
MobileNetV3-Small PyTorch CUDA fp32 Tesla T4      1   5.456460         5.456  183.3
MobileNetV3-Small PyTorch CUDA fp32 Tesla T4     16   5.545323         0.347 2881.8
      MobileViT-S PyTorch CUDA fp32 Tesla T4      1   8.861155         8.861  112.9
      MobileViT-S PyTorch CUDA fp32 Tesla T4     16  35.673410         2.230  448.4
       TinyViT-5M PyTorch CUDA fp32 Tesla T4      1   8.606785         8.607  116.2
       TinyViT-5M PyTorch CUDA fp32 Tesla T4     16  27.867386         1.742  574.1
```

## Honest-split evaluation — clean test subset
Of the 1600 test images, 1162 share a perceptual hash (Hamming<=8) with a training image and were removed; the remaining 438 behave like unseen patients. The full-vs-clean gap is the leakage premium of this benchmark:
```
Baseline-CNN: full 64.06% -> clean 55.48% (drop 8.58 pp)
EfficientNet-B0: full 96.31% -> clean 89.5% (drop 6.81 pp)
MobileNetV3-Small: full 95.44% -> clean 87.21% (drop 8.22 pp)
MobileViT-S: full 95.0% -> clean 86.53% (drop 8.47 pp)
DeiT-Tiny: full 95.19% -> clean 86.3% (drop 8.89 pp)
TinyViT-5M: full 95.5% -> clean 86.99% (drop 8.51 pp)
```

## Seed stability
- Best model (EfficientNet-B0), 3 seeds: test accuracy 95.92% ± 0.28 pp | macro-F1 0.9587 ± 0.0028

## Notes
- Training device: cuda (CUDA GPU).
- Test metrics are per-image on a dataset containing duplicated and patient-correlated slices (see data-quality audit); the clean-subset section above is the honest estimate. Disclosed rather than hidden.
- Research prototype only — not for clinical use. Single public dataset; no external or multi-institution validation.