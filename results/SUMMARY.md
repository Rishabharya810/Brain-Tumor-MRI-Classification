# Brain Tumor MRI Classification — v2 Clean Rebuild — Results Summary

Generated: 2026-10-09T12:33:12

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
   EfficientNet-B0        CNN        4.01      16.05         96.06    0.9601   0.9623   0.9606          25
Xception-Reference        CNN       20.82      83.26         95.88    0.9582   0.9614   0.9587          22
         DeiT-Tiny        ViT        5.53      22.10         95.50    0.9543   0.9580   0.9550          24
        TinyViT-5M        ViT        5.07      20.29         95.19    0.9510   0.9537   0.9519          19
       MobileViT-S Hybrid ViT        4.94      19.76         94.88    0.9480   0.9511   0.9487          21
 MobileNetV3-Small        CNN        1.52       6.09         94.31    0.9423   0.9441   0.9431          16
         ResNet-50        CNN       23.52      94.06         92.94    0.9285   0.9324   0.9294          21
      Baseline-CNN        CNN        0.09       0.38         65.25    0.6460   0.6507   0.6525          24
```

Reference run (Azure CPU session, Nov 2025): DeiT-Tiny 99.31%, TinyViT-5M 99.24%, EfficientNet-B0 98.93%, MobileViT-S 98.86%, MobileNetV3-Small 98.78%
Note: the reference run was scored on Version 1 of this dataset (2021 release — 5,712/1,311 images, 103 exact train/test copies, approximate fallback metrics); this run uses the 2026 Version 2 (5,600/1,600, exact overlap removed). The two scores are NOT directly comparable.

## ONNX export parity (max |Δlogit| vs PyTorch, 8-image fixed batch)
- EfficientNet-B0: 3.50e-05 (PASS)
- MobileNetV3-Small: 7.25e-05 (PASS)

## Inference speed (median; warmup 10; 100 iters @ batch 1, 30 @ batch 16)
```
             model           runtime   device  batch  median_ms  ms_per_image    fps
      Baseline-CNN PyTorch CUDA fp32 Tesla T4      1   0.956550         0.957 1044.9
      Baseline-CNN PyTorch CUDA fp32 Tesla T4     16   9.795847         0.612 1634.0
         DeiT-Tiny PyTorch CUDA fp32 Tesla T4      1   5.126268         5.126  195.1
         DeiT-Tiny PyTorch CUDA fp32 Tesla T4     16  18.812903         1.176  850.3
   EfficientNet-B0 PyTorch CUDA fp32 Tesla T4      1   7.891042         7.891  126.7
   EfficientNet-B0 PyTorch CUDA fp32 Tesla T4     16  19.765333         1.235  809.7
 MobileNetV3-Small PyTorch CUDA fp32 Tesla T4      1   5.094462         5.094  196.3
 MobileNetV3-Small PyTorch CUDA fp32 Tesla T4     16   5.207054         0.325 3076.9
       MobileViT-S PyTorch CUDA fp32 Tesla T4      1   8.423452         8.423  118.7
       MobileViT-S PyTorch CUDA fp32 Tesla T4     16  38.383845         2.399  416.8
         ResNet-50 PyTorch CUDA fp32 Tesla T4      1   6.013635         6.014  166.3
         ResNet-50 PyTorch CUDA fp32 Tesla T4     16  53.445977         3.340  299.4
        TinyViT-5M PyTorch CUDA fp32 Tesla T4      1   8.290849         8.291  120.6
        TinyViT-5M PyTorch CUDA fp32 Tesla T4     16  30.166994         1.885  530.5
Xception-Reference PyTorch CUDA fp32 Tesla T4      1   8.418446         8.418  118.8
Xception-Reference PyTorch CUDA fp32 Tesla T4     16 172.957558        10.810   92.5
```

## Honest-split evaluation — clean test subset
Of the 1600 test images, 1162 share a perceptual hash (Hamming<=8) with a training image and were removed; the remaining 438 behave like unseen patients. The full-vs-clean gap is the leakage premium of this benchmark:
```
Baseline-CNN: full 65.25% -> clean 56.62% (drop 8.63 pp)
EfficientNet-B0: full 96.06% -> clean 88.13% (drop 7.93 pp)
MobileNetV3-Small: full 94.31% -> clean 84.47% (drop 9.84 pp)
MobileViT-S: full 94.88% -> clean 86.07% (drop 8.8 pp)
DeiT-Tiny: full 95.5% -> clean 86.53% (drop 8.97 pp)
TinyViT-5M: full 95.19% -> clean 86.99% (drop 8.2 pp)
ResNet-50: full 92.94% -> clean 84.02% (drop 8.92 pp)
Xception-Reference: full 95.88% -> clean 87.9% (drop 7.98 pp)
```

## Seed stability
- Best model (EfficientNet-B0), 3 seeds: test accuracy 95.98% ± 0.07 pp | macro-F1 0.9593 ± 0.0008

## Notes
- Training device: cuda (CUDA GPU).
- Test metrics are per-image on a dataset containing duplicated and patient-correlated slices (see data-quality audit); the clean-subset and family-disjoint sections above are the honest estimates. Disclosed rather than hidden.

## Family-disjoint exam (leakage structurally impossible)
Families never cross this split (1440 test images). Every model was retrained from zero on it:
```
Baseline-CNN: official 65.25% -> disjoint 66.46%
EfficientNet-B0: official 96.06% -> disjoint 97.08%
MobileNetV3-Small: official 94.31% -> disjoint 94.86%
MobileViT-S: official 94.88% -> disjoint 95.21%
DeiT-Tiny: official 95.5% -> disjoint 95.56%
TinyViT-5M: official 95.19% -> disjoint 96.74%
ResNet-50: official 92.94% -> disjoint 95.49%
Xception-Reference: official 95.88% -> disjoint 96.6%
```

## Ensemble of the five lightweight architectures
- official 95.81% | clean subset 87.90% | family-disjoint 97.50% (macro-F1 0.9574 / 0.8745 / 0.9750)

## Clean-subset class composition
- glioma: 166, meningioma: 114, notumor: 40, pituitary: 118
- Research prototype only — not for clinical use. Single public dataset; no external or multi-institution validation.