# Dataset snapshot — the exact data this project used

This folder pins the **exact dataset** that produced every number in this repository (run of 2026-09-29). The upstream dataset is periodically re-released by its author — its Version 2 update (Feb 2026) changed the image counts *and redrew the train/test boundary* — so a future update would silently invalidate any comparison with this project. This card + the checksum manifest below make the snapshot verifiable forever.

## Identity

| Field | Value |
|---|---|
| Dataset | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Author | Masoud Nickparvar |
| Version used | **Version 2** ("Update 2026-02-13" — the second release; Version 1 was the 2021 initial release) |
| License | CC BY 4.0 |
| Downloaded | from the Kaggle dataset page / CLI, September 2026 |
| Size | 7,200 JPEG images, 167.8 MB |
| Structure | `Training/` 5,600 images (glioma/meningioma/notumor/pituitary = 1,400 each) · `Testing/` 1,600 images (400 each) |
| Underlying sources | [Figshare — Cheng et al. brain tumor dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) + [SARTAJ](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) + [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) |

## Verify the snapshot

- `file_hashes.csv` lists the **MD5 checksum, path and byte size of all 7,200 files** exactly as the pipeline consumed them (the notebook links this folder at `./Brain_tumor_MRI_Dataset`).
- A ZIP of this exact folder is attached to the repository's GitHub **Release**. To confirm a copy matches the run: hash the files and compare against `file_hashes.csv`, e.g. `md5sum -c` after stripping the folder prefix, or with any CSV/`pandas` comparison.

## Redistribution & attribution (CC BY 4.0)

The dataset is redistributed unmodified, as permitted by its license. Attribution: **"Brain Tumor MRI Dataset" by Masoud Nickparvar, CC BY 4.0** ([license text](https://creativecommons.org/licenses/by/4.0/)). The underlying source datasets belong to their respective authors (Cheng et al. / Figshare; SARTAJ; Br35H). No changes were made to any image or label.

## Known data-quality facts about this snapshot (from this project's audit)

- **Zero exact byte-identical copies between Training and Testing** (the Version 2 update removed the 103 exact copies present in Version 1).
- **171 exact duplicate files within Training**; 16 within Testing.
- **78% of Testing images (1,248/1,600) share a perceptual hash (pHash Hamming ≤ 8) with an image in the raw Training folder; 73% (1,162/1,600) with the actual 4,480-image training split** — same-patient slices on both sides. This is the project's central finding; the honest leak-free re-scoring is in the README's *dual metrics* table.

> **Comparability rule:** results produced on Version 1 (2021, 5,712/1,311 images) are *not* comparable with results on this snapshot. See the README's data section for the version history.
