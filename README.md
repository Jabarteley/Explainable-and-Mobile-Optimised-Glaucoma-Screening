# Glaucoma Screening — Implementation (Kaggle Notebooks branch)

This branch runs the whole pipeline on **Kaggle Notebooks** instead of Google Colab — no API token or `kaggle.json` needed for datasets already on Kaggle, just "Add Data" in the UI. The `main` branch has the Colab version of this same notebook.

Open `glaucoma_screening.ipynb` on kaggle.com → Code → New Notebook → File → Import Notebook, and run top to bottom.

## Before you start
1. **Register for REFUGE/REFUGE2** at https://refuge.grand-challenge.org/REFUGE2Download/ — this can take a day or two to approve, so do it first.
2. In the notebook's right-hand panel: **Settings → Accelerator → GPU T4 x2**, and **Internet → On**.

## Where to get each dataset, and how it gets into the notebook

| Dataset | Link | How it gets attached |
|---|---|---|
| G1020 | https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets | Click **Add Data** in the notebook → search `glaucoma-datasets` by `arnavjain1` → Add. Mounts at `/kaggle/input/glaucoma-datasets/` automatically. |
| RIM-ONE DL | https://github.com/miag-ull/rim-one-dl | Cloned automatically by the notebook (Section 4.1) — no manual step. |
| ACRIMA | https://figshare.com/s/c2d31f850af14c5b5232 | Download the zip from figshare, then on kaggle.com go to **Create → New Dataset** and upload it there (private, just for you). Then **Add Data** in the notebook and attach your new dataset. |
| REFUGE | https://refuge.grand-challenge.org/REFUGE2Download/ | Same pattern as ACRIMA: download after registration, upload as your own private Kaggle Dataset, then Add Data. |
| REFUGE2 | Same page as REFUGE (second edition) | Same pattern as ACRIMA. |

After attaching each one, fill in its real folder name in the `DATASET_PATHS` dict in Section 4 (the notebook prints what's currently attached so you can copy the exact path).

## What each section does
| Section | Purpose |
|---|---|
| 0–2 | GPU/Internet check, installs, `/kaggle/input` vs `/kaggle/working` path setup, config (matches Table 3.3) |
| 4 | Dataset acquisition — RIM-ONE DL auto-cloned; G1020/ACRIMA/REFUGE/REFUGE2 attached via Add Data |
| 5 | Builds one unified manifest CSV across all datasets — **you'll need to fill in the G1020/REFUGE parsers once you see the actual folder layout** (the inspection cell in Section 4.2 prints it for you) |
| 6 | Disc/cup-aware preprocessing (mask-based crop where available, brightness-based fallback otherwise) |
| 7 | 70/15/15 stratified split + augmented `tf.data` pipeline |
| 8 | Model: EfficientNetB0 + CBAM attention + classification head |
| 9–10 | Class weighting, two-phase training (frozen warm-up, then fine-tune) |
| 11 | Full metric suite: accuracy, precision, recall, specificity, F1, MCC, Cohen's Kappa, ROC-AUC, PR-AUC, log loss |
| 12 | External validation on REFUGE2 (template — same pattern as the training manifest) |
| 13 | Grad-CAM visualisation |
| 14 | TFLite conversion + size/inference benchmarking |
| 15 | Saves everything to `/kaggle/working` (persists when you click **Save Version**) |

## The parts that need your judgment call once you see real data
- **Section 5 (manifest):** RIM-ONE DL and ACRIMA parsers are written and should work as-is; G1020 and REFUGE/REFUGE2 are left as templates because their exact CSV/folder layout can only be confirmed after you attach them. Run the inspection cell (Section 4.2) first, then fill in the commented block for each.
- **Section 12:** same — build the REFUGE2 manifest the same way once its label CSV is confirmed.

## Switching backbones
Set `CONFIG['backbone']` to `'MobileNetV3Small'` instead of `'EfficientNetB0'` to compare the two, as your methodology proposes testing both.

## A note on Kaggle's free GPU quota
The free tier gives ~30 GPU-hours/week. Turn the accelerator off (Settings → Accelerator → None) whenever you're just editing cells or writing parsers, and only switch it on when you're actually about to train.
