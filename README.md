# Glaucoma Screening — Implementation

Open `glaucoma_screening.ipynb` in Google Colab (File → Upload notebook, or upload it to Google Drive and open with Colaboratory) and run top to bottom.

## Before you start
1. **Register for REFUGE/REFUGE2** at https://refuge.grand-challenge.org/REFUGE2Download/ if you haven't — this can take a day or two to approve, so do it first.
2. Get a Kaggle API token: kaggle.com → Settings → API → **Create New Token** (gives you a `KGAT_...` token). You'll paste it into a hidden-input prompt when the notebook's Section 4.1 cell asks for it — nothing gets written into the notebook file itself, so it's safe to keep committing this notebook to git.

## Where to download each dataset

| Dataset | Link | Access |
|---|---|---|
| G1020 | https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets | Kaggle account (free) — the notebook downloads this automatically once your API token is pasted in |
| ACRIMA | https://figshare.com/s/c2d31f850af14c5b5232 | Open, no account needed — download the zip from the page and upload it to Drive (see Section 4.3 note below) |
| RIM-ONE DL | https://github.com/miag-ull/rim-one-dl (official source also at http://medimrg.webs.ull.es/research/retinal-imaging/rim-one/) | Open — the notebook clones this automatically |
| REFUGE | https://refuge.grand-challenge.org/REFUGE2Download/ | Registration required |
| REFUGE2 | https://refuge.grand-challenge.org/REFUGE2Download/ (same page, second edition) | Registration required |

ACRIMA and REFUGE/REFUGE2 all need a manual download-then-upload-to-Drive step since they're not available via a scriptable API — the notebook's Section 4.4 cell unzips whatever you drop in `MyDrive/glaucoma_screening/data/raw/`.

## What each section does
| Section | Purpose |
|---|---|
| 0–3 | GPU check, installs, Drive mount, config (matches Table 3.3) |
| 4 | Downloads G1020, ACRIMA, RIM-ONE DL automatically; REFUGE/REFUGE2 need manual upload to Drive after registration |
| 5 | Builds one unified manifest CSV across all datasets — **you'll need to fill in the G1020/REFUGE parsers once you see the actual folder layout** (the inspection cell prints it for you) |
| 6 | Disc/cup-aware preprocessing (mask-based crop where available, brightness-based fallback otherwise) |
| 7 | 70/15/15 stratified split + augmented `tf.data` pipeline |
| 8 | Model: EfficientNetB0 + CBAM attention + classification head |
| 9–10 | Class weighting, two-phase training (frozen warm-up, then fine-tune) |
| 11 | Full metric suite: accuracy, precision, recall, specificity, F1, MCC, Cohen's Kappa, ROC-AUC, PR-AUC, log loss |
| 12 | External validation on REFUGE2 (template — same pattern as training manifest) |
| 13 | Grad-CAM visualisation |
| 14 | TFLite conversion + size/inference benchmarking |
| 15 | Saves everything to Google Drive |

## The parts that need your judgment call once you see real data
- **Section 5 (manifest):** RIM-ONE DL and ACRIMA parsers are written and should work as-is; G1020 and REFUGE/REFUGE2 are left as templates because their exact CSV/folder layout can only be confirmed after download. Run the inspection cell first, then fill in the commented block for each.
- **Section 12:** same — build the REFUGE2 manifest the same way once its label CSV is confirmed.

## Switching backbones
Set `CONFIG['backbone']` to `'MobileNetV3Small'` instead of `'EfficientNetB0'` to compare the two, as your methodology proposes testing both.
