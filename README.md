# Depth From Image

Monocular depth estimation from a single RGB image — a learning project.

A U-Net style encoder–decoder (pretrained ResNet18 encoder + skip-connected
upsampling decoder) trained on the NYU Depth V2 labeled subset (1,449 indoor
RGB/depth pairs), using a scale-invariant log loss.

## Contents

- `monocular_depth_unet.ipynb` — the full pipeline as a Colab notebook:
  1. Environment setup, dataset loading (with the fsspec timeout workaround) and exploration
  2. `Dataset`/`DataLoader` with resizing, normalization and depth-safe augmentation
  3. ResNet18 encoder + U-Net decoder, with a discussion of freezing vs. fine-tuning
  4. Scale-invariant log loss (SILog) + optional gradient-matching term
  5. Training loop with Google Drive checkpointing, periodic visualization and per-epoch metrics
  6. Inference on your own uploaded photos, including out-of-distribution outdoor scenes
  7. Summary of strengths, failure modes and next steps

## Running it

Open the notebook in Google Colab, set the runtime to **GPU (T4)**, and run the
cells top to bottom. The first cell pins `datasets==3.6.0` (script-based dataset
loading was removed in `datasets` 4.x) and asks for a runtime restart.

Training is ~8–15 minutes for 20 epochs at 192×256 on a free-tier T4.
Checkpoints are written to `MyDrive/depth_from_image/checkpoints/` so a
disconnect does not lose progress — re-running the training cell resumes from
`last.pt`.

Design choices are flagged inline as 🟦 (chosen to stay Colab-friendly) or
🟩 (how it is genuinely done in practice).
