# FlowMatch

> **FlowMatch: Augmentation-Flow Semi-Supervised Adaptation of Vision Foundation Models**


FlowMatch is a **semi-supervised adaptation** framework for Vision Foundation
Models (VFMs) such as CLIP and DINOv2. It freezes the VFM backbone, trains
only lightweight adaptation modules and classification heads, and uses an
**ordered augmentation flow** to (i) measure pseudo-label reliability,
(ii) decouple pseudo-label selection from sample filtering, and (iii)
stabilize teacher updates with a drift-aware EMA (**LyapEMA**).

<p align="center">
  <img src="docs/FlowMatch.png" width="92%" alt="FlowMatch overview"/>
</p>

---

## 1.  What's in this repository

```text
FlowMatch/
├── train.py                         # entry point
├── eval.py                          # evaluation script
├── requirements.txt
├── config/
│   ├── usb_cv/flowmatch/            # VTAB-style configs (DTD, SUN397, RESISC45,
│   │                                #   CLEVR-C, Retinopathy)
│   └── classic_cv/flowmatch/        # CIFAR-10 / STL-10 configs (Appendix)
└── semilearn/
    ├── algorithms/
    │   ├── flowmatch/               # ★ our method
    │   ├── fixmatch/                # baseline (used by FlowMatch's flow-only ablation)
    │   ├── hooks/   utils/
    │   └── __init__.py              # exposes only fixmatch + flowmatch
    ├── core/                        # USB training framework (sampler, scheduler, …)
    ├── nets/                        # CLIP ViT, DINOv2 ViT, ViT, ResNet, WRN
    └── datasets/
        └── cv_datasets/             # data interfaces for DTD/SUN397/RESISC45/
                                     #   CLEVR-C/Retinopathy/CIFAR/STL-10
```

The repository ships **code only** — no checkpoints and no raw datasets.
All datasets are loaded through their respective HuggingFace / torchvision
interfaces and cached under `--data_dir`.

---

## 2.  Method at a glance

For each unlabeled sample `u_b`, FlowMatch produces an **augmentation flow** of
`K` views from the weakest to the strongest augmentation:

```
v_b^(0)  ----A(s_0)----  v_b^(1)  ---- … ----  v_b^(K-1)
weakest                                            strongest
```

A frozen teacher predicts a class-probability vector at every flow point.
FlowMatch then combines three ingredients:

1. **Decoupled augmentation-flow pseudo-labeling.**
   The pseudo-label is taken from the *most-confident* flow point
   `k* = argmax_k r_b^(k)`, while the reliability gate is computed from the
   *weakest* flow point `r_b^(0) ≥ τ`. This keeps sample selection
   conservative but lets the pseudo-label come from the most informative view.

2. **Flow-level prediction consistency** `L_flow`.
   The student is forced to predict smoothly along the augmentation flow:
   `L_flow = (1 / μB(K-1)) Σ_b Σ_k ||o_b^(k+1) - o_b^(k)||²`.
   This acts as an implicit Lipschitz regularizer on the prediction path.

3. **LyapEMA** drift-aware teacher updates.
   The EMA coefficient is increased when the student drifts far from the
   teacher, slowing the teacher down when student updates may be driven by
   incorrect pseudo-supervision.

The full objective is `L = L_sup + λ_u L_unsup + λ_f L_flow`.

---

## 3.  Installation

```bash
git clone https://github.com/HYDxGenius/FlowMatch.git
cd FlowMatch
python -m venv .env && source .env/bin/activate     # or conda
pip install -r requirements.txt
```

Tested with Python 3.10 / PyTorch ≥ 1.12 / transformers ≥ 4.30.
A CUDA GPU is required for the foundation-model experiments.

---

## 4.  Datasets

FlowMatch accesses every dataset through HuggingFace `datasets` or
`torchvision`. The first run downloads to `--data_dir` (default `./data`).

| Dataset       | Source                                          | Classes | Label budgets used in paper |
|---------------|-------------------------------------------------|--------:|-----------------------------|
| DTD           | `torchvision.datasets.DTD`                      |      47 | 141, 282                    |
| SUN397        | `tanganke/sun397` (HF)                          |     397 | 1191, 2382                  |
| RESISC45      | `timm/resisc45` (HF)                            |      45 | 45, 90                      |
| Retinopathy   | `kaggle/diabetic-retinopathy-detection` (HF)    |       5 | 40, 80                      |
| CLEVR-Count   | `clip-benchmark/wds_vtab-clevr_count_all` (HF)  |       8 | 10, 20                      |
| CIFAR-10      | `torchvision.datasets.CIFAR10`                  |      10 | 40                          |
| STL-10        | `torchvision.datasets.STL10`                    |      10 | 40                          |

> **Note.** The `clip-benchmark/wds_vtab-clevr_count_all` dataset stores its
> labels under `cls` (already 0-indexed). Use `int(sample['cls'])` — do **not**
> subtract 3.

---

## 5.  Quick start

VTAB-style adaptation of a CLIP ViT-B/16 backbone on DTD with 141 labels:

```bash
python train.py \
    --c config/usb_cv/flowmatch/flowmatch_dtd_141_clip_0.yaml \
    --data_dir ./data \
    --save_dir ./saved_models
```

Same setup with a DINOv2 ViT-B/14 backbone:

```bash
python train.py \
    --c config/usb_cv/flowmatch/flowmatch_dtd_141_dino_0.yaml \
    --data_dir ./data
```

Conventional SSL (Appendix) on CIFAR-10 with 40 labels:

```bash
python train.py \
    --c config/classic_cv/flowmatch/flowmatch_cifar10_40_0.yaml
```

Evaluation:

```bash
python eval.py --load_path ./saved_models/<run_name>/model_best.pth
```

---

## 6.  Key hyperparameters

The defaults below match the values used in Table I of the paper and can
be overridden from the config file or the command line.

| Name              | Default | Description                                         |
|-------------------|--------:|-----------------------------------------------------|
| `traj_steps` (`K`)|       5 | augmentation-flow length                            |
| `lambda_f`        |     0.1 | weight of `L_flow`                                  |
| `lambda_u`        |     1.0 | weight of `L_unsup`                                 |
| `p_cutoff` (`τ`)  |    0.95 | reliability threshold on the weakest flow point     |
| `lyap_alpha0`     |   0.999 | LyapEMA base EMA coefficient                        |
| `lyap_gamma`      |    0.02 | LyapEMA sensitivity to student–teacher drift        |
| `lyap_alpha_min/max` | 0.99 / 0.9999 | clipped EMA range                                |

---

## 7.  Reproducing Table I

Each row of Table I corresponds to one config under `config/usb_cv/flowmatch/`.
For example, FlowMatch + CLIP on RESISC45 with 90 labels:

```bash
python train.py --c config/usb_cv/flowmatch/flowmatch_resisc45_90_clip_0.yaml
```

The `_clip_` / `_dino_` suffix selects the backbone; `_<N>_` selects the label
budget.

---

## 8.  Diagnostics (AFI / AWPR)

The paper introduces two diagnostic metrics for pseudo-label reliability:

- **Augmentation-Flow Instability (AFI)**:
  `AFI(u_b) = (1/(K-1)) Σ_k ||p_b^(k+1) - p_b^(k)||²`
- **Accepted Wrong Pseudo-Label Rate (AWPR)**:
  `AWPR = Σ m_b · 1[ỹ_b ≠ y_b*] / Σ m_b`

Both are logged automatically when wandb is enabled (see `flowmatch.py`,
`diag/afi_*` and `diag/awpr_mask_accepted`).

---


---

## 9.  Acknowledgements

FlowMatch is built on top of the
[USB](https://github.com/microsoft/Semi-supervised-learning) semi-supervised
learning framework. We thank the USB authors for providing a flexible base
codebase. Backbone weights are loaded from HuggingFace
(`openai/clip-vit-base-patch16`, `facebook/dinov2-base`).
