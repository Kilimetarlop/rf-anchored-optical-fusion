# Cross-Session Drone Identification by Radio-Frequency-Anchored Optical Feature Fusion

Core implementation of **RAFF** (radio-frequency-anchored feature fusion), the method in the manuscript of the same title.
Section, equation, figure and algorithm numbers in the code refer to that manuscript.

## Problem

Five drones are enrolled in one recording session and identified in another. The set is closed, and two of the drones are the same
model (DJI Air 2S). Two radio-frequency (RF) receivers and four cameras, two visible and two infrared, observe every flight.

- The RF fingerprint of a drone persists from session to session, but it confuses drones whose transmitters belong to one family.
- The camera appearance separates the drones within a session, but a model trained in one session fails in the other: distance,
  viewing geometry, light and camera gain redraw every target.

## Method

1. **Modality tokens.** Each record is cut into 64 windows. Per window, each RF receiver gives a 13-dimensional blind OFDM
   fingerprint, and one receiver also gives the oscillator error ε. Each camera gives a 512-dimensional appearance token and a
   37-dimensional trajectory token.
2. **Token fusion network.**
   - One perceptron per token source, plus a source-type embedding.
   - A class token and two Transformer layers.
   - Masking tokens gives three readouts of one network: the RF view, the optical view and the fused (full) view.
3. **Enrolment training.** The network is trained on the labelled enrolment session through its RF view only. Its loss adds a
   window cross-entropy, a record-bag loss and a session-adversarial loss on unlabelled deployment windows. Its RF view then gives
   every deployment record an *RF anchor* score.
4. **In-session naming.** Five rounds, each with three steps:
   - **Balanced names.** Sinkhorn balancing turns the record scores into soft names, the same number of records per drone.
   - **Cross-fitting.** The deployment records are split into five folds. Each fold network restarts from the enrolment network
     and learns the names of the other folds. On half of the windows it sees only the optical tokens, so the names must be
     learned through the cameras.
   - **Renaming.** The held-out RF score plus the held-out optical score of every record gives the next names.
5. **Inference.** The last round's fold networks score every window on all tokens. The record decision is the identity with the
   highest mean window log-posterior.
6. **Counterfactual target audit.** Two tests of whether the optical branch reads the drone or its time slot:
   - a background-only run, scored by the retained gain G_bg;
   - a target transplant, scored by the target-follow rate F_tgt.

Deployment labels are never used for training or naming; they only score the results.

## Results

Record accuracy on DroneMMset, two directions between sessions D00 and D01, mean of 10 seeds:

| Method | D00 → D01 | D01 → D00 |
|---|---|---|
| RF only (the RF anchor) | 0.828 | 0.755 |
| Optical only (trained on the enrolment labels) | 0.199 | 0.191 |
| CBST, the best published method in our comparison, adapted to record decisions | 0.917 | 0.885 |
| **RAFF** | **0.988** | **0.957** |

## Files

| File | Content | Manuscript |
|---|---|---|
| [tokens_rf.py](tokens_rf.py) | Blind OFDM fingerprint per segment and receiver; oscillator error ε from the arrival times of repeated reference symbols | Sec. IV-B |
| [tokens_optical.py](tokens_optical.py) | Appearance token and trajectory token per camera and window | Sec. IV-B |
| [data.py](data.py) | Session split; token loading, standardization with enrolment statistics, availability, sensor dropout | Eq. (8), Sec. IV-C |
| [model.py](model.py) | Token fusion network | Sec. IV-C, Eqs. (8)–(9) |
| [enrol.py](enrol.py) | Enrolment training on the RF view | Sec. IV-D, Eqs. (10)–(12) |
| [naming.py](naming.py) | RF anchor, balanced naming, cross-fitting, RF-token masking, renaming, inference | Sec. IV-E, Eqs. (13)–(16), Algorithm 1 |
| [audit.py](audit.py) | Background-only and target-transplant crops, token swap after naming, G_bg and F_tgt | Sec. IV-F, Eqs. (17)–(18) |
| [run.py](run.py) | One cross-session run of Algorithm 1 | Algorithm 1 |

## Data

The experiments use DroneMMset [1], sessions D00 (about 50 m) and D01 (about 200 m, from day into night), recorded hours apart on
the same day. Each session holds 75 records: 5 drones × 5 actions × 3 rounds, with action A0110 excluded. This repository contains
no data. `run.py` expects the following inputs.

**windows.csv**, one row per window, with columns:
- `tuple` (record identifier);
- `D` (session), `A` (action), `S` (round);
- `T` (drone: `T0001`, `T0010`, `T0011`, `T0100`, `T0101`; T0001 and T0010 are both Air 2S);
- `k` (window index);
- the 0/1 columns `Dev02`, `Dev01`, `Cam01`, `Cam02`, `Inf01`, `Inf02`, telling whether each sensor recorded the window.

Every record has the same number of windows. `A` and `S` are needed only by `audit.donors`, and `Dev01_fc_mhz` (band centre of the
second receiver) only by `tokens_rf.dev01_carrier`.

**Token files** (`.npz`), rows aligned with `windows.csv`. Every array comes with a boolean `<key>_ok` that tells whether the token
could be extracted:

| Keys | Shape | Built with |
|---|---|---|
| `fp02`, `fp01` | (n, 13) | `tokens_rf.segment_fingerprints`, then `tokens_rf.window_tokens` |
| `eps` | (n,) | `tokens_rf.record_zeta`, then `tokens_rf.session_eps`, broadcast to the windows of each record |
| `vs01`, `vs02`, `vi01`, `vi02` | (n, 512) | `tokens_optical.appearance_token` |
| `tc01`, `tc02`, `ti01`, `ti02` | (n, 37) | `tokens_optical.trajectory_token` |

A token is available in a window when its sensor recorded the window and the token was extracted. Not part of this release:
- the detection, tracking and cropping of the targets that produce the 64 × 64 crops and the tracks;
- the carrier estimate of the first receiver (Dev02).

## Usage

```bash
pip install -r requirements.txt
python run.py --windows data/windows.csv \
  --tokens data/rf_tokens.npz,data/appearance_tokens.npz,data/trajectory_tokens.npz \
  --train D00 --test D01 --seed 0 --out d00_d01_s0.json
```

The run prints the RF anchor and, for every naming round, the accuracy of the fused, RF and optical views and of the names. The JSON
holds these readouts and the final fused accuracy (`fused`). The device is chosen automatically: MUSA (`torch_musa`), then CUDA,
then CPU.

**Audit.**
- *Target transplant.* Add, for example, `--audit bg=data/appearance_bg.npz,tp0=data/appearance_tp0.npz,tp1=data/appearance_tp1.npz,tp2=data/appearance_tp2.npz`.
  - Each file holds appearance tokens computed with `audit.appearance_token` on crops processed by `audit.background_only` or
    `audit.transplant`, with donors from `audit.donors`.
  - The files replace the deployment tokens after naming. `tp1_vis_F_tgt` in the JSON is the target-follow rate of Eq. (18).
- *Retained gain* (Eq. (17)). It needs a second set of runs whose appearance tokens come from background-only crops in both
  sessions. Feed the mean fused and RF-anchor accuracies of both sets of runs to `audit.retained_gain`.
- The audit in the manuscript uses the appearance tokens without the trajectory tokens:
  `--names fp02,fp01,eps,vs01,vs02,vi01,vi02`.

## Settings

- **Training dropout.** Every sensor is dropped with probability 0.2, and all cameras together with probability 0.5, but a
  window always keeps at least one token.
- **Network.** 1.74 M parameters: token perceptrons d → 128 → 256, then two pre-norm Transformer layers (4 heads, width 256,
  feed-forward 512), then layer norm → dropout 0.3 → linear map to the 5 drones.
- **Enrolment.**
  - 60 epochs (1140 steps); every batch holds 16 records × 16 windows.
  - AdamW (learning rate 1e-3, weight decay 0.05), one-cycle schedule with 5 % warm-up, gradient clipping at 1.0, label
    smoothing 0.1.
  - The gradient-reversal weight rises as 2/(1 + e^(−10η)) − 1 over the training progress η.
- **Naming.**
  - 5 rounds × 5 folds; each fold network trains 380 steps at a learning rate of 3e-4 with cosine annealing.
  - Every step uses 256 enrolment windows (RF view, true labels) and 256 deployment windows of the other folds (soft names, RF
    tokens masked with probability 0.5 when a camera recorded the window).
  - 50 Sinkhorn iterations per balancing.

## Relation to the experiment code

The numbers in the manuscript were produced with the full experiment code, which also holds the baselines, ablations and controls.
This repository keeps the code path of the final configuration. It was checked against the experiment code:
- **Tokens.** The token functions reproduce the token files used in the manuscript. The optical tokens and ε are bit-identical; the
  RF fingerprints differ only by float32 rounding.
- **Training and naming.** For the same inputs and seeds, the following are identical: inputs and availability (including the
  training-time dropout), initial weights, forward outputs, balanced names, and fold-routed scores.

The reported runs have not been repeated with this condensed code. The experiment code draws additional random numbers, and the
naming stage is not bit-reproducible on the accelerator used, so a rerun will differ from the reported runs seed by seed.

## Requirements

Python 3.10 or later, PyTorch, NumPy and pandas; `tokens_rf.py` also needs SciPy.
- The experiments ran with Python 3.10 and PyTorch 2.11 (`torch_musa`) on an MTT S4000 GPU.
- The token functions were checked with Python 3.12, NumPy 1.26, SciPy 1.14, pandas 3.0 and PyTorch 2.7.

## Citation

Citation details will be added when the manuscript is published. Please also cite the dataset:

```bibtex
@misc{zhang2025dronemmset,
  author       = {Zhang, Junning and Ya, Zhuwei and Lan, Song and Zhang, Dongli and Fu, Xingpeng and Xia, Xuecheng and Chen, Shengzhi and Duan, Yongsheng},
  title        = {{DroneMMset}: A multi-modal perception dataset for low-altitude {UAV} monitoring},
  howpublished = {ModelScope},
  year         = {2025},
  url          = {https://modelscope.cn/datasets/DroneMMset/DroneMMset}
}
```

[1] J. Zhang et al., "DroneMMset: A multi-modal perception dataset for low-altitude UAV monitoring," ModelScope, 2025.
https://modelscope.cn/datasets/DroneMMset/DroneMMset
