# SepDreamer

**SepDreamer** is a model-based offline reinforcement learning framework for sepsis treatment optimisation in the ICU. It extends [MedDreamer](https://github.com/RoyalSkye/MedDreamer) — a DreamerV3-based architecture for clinical decision support — with three original contributions designed to address the specific challenges of real-world EHR data: irregular time sampling, sparse mortality-driven reward, and unstable offline policy learning.

> **MSc Thesis** · University of Groningen / Queen Mary University of London · 2026  
> Thesis link: _coming soon_

---

## Overview

Standard Dreamer-style world models assume fixed-interval observations and dense reward signals — assumptions that break down on ICU data. SepDreamer addresses this with:

1. **AFI Encoder** — an Adaptive Feature Integration module that encodes per-feature time deltas alongside observations, enabling the RSSM to reason over irregularly sampled clinical measurements without imputation.
2. **MedR Reward** — a mortality-calibrated reward model that replaces noisy clinical reward with a learned outcome predictor, trained on terminal survival/death signals from MIMIC-IV.
3. **Real/Imagined Loss Separation** — a weighted actor loss that down-weights imagined rollouts (`imag_loss_weight: 0.1`) relative to real transitions, preventing policy collapse caused by compounding world model errors in offline settings.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SepDreamer                           │
│                                                             │
│  o_t (EHR obs)  ──►  AFI Encoder  ──►  x̃_t               │
│  Δ_t (time deltas)                   (feature-interaction  │
│  m_t (missingness mask)               embedding)           │
│                             │                               │
│                             ▼                               │
│                   ┌─────────────────┐                       │
│                   │   RSSM (World   │  ──►  Decoder        │
│                   │    Model)       │  ──►  MedR Head      │
│                   └────────┬────────┘  ──►  Cont Head      │
│                            │                                │
│                   Imagination rollouts                      │
│                            │                                │
│                   ┌────────▼────────┐                       │
│                   │  Actor / Critic │  real:  w = 1.0      │
│                   │   (P1 / P2)     │  imag:  w = 0.1      │
│                   └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### AFI Encoder

Each observation $o_t \in \mathbb{R}^D$ is encoded jointly with the per-feature time-since-last-measurement vector $\Delta_t \in \mathbb{R}^D$:

$$\tilde{o}_t = \sigma\!\left(\mathrm{Linear}(o_t) + \mathrm{FM}([W_o \cdot o_t \mid W_\Delta \cdot \Delta_t])\right)$$

where FM is a Factorization Machine over the joint embedding. This gives the world model an explicit signal about measurement staleness without requiring manual imputation — a critical design choice for ICU data where clinical variables arrive at heterogeneous frequencies.

### MedR Reward

Rather than using the original SOFA/lactate clinical reward (which produced a flat reward surface and degenerate policies), SepDreamer replaces it with a **potential-based reward shaping** function grounded in physiological homeostasis:

$$r_t = \lambda \cdot (\gamma \cdot \phi(s_{t+1}) - \phi(s_t)) - c(a_t)$$

The potential $\phi(s_t) \in [0, 1]$ evaluates the current patient state across 6 clinical features (SOFA, base excess, lactate, urine output, MAP, heart rate). Each feature contributes a **survival score** — computed via bell curves or decay functions centred on clinical normal ranges — weighted by a **confidence term** $e^{-\Delta t / \tau}$ that down-weights stale measurements using the same per-feature time deltas as the AFI encoder. The action cost $c(a_t)$ penalises high vasopressor and IV fluid doses.

Hyperparameters (half-life, confidence decay $\tau$, sigma scale, action cost scale, survival/confidence weights) were selected via **Pareto-optimal search** over three objectives evaluated on the training cohort:

- $J_\text{surv}$: correlation between cumulative reward and patient survival
- $J_\text{conf}$: negative correlation between reward and measurement uncertainty  
- $J_\text{comp}$: correlation between reward and clinical efficiency (homeostasis gain per unit drug dose)

The selected candidate (`medr_cand_0004`) achieves the minimum utopia distance across all three objectives.

---

## Dataset

SepDreamer trains on the **MIMIC-IV** sepsis cohort extracted via BigQuery. Treatment decisions are discretised into **25 actions** (5 vasopressor levels × 5 IV fluid levels), following the standard sepsis RL benchmark setup.

> **Data access**: MIMIC-IV requires credentialed PhysioNet access. Raw data is not included in this repository. See `src/data_extraction/` for the SQL extraction pipeline.

---

## Installation

```bash
# Python 3.13+ required
git clone https://github.com/<your-username>/SepDreamer.git
cd SepDreamer

# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

---

## Usage

### 1 — Train the World Model

```bash
python src/meddreamer/main.py --config defaults
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./logs_wm
```

### 2 — Evaluate the World Model

```bash
python src/meddreamer/main.py --config eval-wm --checkpoint <path_to_checkpoint>
```

Evaluation logs (reconstruction MSE per feature, reward calibration, action sensitivity) are written to `logs_eval_wm/`.

### 3 — Train the Policy

```bash
# Phase 1 — hybrid real + imagined transitions
python src/meddreamer/main.py --config p1-sepsis --wm_checkpoint <path>

# Phase 2 — imagination-only
python src/meddreamer/main.py --config p2-sepsis --wm_checkpoint <path>
```

---

## Key Results (World Model at Step 5000)



---

## Project Structure

```
SepDreamer/
├── src/
│   ├── meddreamer/
│   │   ├── main.py              # Entry point
│   │   ├── dreamer.py           # Dreamer + MedDreamer agent classes
│   │   ├── configs.yaml         # All training configurations
│   │   ├── models/
│   │   │   ├── afi.py           # AFI encoder (original contribution)
│   │   │   ├── models.py        # RSSM, reward head, continuation head
│   │   │   └── networks.py      # MLP, GRU building blocks
│   │   ├── utils/
│   │   └── analysis/
│   ├── data_extraction/         # MIMIC-IV SQL queries + BigQuery pipeline
│   └── preprocessing/           # Cohort selection, feature engineering, reward shaping
├── data/
│   └── meddreamer_dataset/      # Processed episodes (not tracked by git)
├── reward_analysis/             # Reward distribution diagnostics
└── logs_*/                      # Training and evaluation logs (not tracked by git)
```

---

## Acknowledgements

- [MedDreamer](https://github.com/RoyalSkye/MedDreamer) — the base framework this work extends
- [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) — PyTorch DreamerV3 implementation
- [danijar/dreamerv3](https://github.com/danijar/dreamerv3) — original DreamerV3 (JAX)
- MIMIC-IV — Johnson et al., PhysioNet
