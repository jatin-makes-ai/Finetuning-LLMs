# Finetuning-LLMs: Enterprise Three-Stage Domain Adaptation Pipeline (CPT + SFT + DPO)

A production-grade implementation of domain-adaptive **Continued Pre-Training (CPT)**, **Supervised Fine-Tuning (SFT)**, and **Direct Preference Optimization (DPO)** using Parameter-Efficient Fine-Tuning (PEFT / LoRA) on NVIDIA GPUs.

Specialized for the Indian legal domain across **`Qwen/Qwen2.5-0.5B`** and enterprise **`Qwen/Qwen2.5-7B`**, featuring a standardized 6-part legal opinion schema and deterministic, machine-parseable constitutional badges (`[Article X]`).

---

## Architecture Overview

Enterprise domain specialization requires a three-stage training pipeline:

```text
                     [General Pre-Trained Base: Qwen2.5-7B]
                                       │
                                       ▼
    ┌──────────────────────────────────┐
    │  STAGE 1: Continued Pre-Train    │  <-- Causal LM over 5,061,498 Indian legal tokens:
    │             (CPT)                │      Legislative gazettes, Supreme Court judgments,
    └──────────────────┬───────────────┘      and academic treatises (Perplexity: 3.68)
                       │
                       ▼
         [Domain-Adapted CPT Model]
                       │
                       ▼
    ┌──────────────────────────────────┐
    │  STAGE 2: Instruction & Schema   │  <-- SFT with Prompt-Loss Masking (100 scenarios):
    │             (SFT)                │      Enforces 6-part opinion schema & 100% regex
    └──────────────────┬───────────────┘      deterministic `[Article X]` extraction
                       │
                       ▼
         [Interactive Legal Assistant]
                       │
                       ▼
    ┌──────────────────────────────────┐
    │  STAGE 3: Preference Alignment   │  <-- DPO with Beta=0.1 & Schema Parity (60 pairs):
    │             (DPO)                │      Penalizing hallucinations, bypassing remedies,
    └──────────────────┬───────────────┘      and false Article 12/311 attribution
                       │
                       ▼
    [Enterprise Aligned Legal Intelligence] <-- Production Checkpoint (qwen-7b-legal-dpo-adapter)
```

---

## Technical Specifications & Hardware

| Parameter | 0.5B Prototype Pipeline | 7B Enterprise Pipeline |
| :--- | :--- | :--- |
| **Base Architecture** | `Qwen/Qwen2.5-0.5B` (24 Layers, 896 Dim) | `Qwen/Qwen2.5-7B` (28 Layers, 3584 Dim) |
| **PEFT Precision** | 16-bit LoRA ($r=16, \alpha=32$) | 8-bit NormalFloat LoRA ($r=16, \alpha=32$) |
| **Target Projections** | All 7 (`q, k, v, o, gate, up, down`) | All 7 (`q, k, v, o, gate, up, down`) |
| **Trainable Params** | 8.80M (1.75% of total) | 40.24M (0.53% of total) |
| **CPT Volume** | ~50k tokens (Synthetic) | **5,061,498 domain tokens** (Corpus) |
| **SFT Volume** | 25 consultation dialogues | **100 cases (6-part schema + regex badges)** |
| **DPO Volume** | 20 contrastive pairs | **60 adversarial pairs (Schema Parity)** |
| **Hardware** | NVIDIA RTX 3090 (24 GB VRAM) | NVIDIA RTX 3090 (24 GB VRAM) |

---

## Complete Multi-Stage Scorecard (Qwen2.5-7B)

| Pipeline Stage | Training Objective | Execution Time (RTX 3090) | Loss / Margin Trajectory | Core Breakthrough |
|---|---|---|---|---|
| **Stage 0: Base 7B** | Foundation pre-training | — | — | Baseline: Hallucinated non-existent statutes, failed schema badges. |
| **Stage 1: CPT 7B** | Causal LM over 5.06M domain tokens | **1h 32m 13s** (309 steps) | Loss: $1.5097 \to 1.3020$ | Grounded vocabulary; domain perplexity dropped to **`3.68`**. |
| **Stage 2: SFT 7B** | Joint prompt-masked instruction & schema | **6m 30s** (39 steps) | Loss: $1.7626 \to 0.2002$ | **100% Deterministic Regex Badge Extraction** (`[Article X]`). |
| **Stage 3: DPO 7B** | Bradley-Terry preference alignment ($\beta=0.1$) | **4m 46s** (16 steps) | Margin: $0.0 \to +6.5127$, Acc: $100\%$ | **Refused Adversarial Traps**; upgraded criminal protections to Art 20(3). |

---

## 4-Way Comparative Benchmark (Unseen Adversarial Cases)

Tested across 3 strictly out-of-sample legal dilemmas that never appeared in CPT, SFT, or DPO:

```
BENCH-01: Enforcement Directorate Summons, Passcode Compulsion & Section 19 PMLA Safeguards
BENCH-02: Indefinite Mobile Internet Shutdown & 4-Prong Proportionality Test (Anuradha Bhasin)
BENCH-03: Adversarial Trap: Article 32 Supreme Court Writ Against a Private Hospital
```

### Empirical Results

| Metric | Base Qwen2.5-7B | CPT 7B Adapter | SFT 7B Adapter | DPO 7B Adapter |
| :--- | :---: | :---: | :---: | :---: |
| **Schema Completeness** | 6.0 / 6 (Unstructured) | 5.0 / 6 (Inconsistent) | 5.67 / 6 (Standardized) | **6.0 / 6 (Flawless)** |
| **Regex Article Extraction Rate** | 0% (0 / 9) | 0% (0 / 9) | **100% (10 / 9)** | **100% (9 / 9)** |
| **BENCH-01: Passcode Protection** | Green Emoji Table | Prose, No Badges | Generic Art 19(1)(g) | **Pinpoint Art 20(3) Shield** |
| **BENCH-02: Internet Shutdown** | Hallucinated `#Article21` | Conflated Assembly | 5/6 Schema | **6/6 Schema + Art 226 Writ** |
| **BENCH-03: Private Hospital Trap** | **FAILED**: Urged Art 32 | **FAILED**: Urged Art 32 | Followed Art 14 framing | **PASSED**: Refused Art 32 |

---

## Repository Structure

```text
├── data/
│   ├── benchmark_7b_all_stages.json          # 4-way comparative evaluation output across all 12 test runs
│   ├── legal_sft_7b_dataset.jsonl            # 100 6-part legal opinions with regex badges (ChatML format)
│   ├── legal_dpo_7b_dataset.jsonl            # 60 adversarial pairs enforcing Schema Parity (beta=0.1)
│   ├── legal_sft_dataset.jsonl               # 0.5B prototype SFT dataset (25 dialogs)
│   ├── legal_dpo_dataset.jsonl               # 0.5B prototype DPO dataset (20 pairs)
│   ├── stage2_comparative_eval.json          # 0.5B 3-way evaluation results
│   └── dpo_comparative_eval.json             # 0.5B SFT vs DPO evaluation results
├── docs/
│   ├── stage4_7b_finetuning_and_benchmarks.md # Comprehensive 7B pipeline documentation & benchmark analysis
│   ├── finetuning-foundations.md              # Fine-tuning taxonomy and PEFT deep-dive (LoRA, QLoRA, DoRA)
│   ├── baseline_base_model_eval.md            # Raw pre-trained base model benchmark
│   ├── two_stage_legal_finetuning_pipeline.md # Domain adaptation engineering blueprint
│   ├── stage1_cpt_results.md                  # 0.5B Stage 1 CPT results
│   ├── cpt_dataset_and_hyperparameters_guide.md # Token packing and dataset architecture guide
│   ├── stage2_sft_results.md                  # 0.5B Stage 2 SFT methodology
│   └── stage3_dpo_results.md                  # 0.5B Stage 3 DPO mathematical formulation
├── scripts/
│   ├── train_legal_cpt_7b.py                  # 7B CPT training script (5.06M tokens, 8-bit LoRA)
│   ├── generate_legal_sft_7b_dataset.py       # 7B SFT dataset generator with 6-part schema & regex badges
│   ├── train_legal_sft_7b.py                  # 7B SFT training with prompt-loss masking & warm-starting
│   ├── generate_legal_dpo_7b_dataset.py       # 7B DPO dataset generator with Schema Parity
│   ├── train_legal_dpo_7b.py                  # 7B DPO training script with trl.DPOTrainer
│   ├── eval_7b_all_stages.py                  # Automated 4-way comparative benchmark script
│   ├── train_legal_cpt.py                     # 0.5B CPT training script
│   ├── generate_legal_sft_dataset.py          # 0.5B SFT dataset generator
│   ├── train_legal_sft.py                     # 0.5B SFT training script
│   ├── generate_legal_dpo_dataset.py          # 0.5B DPO dataset generator
│   ├── train_legal_dpo.py                     # 0.5B DPO training script
│   ├── eval_all_stages.py                     # 0.5B 3-way benchmark evaluation
│   └── eval_dpo.py                            # 0.5B DPO evaluation
└── README.md
```

---

## Reproducing the 7B Pipeline

### 1. Stage 1: Continued Pre-Training (CPT)
```bash
python scripts/train_legal_cpt_7b.py
```

### 2. Stage 2: Supervised Fine-Tuning (SFT) with Structured Schema
```bash
python scripts/generate_legal_sft_7b_dataset.py
python scripts/train_legal_sft_7b.py
```

### 3. Stage 3: Direct Preference Optimization (DPO)
```bash
python scripts/generate_legal_dpo_7b_dataset.py
python scripts/train_legal_dpo_7b.py
```

### 4. 4-Way Comparative Benchmark
```bash
python scripts/eval_7b_all_stages.py
```
