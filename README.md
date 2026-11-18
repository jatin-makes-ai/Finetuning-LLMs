# Finetuning-LLMs: Enterprise Three-Stage Domain Adaptation Pipeline (CPT + SFT + DPO)

A production-grade implementation of domain-adaptive **Continued Pre-Training (CPT)**, **Supervised Fine-Tuning (SFT)**, and **Direct Preference Optimization (DPO)** using Parameter-Efficient Fine-Tuning (PEFT / LoRA) on NVIDIA GPUs.

---

## Architecture Overview

Enterprise domain specialization requires a three-stage training pipeline:

```text
[General Pre-Trained Base: Qwen2.5-0.5B]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ STAGE 1: Continued Pre-Train │  <-- Causal Language Modeling over raw domain text:
   │            (CPT)             │      Statutory frameworks, constitutional articles,
   └──────────────┬───────────────┘      precedents, and legal doctrines (No padding)
                  │
                  ▼
      [Domain-Adapted Base Model]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ STAGE 2: Instruction Tuning  │  <-- Supervised Fine-Tuning with ChatML formatting:
   │            (SFT)             │      Prompt-loss masking, legal consultation QA,
   └──────────────┬───────────────┘      structured statutory analysis
                  │
                  ▼
      [Interactive Legal Assistant]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ STAGE 3: Preference Align    │  <-- Direct Preference Optimization (DPO):
   │            (DPO)             │      Penalizing hallucinations, foreign legal transplants,
   └──────────────┬───────────────┘      and illegal self-help advice (beta=0.1)
                  │
                  ▼
   [Aligned Legal AI Production Model] <-- 943 MB standalone safetensors checkpoint
```

---

## Technical Specifications & Hardware

- **Base Architecture**: `Qwen/Qwen2.5-0.5B` (RoPE, GQA, SwiGLU, 24 Layers, 896 Hidden Dimension)
- **PEFT Methodology**: LoRA ($r=16, \alpha=32$, targeting all 7 linear projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Trainable Parameters**: **8,798,208** (only **1.75%** of total parameters)
- **Hardware**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Software Stack**: Linux x86_64, CUDA 13.2, PyTorch 2.14.1, Hugging Face `transformers`, `peft`, `trl`, `vllm`

---

## Repository Structure

```text
├── data/
│   ├── legal_sft_dataset.jsonl               # 25 curated legal consultation dialogs in ChatML format
│   ├── legal_dpo_dataset.jsonl               # 20 curated (prompt, chosen, rejected) legal preference pairs
│   ├── stage2_comparative_eval.json          # 3-way benchmark evaluation results (Base vs CPT vs SFT)
│   └── dpo_comparative_eval.json             # SFT vs DPO comparative evaluation results
├── docs/
│   ├── finetuning-foundations.md              # Fine-tuning taxonomy and PEFT deep-dive (LoRA, QLoRA, DoRA, AdaLoRA, (IA)³)
│   ├── baseline_base_model_eval.md            # Raw pre-trained base model benchmark before fine-tuning
│   ├── two_stage_legal_finetuning_pipeline.md # Domain adaptation engineering blueprint
│   ├── stage1_cpt_results.md                  # Empirical Before vs. After results for Stage 1 CPT
│   ├── cpt_dataset_and_hyperparameters_guide.md # Token packing, dataset architecture, and hyperparameter rationale
│   ├── stage2_sft_results.md                  # Stage 2 SFT methodology, prompt-loss masking & 3-way scorecard
│   └── stage3_dpo_results.md                  # Stage 3 DPO mathematical formulation, margin dynamics & post-mortem
├── scripts/
│   ├── train_legal_cpt.py                     # LoRA Continued Pre-Training execution script
│   ├── generate_legal_sft_dataset.py          # Script generating 25 multi-domain legal consultation QA pairs
│   ├── train_legal_sft.py                     # SFT training with prompt-loss masking & weight merging
│   ├── eval_all_stages.py                     # 3-way benchmark evaluation across Base, CPT, and SFT
│   ├── generate_legal_dpo_dataset.py          # Script generating 20 high-contrast legal preference pairs
│   ├── train_legal_dpo.py                     # DPO training script using trl.DPOTrainer & weight merging
│   └── eval_dpo.py                            # SFT vs DPO comparative evaluation benchmark
└── README.md
```

---

## Complete Multi-Stage Scorecard

| Pipeline Stage | Training Objective | Execution Time (RTX 3090) | Loss / Margin Trajectory | Core Breakthrough |
|---|---|---|---|---|
| **Stage 0: Base Model** | Pre-trained next-token prediction | — | — | Baseline: Hallucinated US/UK laws, collapsed into repetition loops. |
| **Stage 1: CPT** | Causal LM over statutory text | **6.28 seconds** | Loss: $2.057 \to 0.929$ (-54.8%) | Grounded model in Indian statutory vocabulary and constitutional doctrines. |
| **Stage 2: SFT** | Prompt-masked ChatML tuning | **8.73 seconds** | Loss: $2.649 \to 2.040$ (-23.0%) | Installed structured consultative persona, eliminated repetition loops. |
| **Stage 3: DPO** | Bradley-Terry preference alignment | **12.13 seconds** | Margin: $0.0 \to +3.502$, Acc: $100\%$ | Suppressed foreign legal transplants, eliminated reckless self-help advice. |

---

## Reproducing the Pipeline

### 1. Stage 1: Continued Pre-Training (CPT)
```bash
python scripts/train_legal_cpt.py
```

### 2. Stage 2: Supervised Fine-Tuning (SFT) with Prompt-Loss Masking
```bash
python scripts/generate_legal_sft_dataset.py
python scripts/train_legal_sft.py
```

### 3. Stage 3: Direct Preference Optimization (DPO)
```bash
python scripts/generate_legal_dpo_dataset.py
python scripts/train_legal_dpo.py
```

### 4. Run Comparative Benchmarks
```bash
python scripts/eval_all_stages.py
python scripts/eval_dpo.py
```
