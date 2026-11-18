# Finetuning-LLMs: Enterprise Two-Stage Domain Adaptation (CPT + SFT)

A production-grade implementation of domain-adaptive **Continued Pre-Training (CPT)** followed by **Supervised Fine-Tuning (SFT)** using Parameter-Efficient Fine-Tuning (PEFT / LoRA) on NVIDIA GPUs.

---

## Architecture Overview

Enterprise domain specialization requires a two-stage training pipeline:

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
   [Specialized Legal AI Assistant]  <-- Standalone merged checkpoint ready for vLLM serving
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
│   └── legal_sft_dataset.jsonl               # 25 curated legal consultation dialogs in ChatML format
├── docs/
│   ├── finetuning-foundations.md              # Fine-tuning taxonomy and PEFT deep-dive (LoRA, QLoRA, DoRA, AdaLoRA, (IA)³)
│   ├── baseline_base_model_eval.md            # Raw pre-trained base model benchmark before fine-tuning
│   ├── two_stage_legal_finetuning_pipeline.md # Two-stage domain adaptation engineering blueprint
│   ├── stage1_cpt_results.md                  # Empirical Before vs. After results for Stage 1 CPT
│   ├── cpt_dataset_and_hyperparameters_guide.md # Token packing, dataset architecture, and hyperparameter rationale
│   └── stage2_sft_results.md                  # Stage 2 SFT methodology, prompt-loss masking & 3-way scorecard
├── scripts/
│   ├── train_legal_cpt.py                     # LoRA Continued Pre-Training execution script
│   ├── generate_legal_sft_dataset.py          # Script generating 25 multi-domain legal consultation QA pairs
│   ├── train_legal_sft.py                     # SFT training with prompt-loss masking & weight merging
│   └── eval_all_stages.py                     # 3-way benchmark evaluation across Base, CPT, and SFT
└── README.md
```

---

## Master 3-Way Empirical Benchmark

| Evaluation Query | Stage 0: Raw Base Model | Stage 1: Base + CPT Adapter | Stage 2: Merged SFT Model (CPT + SFT) |
|---|---|---|---|
| **Arrest Without Warrant (Sec 41 CrPC)** | Hallucinated US attorney timelines, followed by synthetic FAQ loop (`orda\norda...`) | Grounded in Section 41 CrPC and proportionality tests, but generated multi-turn ramblings | **Structured legal breakdown**: Section 41 CrPC, Article 21/22 rights, 24-hr magistrate rule, actionable remedies |
| **Writ Jurisdiction (Art 32 vs Art 226)** | Collapsed into repeating `-transitional\nExplain the...` 25 times | Grounded Article 32 & 226 definitions, then degenerated into repeating `CTYPE: Question\nCTYPE: Answer` 40 times | **Full comparative breakdown**: Jurisdiction scope, limitation periods, presidential consent, remedies |
| **Basic Structure Doctrine (Art 368)** | Degenerated into repeating `-transitionalstructure` over **150 times** | Mentioned Art 368 amending power, but hallucinated US Indian law (*Cherokee Nation v. US*) | **Precise constitutional analysis**: Explains amending power limits, unamendable core features, and judicial review |
| **Cheque Dishonour (Sec 138 NI Act)** | Leaked prompt artifacts (`erokee system...`) and confused civil contract damages | Accurately cited Section 138(1)/(2), then degenerated into repeating `-transitional` **190 times** | **Production advisory structure**: Statutory conditions, mandatory 15-day notice, limitation timeline, disclaimer |
| **Execution Speed** | — | **6.28 seconds** on RTX 3090 | **8.73 seconds** on RTX 3090 |

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

### 3. Run Comparative Evaluation Across All Stages
```bash
python scripts/eval_all_stages.py
```
