# Finetuning-LLMs: Enterprise Two-Stage Domain Adaptation (CPT + SFT)

A production-grade implementation of domain-adaptive **Continued Pre-Training (CPT)** followed by **Supervised Fine-Tuning (SFT)** using Parameter-Efficient Fine-Tuning (PEFT / LoRA) on NVIDIA GPUs.

---

## Architecture Overview

Enterprise domain specialization requires a two-stage training pipeline:

```
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
   [Specialized Legal AI Assistant]
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
├── docs/
│   ├── finetuning-foundations.md              # Fine-tuning taxonomy and PEFT deep-dive (LoRA, QLoRA, DoRA, AdaLoRA, (IA)³)
│   ├── baseline_base_model_eval.md            # Raw pre-trained base model benchmark before fine-tuning
│   ├── two_stage_legal_finetuning_pipeline.md # Two-stage domain adaptation engineering blueprint
│   ├── stage1_cpt_results.md                  # Empirical Before vs. After results for Stage 1 CPT
│   └── cpt_dataset_and_hyperparameters_guide.md # Token packing, dataset architecture, and hyperparameter rationale
├── scripts/
│   └── train_legal_cpt.py                     # LoRA Continued Pre-Training execution script
└── README.md
```

---

## Empirical Benchmark: Stage 1 CPT Impact

| Metric / Test | Baseline (Raw Base Model) | Stage 1 (Qwen + Legal CPT Adapter) | Impact |
|---|---|---|---|
| **Article 21 Text** | Hallucinated US 5th Amendment ("capital offence") | Accurately generated **"procedure established by law"** | Eliminated foreign law hallucination |
| **Habeas Corpus Definition** | Claimed writ prevents execution of non-bailable prisoners | Accurately stated **"secure the release of an unlawfully detained person"** | Constitutional accuracy restored |
| **Article 368 Amendment Limits**| Hallucinated British Parliament & UK constitutional system | Correctly stated **"cannot alter the basic structure of the Constitution"** | Doctrine grounded in Indian jurisprudence |
| **Right to Privacy Holding** | Stalled in infinite 4x repetition loop | Accurately cited **Article 21 & 9-judge bench holding** | Eliminated generation degradation |
| **Training Duration** | — | **6.28 seconds** on RTX 3090 | 54.8% loss drop ($2.057 \rightarrow 0.929$) |
