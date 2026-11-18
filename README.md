# Finetuning-LLMs: Complete Practical Curriculum

Welcome to the **Finetuning-LLMs** project repository! This repository documents our hands-on engineering journey through LLM fine-tuning, starting from foundations and PEFT architectures to Supervised Fine-Tuning (SFT), LoRA/QLoRA, and Preference Alignment.

---

## 📚 Curriculum Structure

1. **Foundations & Taxonomies** ([`docs/finetuning-foundations.md`](docs/finetuning-foundations.md))
   - Why fine-tune? The Pre-Training next-token gap.
   - Classification by objective: Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), Preference Alignment (RLHF / DPO / ORPO).
   - Classification by parameter scope: Full-Parameter Fine-Tuning (FPFT) vs. Parameter-Efficient Fine-Tuning (PEFT).
   - Deep-dive into PEFT architectures: LoRA, QLoRA, DoRA, AdaLoRA, $(IA)^3$, Prefix/Prompt Tuning, and Bottleneck Adapters.
2. **Base vs. Instruct: Empirical Comparison**
   - Evaluating a raw, pre-trained base model (`Qwen2.5-0.5B`) before fine-tuning.
   - Observing text completion rambling vs. instruction following.
3. **Supervised Instruction Tuning (SFT)**
   - Curating conversational instruction datasets with chat templates.
   - Training LoRA adapters with `peft` and `trl`.
4. **Evaluation & Weight Merging**
   - Evaluating identical prompts Before vs. After fine-tuning.
   - Merging adapter weights ($W_0 + BA$) and serving via vLLM.

---

## 🛠️ Hardware & Environment

- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Host**: AMD EPYC 7302 16-Core, 503 GB RAM
- **Software**: PyTorch 2.14.1 (CUDA 13.2), Transformers, PEFT, TRL, vLLM
