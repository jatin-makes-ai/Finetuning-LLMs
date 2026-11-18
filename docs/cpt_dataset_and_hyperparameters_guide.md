# CPT Dataset Architecture & Hyperparameter Engineering Guide

This guide documents the dataset engineering, token packing, PEFT architecture, and hyperparameter selection rationale used during **Stage 1 Continued Pre-Training (CPT)** on `Qwen2.5-0.5B`.

---

## 1. Dataset Architecture & Construction

### Format: Unsupervised Continuous Token Stream
Unlike Supervised Fine-Tuning (SFT) which requires question-answer pairs, Continued Pre-Training (CPT) exposes the model to raw domain text. The goal is to maximize token density and minimize uninformative padding.

### Substantive Coverage (6 Legal Modules)
1. **The Preamble & Foundational Constitutional Philosophy**:
   - Sovereign Socialist Secular Democratic Republic, Justice, Liberty, Equality, Fraternity.
   - Interpretation under *Kesavananda Bharati* (the Preamble as an integral part of the basic structure).
2. **Part III Fundamental Rights (Articles 14 to 21)**:
   - *Article 14*: Equality before law; *E.P. Royappa* rule against state arbitrariness.
   - *Article 19(1)(a) & 19(2)*: Freedom of speech and legitimate grounds for reasonable restrictions.
   - *Article 20*: Protection against ex-post facto laws, double jeopardy, and self-incrimination.
   - *Article 21*: Protection of life and personal liberty; *Maneka Gandhi (1978)* test of just, fair, and reasonable procedure.
3. **Constitutional Remedies & The 5 Extraordinary Writs (Articles 32 & 226)**:
   - Exact legal definitions of *Habeas Corpus, Mandamus, Prohibition, Certiorari, and Quo Warranto*.
   - Supreme Court Article 32 vs. High Court Article 226 jurisdiction.
4. **The Basic Structure Doctrine & Article 368**:
   - 13-judge constitutional bench ruling in *Kesavananda Bharati (1973)*.
   - Explicit limits on Parliament's amending power under Article 368 (cannot alter secularism, federalism, judicial review, or democracy).
   - *Minerva Mills (1980)* balance between Fundamental Rights and Directive Principles.
5. **The Right to Privacy (*Puttaswamy 2017*)**:
   - 9-judge bench holding under Article 21; overruling of *M.P. Sharma* and *Kharak Singh*.
   - The strict 3-fold proportionality test (*Legality, Legitimate State Aim, Proportionality*).
6. **Statutory Criminal Law & Legal Maxims**:
   - *Actus Reus* (physical act) and *Mens Rea* (blameworthy mental state).
   - Cognizable vs. Non-cognizable offenses, warrant requirements.
   - Public Interest Litigation (PIL) and relaxation of *locus standi*.
   - Core Latin maxims: *Audi Alteram Partem, Nemo Judex In Causa Sua, Res Judicata*.

---

### Sequence Packing (No Wasteful Padding)
Standard naive tokenization pads short text passages with `<pad>` tokens, wasting 40%–60% of GPU compute and corrupting attention masks. We implemented **sequence packing**:
```python
tokenized = tokenizer(text, truncation=False, return_attention_mask=False)["input_ids"]
chunks = [tokenized[i : i + BLOCK_SIZE] for i in range(0, total_length - BLOCK_SIZE + 1, BLOCK_SIZE)]
```
- Total Tokens: **19,456 tokens**
- Training Sequences: **38 continuous blocks of 512 tokens**
- Padding tokens: **0 (100% token compute efficiency)**

---

## 2. PEFT Technique: LoRA for Causal Language Modeling

We applied **Low-Rank Adaptation (LoRA)**:
- **Base Model ($W_0$)**: Frozen (502.8M parameters).
- **Trainable Parameters**: **8,798,208** (**1.75% of total model**).
- **Formula**:
  $$W = W_0 + \frac{\alpha}{r} (B \times A)$$

---

## 3. Hyperparameter Deep-Dive: Mechanics & Selection Rationale

| Hyperparameter | Value | Mechanical Function | Selection Rationale |
|---|:---:|---|---|
| **`r` (Rank)** | `16` | The inner bottleneck dimension of matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$. Dictates degrees of freedom. | For shallow style adjustments, $r=4$ or $r=8$ suffices. For domain adaptation / CPT (absorbing new legal entities, citations, and statutory phrases), $r=16$ provides necessary capacity without overfitting. |
| **`lora_alpha` ($\alpha$)** | `32` | Constant numerator in scaling factor $\frac{\alpha}{r} = \frac{32}{16} = 2.0$. Scales magnitude of adapter update. | Established research standard is **$\alpha = 2 \times r$**. Setting $\alpha = 2r$ stabilizes gradient flow and decouples rank tuning from learning rate adjustments. |
| **`target_modules`** | `All 7` | Injects adapters into `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. | Research shows adapting both Attention and MLP/FFN layers yields significantly higher domain adaptation capacity at minimal VRAM overhead compared to attention-only LoRA. |
| **`lora_dropout`** | `0.05` | Drop probability for matrix $A$ during training passes. | Regularization preventing the 8.8M adapter parameters from memorizing exact token sequences verbatim. |
| **`learning_rate`** | `2e-4` | Step size of AdamW optimizer. | In full fine-tuning, learning rates must be small ($10^{-5}$) to prevent catastrophic forgetting. Because base weights are frozen and adapters are initialized near zero, LoRA requires a $10\times$ higher learning rate ($2 \times 10^{-4}$) to converge swiftly. |
| **`per_device_train_batch_size`** | `4` | Number of 512-token blocks per forward pass. | Keeps RTX 3090 Tensor Core utilization high while consuming only ~3.5 GB VRAM. |
| **`gradient_accumulation_steps`**| `2` | Accumulates gradients across micro-batches before calling optimizer step. | Effective batch size = $4 \times 2 = 8$ blocks (**4,096 tokens per step**). Smoothes gradient variance. |
| **`num_train_epochs`** | `3` | Total passes over the legal corpus. | Loss dropped consistently: Epoch 1 (`2.057`) $\rightarrow$ Epoch 2 (`1.417`) $\rightarrow$ Epoch 3 (`0.9291`). |
| **`warmup_steps`** | `5` | Linear ramp-up of learning rate from 0. | Prevents gradient instability in step 0 before Adam's running variance statistics stabilize. |
| **`block_size`** | `512` | Attention context window per training block. | Long enough to encompass full constitutional articles, case holdings, and legal arguments in a single context window. |
