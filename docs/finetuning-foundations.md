# LLM Fine-Tuning Foundations: Concepts, Taxonomies & PEFT Deep-Dive

Fine-tuning is the process of taking a pre-trained Large Language Model (LLM) that has learned general statistical patterns of language from vast unlabelled corpora and specializing its weights to perform targeted tasks, adopt specific personas, adhere to strict schemas, or master domain-specific knowledge.

---

## 1. Why Fine-Tune? (The Pre-Training Gap)

A base pre-trained model (e.g. `Qwen2.5-0.5B`, `Llama-3-8B`) is purely a **causal next-token statistical predictor**:
- When given the prompt:
  ```text
  User: What is the capital of Australia?
  Assistant:
  ```
  A base model has **no inherent concept that it is an assistant**. Instead of answering, it frequently continues the pattern it saw during web pre-training:
  ```text
  What is the capital of New Zealand?
  What is the capital of Canada?
  1. Sydney 2. Melbourne 3. Canberra
  ```
- **Fine-tuning bridges this gap** by conditioning the model to transition from text completion to **instruction execution**, structured reasoning, and conversational interaction.

---

## 2. Classification Taxonomy: Two Core Dimensions

Fine-tuning is classified along two independent axes:

```
                                 FINE-TUNING TAXONOMY
                                          │
       ┌──────────────────────────────────┴──────────────────────────────────┐
       ▼                                                                     ▼
[AXIS 1: BY TRAINING OBJECTIVE]                                     [AXIS 2: BY PARAMETER SCOPE]
  ├── Continued Pre-Training (CPT)                                    ├── Full-Parameter Fine-Tuning (FPFT)
  ├── Supervised Fine-Tuning (SFT)                                    └── Parameter-Efficient Fine-Tuning (PEFT)
  └── Preference Alignment                                                  ├── LoRA (Low-Rank Adaptation)
        ├── RLHF (Reward Model + PPO)                                       ├── QLoRA (4-bit Quantized LoRA)
        ├── DPO (Direct Preference Optimization)                            ├── DoRA (Weight-Decomposed LoRA)
        └── ORPO / KTO                                                      ├── AdaLoRA (Adaptive Rank Allocation)
                                                                            ├── (IA)³ (Inhibited/Amplified Activations)
                                                                            ├── Prefix & Prompt Tuning
                                                                            └── Bottleneck Adapters (Houlsby/Pfeiffer)
```

---

## 3. Axis 1: By Training Objective & Methodology

### 1. Continued Pre-Training (CPT / Domain-Adaptive Pre-Training)
- **Objective**: Ingest raw domain knowledge, jargon, and syntax before task training.
- **Data Format**: Large corpora of raw, unlabelled text (e.g. 10 GB of legal filings, clinical trials, or private codebases).
- **Loss Function**: Standard autoregressive causal language modeling loss computed across **all tokens**:
  $$\mathcal{L}_{\text{CPT}} = - \sum_{i=1}^N \log P(x_i \mid x_1, \dots, x_{i-1})$$
- **Use Case**: When a base model lacks knowledge of domain-specific entities (e.g. internal APIs, local tax codes).

---

### 2. Supervised Fine-Tuning (SFT / Instruction Tuning)
- **Objective**: Teach the model conversational turn-taking, task execution, format constraints (JSON, Markdown), and system prompt adherence.
- **Data Format**: Curated input-output pairs structured with chat templates:
  ```json
  {"prompt": "<|im_start|>user\nWrite a merge sort in Python.<|im_end|>\n<|im_start|>assistant\n", "response": "```python\ndef merge_sort..."}
  ```
- **Loss Masking**: Loss is calculated **strictly on the response tokens**. Prompt tokens are masked with label `-100` so the optimizer does not penalize or reinforce the user's input.

---

### 3. Preference Alignment (RLHF / DPO / ORPO)
- **Objective**: Align model outputs with human intent, nuanced style, truthfulness, and safety boundaries.
- **RLHF (Reinforcement Learning from Human Feedback)**:
  1. Train a separate **Reward Model** on pairs where humans ranked response $y_w \succ y_l$.
  2. Optimize the base model policy using **PPO (Proximal Policy Optimization)** against the scalar reward, with a KL-divergence penalty to prevent policy collapse.
- **DPO (Direct Preference Optimization)**:
  - Derives an exact mathematical equivalence between the reward function and the optimal policy:
    $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
  - Completely eliminates the separate reward model and unstable PPO reinforcement learning loop.

---

## 4. Axis 2: By Parameter Scope (Hardware & Memory Efficiency)

### 1. Full-Parameter Fine-Tuning (FPFT)
- **Mechanics**: Updates every single weight matrix $W \in \mathbb{R}^{d \times k}$ in every Transformer layer.
- **Memory Anatomy (AdamW in BF16)**:
  - Model Weights ($W$): $2 \text{ bytes/parameter}$
  - Gradients ($\nabla W$): $2 \text{ bytes/parameter}$
  - AdamW 1st Moment Vector ($m$): $4 \text{ bytes/parameter}$ (FP32)
  - AdamW 2nd Moment Vector ($v$): $4 \text{ bytes/parameter}$ (FP32)
  - Master Weights (FP32 copy for numerical stability): $4 \text{ bytes/parameter}$
  - **Total Static Optimizer Footprint**: $\mathbf{16 \text{ to } 18 \text{ bytes per parameter}}$!
  - *Example*: Fine-tuning a 7B model requires $\approx 7 \times 18 = 126 \text{ GB}$ VRAM purely for weights and optimizer states, before allocating memory for activations and KV caches.
- **Pitfalls**: High risk of **catastrophic forgetting** and massive GPU cluster costs.

---

### 2. Parameter-Efficient Fine-Tuning (PEFT)
PEFT methods freeze the pre-trained parameters $W_0$ and introduce a tiny fraction ($<1\%$) of auxiliary trainable parameters.

```
       FULL-PARAMETER FINE-TUNING                    PEFT (e.g. LoRA)
             ┌─────────────┐                          ┌─────────────┐
             │  W = W₀+ΔW  │ (All parameters         │     W₀      │ (Frozen base weights)
             │  (Trainable)│  updated)                │  (Frozen)   │  No gradients/Adam states
             └─────────────┘                          └──────┬──────┘
                                                             │ +
                                                      ┌──────┴──────┐
                                                      │  B × A = ΔW │ (Trainable rank r << d)
                                                      │ (Trainable) │  ~0.1% parameters!
                                                      └─────────────┘
```

#### A. LoRA (Low-Rank Adaptation)
- **Formulation**: Approximates the dense weight update matrix $\Delta W \in \mathbb{R}^{d \times k}$ with two low-rank matrices $A$ and $B$:
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A)$$
  where $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$, $B \in \mathbb{R}^{d \times r}$ is initialized to zero, and rank $r \ll \min(d, k)$.
- **Scaling Factor $\alpha$**: Controls the magnitude of the adapter's influence.
- **Weight Merging**: At deployment, $W_{\text{merged}} = W_0 + \frac{\alpha}{r}(BA)$ is pre-computed offline, yielding **zero added latency** during inference.

---

#### B. QLoRA (Quantized Low-Rank Adaptation)
- **Formulation**: Keeps base weights $W_0$ compressed in **4-bit NormalFloat (NF4)** precision while calculating activations and LoRA updates in 16-bit (BF16):
  $$Y = X \cdot \text{dequantize}(\text{NF4}, W_0) + \frac{\alpha}{r} (X \cdot A^T B^T)$$
- **Three Core Pillars**:
  1. **NF4 (NormalFloat 4)**: Quantization bins spaced according to the empirical normal distribution of neural network weights.
  2. **Double Quantization (DQ)**: Quantizes the group scale factors themselves, reducing scale overhead from 0.5 bits/param to 0.127 bits/param.
  3. **Paged Optimizers**: Leverages CUDA Unified Memory to automatically page memory allocation spikes into host CPU RAM during backward passes.

---

#### C. DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Problem with LoRA**: In full fine-tuning, weight updates exhibit independent variations in both **magnitude** (vector norm) and **direction** (orientation). Standard LoRA couples both into $\Delta W$, causing proportional changes that limit learning capacity.
- **Formulation**: DoRA explicitly decomposes the weight matrix into its **magnitude vector** $m \in \mathbb{R}^{1 \times k}$ and **directional matrix** $V$:
  $$W = m \odot \frac{V}{\|V\|_c} = m \odot \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c} = m \odot \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_c}$$
  where $\|\cdot\|_c$ is the column-wise $L_2$ norm.
- **Result**: DoRA consistently outperforms standard LoRA and often **matches or exceeds Full-Parameter Fine-Tuning** while retaining PEFT parameter efficiency.

---

#### D. AdaLoRA (Adaptive Budget Allocation for LoRA)
- **Problem with LoRA**: Standard LoRA assigns an identical, static rank $r$ (e.g. $r=16$) uniformly across every layer and projection (`q, k, v, o, gate, up, down`), ignoring the fact that deeper layers and cross-attention blocks often require higher capacity.
- **Formulation**: Parameterizes incremental updates using **Singular Value Decomposition (SVD)**:
  $$\Delta W = P \Lambda Q$$
  where $P$ and $Q$ are left and right singular matrices, and $\Lambda = \text{diag}(\lambda)$ contains singular values.
- **Dynamic Rank Pruning**: During training, AdaLoRA measures the importance of each singular triplet $(P_i, \lambda_i, Q_i)$ and iteratively prunes zero-importance triplets, dynamically routing parameter budget to the most critical layers.

---

#### E. $(IA)^3$ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- **Mechanics**: Instead of adding low-rank matrices, $(IA)^3$ introduces **element-wise scaling vectors** $l_k \in \mathbb{R}^{d}$, $l_v \in \mathbb{R}^{d}$, and $l_{ff} \in \mathbb{R}^{d_{\text{ff}}}$ that rescale intermediate activations:
  $$K = l_k \odot (X W_k), \quad V = l_v \odot (X W_v), \quad \text{FFN}_{\text{out}} = l_{ff} \odot \gamma(X W_1) W_2$$
- **Parameter Count**: Requires **$<0.01\%$ of model parameters** (orders of magnitude smaller than LoRA), making it ideal for extreme multi-task serving.

---

#### F. Prefix Tuning & Prompt Tuning
- **Prompt Tuning**: Prepends virtual continuous trainable embedding tokens $P \in \mathbb{R}^{l \times d}$ to the input word embeddings.
- **Prefix Tuning**: Extends prompt tuning by prepending trainable virtual keys and values $P_K, P_V \in \mathbb{R}^{l \times d}$ to **every attention layer**.
- **Limitation**: Reduces effective context window length because virtual prefix tokens consume positions in the attention budget.

---

#### G. Bottleneck Adapters (Houlsby & Pfeiffer)
- **Mechanics**: Inserts small feed-forward residual bottleneck blocks:
  $$\text{Adapter}(x) = \sigma(x W_{\text{down}}) W_{\text{up}} + x$$
- **Houlsby Architecture**: Inserts two adapters per Transformer block (one after Multi-Head Attention, one after FFN).
- **Pfeiffer Architecture**: Inserts one adapter per block (after FFN only) to reduce sequential latency.

---

## 5. Master PEFT Comparison Matrix

| PEFT Method | Trainable Params (%) | Memory Footprint (VRAM) | Added Inference Latency | Merging Capability | Primary Strength |
|---|:---:|:---:|:---:|:---:|---|
| **Full Fine-Tuning** | 100% | Ultra High ($\sim 18\times \text{Params}$) | 0 ms | Native | Maximum task adaptation capacity |
| **LoRA** | 0.05% – 0.5% | Low ($\sim 0.35\times \text{FPFT}$) | 0 ms (when merged) | Yes ($W_0 + BA$) | Industry standard; zero latency overhead |
| **QLoRA** | 0.05% – 0.5% | **Ultra Low** ($\sim 0.2\times \text{FPFT}$) | 0 ms (when merged) | Yes | Fits large models onto single consumer GPUs |
| **DoRA** | 0.05% – 0.5% | Low | 0 ms (when merged) | Yes | Outperforms LoRA; matches Full Fine-Tuning |
| **AdaLoRA** | 0.05% – 0.3% | Low | 0 ms (when merged) | Yes | Automatically allocates rank budget where needed |
| **$(IA)^3$** | **<0.01%** | Ultra Low | 0 ms (when merged) | Yes | Minimal parameters; extreme multi-task scaling |
| **Prefix Tuning** | 0.1% – 1.0% | Low | Minor (extra KV tokens) | No | Preserves base weights without modifying GEMM |
| **Bottleneck Adapters**| 0.5% – 2.0% | Moderate | High (sequential layers) | No | Modular modularity for domain switching |
