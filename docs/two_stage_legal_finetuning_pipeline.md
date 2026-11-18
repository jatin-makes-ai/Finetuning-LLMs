# Two-Stage Legal Domain Adaptation: Continued Pre-Training (CPT) to Supervised Fine-Tuning (SFT)

This document specifies the technical architecture for transforming a general-purpose base model (`Qwen2.5-0.5B`) into an **Indian Legal Domain Assistant** using a two-stage training methodology.

---

## 1. The Two-Stage Engineering Blueprint

```
[Raw Pre-Trained Base Model: Qwen2.5-0.5B]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ STAGE 1: Continued Pre-Train │  <-- Causal Language Modeling over raw legal text:
   │            (CPT)             │      Constitution of India, Landmark SC Precedents,
   └──────────────┬───────────────┘      Statutory Frameworks, Legal Jurisprudence
                  │
                  ▼
      [Legal Domain-Adaptive Base]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ STAGE 2: Instruction Tuning  │  <-- Supervised Fine-Tuning with ChatML formatting:
   │            (SFT)             │      Legal QA, Statutory Consultation, Case Analysis,
   └──────────────┬───────────────┘      Prompt Loss Masking
                  │
                  ▼
   [Specialized Legal AI Assistant]
```

---

## 2. Stage 1: Domain-Adaptive Continued Pre-Training (CPT)

### Objective
Expose the general pre-trained weights to dense legal terminology, constitutional articles, case precedents, and statutory syntax so the model forms strong internal representations of legal entities.

### Legal Corpus Components
1. **The Constitution of India**:
   - Preamble
   - Part III: Fundamental Rights (Articles 12 to 35: Equality, Freedom of Speech, Protection of Life & Personal Liberty, Constitutional Remedies)
   - Part IV: Directive Principles of State Policy (Articles 36 to 51)
   - Union & State Judiciary (Articles 124–147, 214–227)
   - Constitutional Writs (*Habeas Corpus, Mandamus, Prohibition, Quo Warranto, Certiorari*)
2. **Landmark Precedents & Jurisprudence**:
   - *Kesavananda Bharati v. State of Kerala (1973)*: Basic Structure Doctrine
   - *Maneka Gandhi v. Union of India (1978)*: Substantive Due Process under Article 21
   - *Justice K.S. Puttaswamy v. Union of India (2017)*: Fundamental Right to Privacy
   - *Vishaka v. State of Rajasthan (1997)*: Guidelines against Workplace Harassment
   - *Navtej Singh Johar v. Union of India (2018)*: Decriminalization of Section 377
3. **Statutory Frameworks**:
   - Essential provisions of IPC / Bharatiya Nyaya Sanhita (BNS)
   - Information Technology Act (Section 66A, Intermediary Guidelines)
   - Consumer Protection Act

### Training Configuration (CPT)
- **Task**: Causal Language Modeling (CLM)
- **Loss**: Autoregressive cross-entropy across all sequence tokens
- **Packing**: Concatenated token streams segmented into chunks of `max_seq_len = 512`
- **Parameter Efficiency**: LoRA ($r=16, \alpha=32$) on attention & MLP projections

---

## 3. Stage 2: Legal Supervised Fine-Tuning (SFT)

### Objective
Condition the domain-adapted model to behave as an articulate, accurate, and ethical legal assistant using conversational turn-taking.

### Training Mechanics
- **Template**: Qwen ChatML standard:
  ```text
  <|im_start|>system
  You are an expert legal assistant specializing in Indian Constitutional and Statutory Law. Provide clear, accurate, and structured legal analysis with relevant citations.<|im_end|>
  <|im_start|>user
  Under what conditions can the Right to Personal Liberty under Article 21 be restricted?<|im_end|>
  <|im_start|>assistant
  Under the landmark ruling of Maneka Gandhi v. Union of India (1978), any procedure depriving a person of personal liberty under Article 21 must satisfy three cumulative tests:
  1. Legality: It must be backed by a validly enacted law.
  2. Fairness: The procedure prescribed must be just, fair, and reasonable, not arbitrary.
  3. Proportionality: The restriction must bear a direct, non-excessive nexus to the legitimate state objective.<|im_end|>
  ```
- **Prompt Loss Masking**: Loss is evaluated **only on tokens following `<|im_start|>assistant`**, ensuring the model learns generation rather than memorizing questions.

---

## 4. Evaluation Framework: The 3-Way Before-and-After Matrix

We will evaluate three checkpoints on identical legal prompts:
1. **Checkpoint 0**: Raw Pre-Trained `Qwen2.5-0.5B`
2. **Checkpoint 1**: After Stage 1 CPT (Legal Knowledge Ingestion)
3. **Checkpoint 2**: After Stage 2 SFT (Instruction-Following Legal Assistant)
