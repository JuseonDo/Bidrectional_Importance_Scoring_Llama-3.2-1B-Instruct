# Bidirectional Importance Scoring Llama-3.2-1B-Instruct

This repository provides a **trained Llama-3.2-1B-Instruct model without a causal mask**, designed for **token-level importance scoring**.

---

## 📌 Model Overview

- **Base Model**: Llama-3.2-1B-Instruct  
- **Architecture Modification**:  
  - **Causal mask removed**
  - Enables **bidirectional context modeling**
- **Training Dataset**:  
  - `microsoft/MeetingBank-LLMCompressed`
- **Primary Use Case**:  
  - Scoring the **importance of individual tokens** in a sequence  
  - Useful for:
    - Token pruning
    - Compression-aware generation
    - Importance-based reranking
    - Analysis of token contributions in LLM outputs

---

## 🎯 Intended Use

This model is **not** intended for standard autoregressive text generation.  
Instead, it is optimized for **importance estimation**, where access to both left and right context is required.

Example applications include:
- Token importance estimation for long-context compression
- Preprocessing for retrieval-augmented generation (RAG)
- Interpretability and attribution analysis
- Training or supervising token-selection modules

---

## 📦 Checkpoints

The trained model checkpoints are available on Hugging Face:

👉 **Hugging Face Hub**  
https://huggingface.co/Juseon/Bidrectional_Importance_Scoring_Llama-3.2-1B-Instruct

---