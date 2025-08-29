# Customer Support Fine-Tuning with TinyLlama (MLX-LM)

This repository contains my experiments on fine-tuning a small open-source LLM (TinyLlama 1.1B) for **customer support tasks** using Apple’s [MLX](https://github.com/ml-explore/mlx) framework.

The goal is to adapt a lightweight model to handle typical customer service intents such as:
- Order cancellation  
- Returns and refunds  
- Delivery tracking  
- Address changes  
- Contacting support  
- Basic product information  

---

## Hardware Setup

I trained the model on a **MacBook Pro with Apple Silicon (M1/M2, 16 GB Unified Memory)** using **MLX**, Apple’s optimized framework for on-device machine learning.  

Because Apple Silicon GPUs (Metal backend) have **limited memory for large sequence lengths and big batch sizes**, I adjusted the training configuration accordingly:
- **Small context length (256 tokens)** to reduce quadratic memory growth.  
- **Micro batch size of 1** to fit training into memory.  
- **LoRA fine-tuning** instead of full model training to keep trainable parameters small (~0.07% of total).  

This setup demonstrates that **LLM fine-tuning is possible even on consumer laptops**, without requiring an expensive GPU cluster.

---

## Dataset

I prepared a **synthetic customer support dataset** with templated examples (order numbers, placeholders for company info).  

- **Size:** a few hundred examples (e.g., 25–200 for testing, extendable to a few thousand).  
- **Format:** JSONL in chat-style (`user`/`assistant` messages).  
- **Why small?**  
  - For demonstration and prototyping, a small curated dataset is enough to show measurable effects (style transfer, tone, dialog policy).  
  - Larger datasets would cause memory issues on Mac hardware during training.  
  - The focus is on **teaching the model tone and structure**, not memorizing thousands of examples.  

---

## Training Commands

I used MLX’s built-in LoRA fine-tuning module. Example commands:

**Start training:**
```bash
python -m mlx_lm lora --config lora_config.yaml --train --data ./data
```

**Fuse adapter with base model:**
```bash
python -m mlx_lm fuse \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter adapters/0000100_adapters.safetensors \
  --save-path out/tinyllama-fused
```

**Generate responses with fused model**
```bash
python -m mlx_lm generate \
  --model out/tinyllama-fused \
  --prompt "User: I want to cancel order {{Order Number}}.\nAssistant:" \
  --max-tokens 200
```

## Results
- The model quickly adapted to polite customer support style after only ~100 training steps.
- Training loss plateaued early on small datasets, so 1–2 epochs were sufficient.
- A fused model (base + adapter) can be exported and used for inference.

