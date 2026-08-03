# 🩺 Medical Chatbot using QLoRA Fine-Tuning

A domain-specific AI medical assistant built by fine-tuning **Qwen2.5-3B-Instruct** using **QLoRA (Quantized Low-Rank Adaptation)** on the **ChatDoctor** dataset from Hugging Face.

The project demonstrates how a large language model can be adapted to a specialized medical domain while training only a small number of additional parameters, making fine-tuning efficient in terms of both memory and computation.

---

# 📌 Project Overview

Large Language Models (LLMs) such as Qwen possess strong general language understanding, but they are not specialized in any particular domain.

This project fine-tunes the model on a medical question-answer dataset so that it can provide more natural and domain-focused medical responses.

Instead of retraining the entire model, the project uses **QLoRA**, which significantly reduces GPU memory usage while maintaining high performance.

---

# 🚀 Features

- Fine-tuned **Qwen2.5-3B-Instruct**
- Efficient **QLoRA** fine-tuning
- 4-bit model quantization using BitsAndBytes
- LoRA adapters with PEFT
- Supervised Fine-Tuning (SFT)
- Medical question-answer chatbot
- Google Colab compatible
- Hugging Face ecosystem

---

# 🧠 Model

**Base Model**

```
Qwen/Qwen2.5-3B-Instruct
```

This instruction-tuned Large Language Model was used as the foundation for the chatbot.

---

# 📚 Dataset

**Dataset**

```
lavita/ChatDoctor-HealthCareMagic-100k
```

The dataset contains thousands of medical conversations between patients and doctors, including:

- Medical symptoms
- Disease explanations
- Treatment suggestions
- Healthcare guidance
- Medical question-answer pairs

The dataset was formatted into the Qwen chat template before training.

---

# 🏗️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL (Transformer Reinforcement Learning)
- PEFT
- BitsAndBytes
- QLoRA
- Google Colab

---

# ⚙️ Fine-Tuning Method

This project uses **QLoRA**, which combines:

- 4-bit Quantization (NF4)
- LoRA adapters
- PEFT
- Supervised Fine-Tuning (SFT)

Instead of updating all model parameters, only the LoRA adapter weights are trained.

This dramatically reduces:

- GPU memory usage
- Training time
- Storage requirements

while maintaining strong performance.

---

# ⚡ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-3B-Instruct |
| Fine-Tuning | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Learning Rate | 2e-4 |
| Epochs | 2 |
| Max Length | 512 |

---

# 💬 Example

### Input

```
What are the symptoms of diabetes?
```

### Output

```
Hello and welcome to Chat Doctor.

Diabetes is a chronic disease characterized by elevated blood sugar levels.
Common symptoms include:

• Increased thirst
• Frequent urination
• Unexplained weight loss
• Fatigue
• Blurred vision
• Slow wound healing

If these symptoms persist, consult a healthcare professional for proper diagnosis.
```

---

# 📊 What the Model Learned

After fine-tuning, the chatbot learned to:

- Respond using medical terminology.
- Answer healthcare-related questions more naturally.
- Follow a doctor–patient conversational style.
- Generate more detailed medical explanations.
- Preserve the general language understanding of the original Qwen model while improving performance on medical conversations.

---

# 📈 Future Improvements

- Train on larger and more diverse medical datasets.
- Integrate Retrieval-Augmented Generation (RAG) with trusted medical resources.
- Build a web interface using Gradio or Streamlit.
- Deploy the chatbot as a FastAPI service.
- Evaluate the model using medical benchmark datasets.
- Support multilingual medical conversations.
