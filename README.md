# 🩺 Medical Chatbot using QLoRA Fine-Tuning

This project demonstrates how to build a **domain-specific medical chatbot** by fine-tuning **Qwen2.5-3B-Instruct** using **QLoRA (Quantized Low-Rank Adaptation)** on the **ChatDoctor-HealthCareMagic-100k** dataset from Hugging Face.

Instead of retraining the entire Large Language Model, only lightweight **LoRA adapters** are trained while the original model remains frozen. This approach significantly reduces GPU memory usage and training time, making it possible to fine-tune a powerful language model on **Google Colab with a Tesla T4 GPU**.

---

# 📌 Project Overview

Large Language Models (LLMs) are capable of understanding and generating natural language across many tasks. However, they are trained for general-purpose conversations and are not specialized in specific domains.

In this project, the **Qwen2.5-3B-Instruct** model is adapted for medical question answering using supervised fine-tuning on doctor–patient conversations from the **ChatDoctor-HealthCareMagic-100k** dataset.

The notebook demonstrates the complete fine-tuning pipeline, including:

- Loading a pretrained Large Language Model
- Preparing a medical instruction dataset
- Formatting conversations using the Qwen chat template
- Applying QLoRA with 4-bit quantization
- Fine-tuning using TRL's SFTTrainer
- Saving the trained LoRA adapter
- Reloading the model for inference

---

# 🚀 Features

- Fine-tuned **Qwen2.5-3B-Instruct**
- Medical domain adaptation
- QLoRA fine-tuning
- 4-bit NF4 quantization using BitsAndBytes
- LoRA adapters with PEFT
- Supervised Fine-Tuning (SFT)
- Memory-efficient training
- Google Colab compatible
- Interactive chatbot inference

---

# 🧠 Base Model

**Model**

```
Qwen/Qwen2.5-3B-Instruct
```

Qwen2.5-3B-Instruct is an instruction-tuned Large Language Model designed to understand user instructions and generate coherent conversational responses.

Rather than training a model from scratch, this project adapts the pretrained model to the medical domain through efficient fine-tuning.

---

# 📚 Dataset

**Dataset**

```
lavita/ChatDoctor-HealthCareMagic-100k
```

The dataset contains thousands of medical conversations between patients and healthcare professionals.

It includes examples covering topics such as:

- Disease symptoms
- Medical diagnoses
- Treatment recommendations
- Medication information
- Healthcare advice
- Doctor–patient conversations

For faster experimentation and to fit within Google Colab's GPU limitations, a subset of the dataset is used during training. The same pipeline can be applied to the full dataset.

Before training, each example is converted into the **Qwen chat format** using the tokenizer's chat template.

---

# 🏗️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL
- PEFT
- BitsAndBytes
- QLoRA
- Google Colab

---

# ⚙️ Fine-Tuning Pipeline

The notebook follows these steps:

1. Install the required libraries.
2. Verify GPU availability.
3. Load the pretrained Qwen model.
4. Load the ChatDoctor dataset.
5. Select a subset of the dataset.
6. Format each conversation using the Qwen chat template.
7. Quantize the model to 4-bit precision using BitsAndBytes.
8. Configure LoRA adapters.
9. Fine-tune the model using TRL's SFTTrainer.
10. Save the LoRA adapter and tokenizer.
11. Reload the fine-tuned model.
12. Generate answers to medical questions.

---

# ⚡ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-3B-Instruct |
| Dataset | lavita/ChatDoctor-HealthCareMagic-100k |
| Fine-Tuning Method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 2 |
| Batch Size | 1 |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 8 |
| Learning Rate | 2e-4 |
| Maximum Sequence Length | 256 |
| Mixed Precision | FP16 |
| Packing | Disabled |
| Checkpoint Saving | Every Epoch |
| Training Framework | TRL SFTTrainer |

---

# 💬 Example

### User

```
What are the symptoms of diabetes?
```

### Chatbot

```
Hello and welcome to Chat Doctor.

Diabetes is a chronic disease characterized by high blood sugar levels.
---

# 📊 Results

After fine-tuning, the model demonstrates improved performance in medical conversations by:

- Producing responses in a doctor–patient conversational style.
- Providing more detailed healthcare-related explanations.
- Generating medically focused answers while preserving the general language capabilities of the original Qwen model.
- Responding more naturally to medical questions compared to the base model.

---

# 📈 Future Improvements

- Fine-tune on the complete ChatDoctor dataset.
- Train using additional medical instruction datasets.
- Integrate Retrieval-Augmented Generation (RAG) with trusted medical resources.
- Build a web interface using Gradio or Streamlit.
- Deploy the model using FastAPI.
- Support multilingual medical conversations.
- Evaluate the chatbot using medical benchmark datasets.
