# Multimodal-LLM-Playground
Hands-on experiments with multimodal large language models (Qwen-VL) including deployment and SFT fine-tuning
# Multimodal-LLM-Playground

Hands-on experiments with **Multimodal Large Language Models (MLLMs)**, focusing on **vision-language understanding**, **local deployment**, and **fine-tuning workflows**.

This repository documents my practical exploration of multimodal models such as **Qwen-VL**, including inference and supervised fine-tuning (SFT) using **LLaMA-Factory**.

---

## 📌 Project Goals

- Understand how vision and language are aligned in multimodal LLMs  
- Deploy a vision-language model locally and run image-text inference  
- Practice the full fine-tuning pipeline with instruction-style datasets  
- Build engineering intuition for multimodal AI systems  

---

## 🧠 Model: Qwen-VL

Qwen-VL is a vision-language model that combines a visual encoder with a large language model, enabling image + text input and natural language output.

In this project, I explored:

- Loading and running Qwen-VL locally  
- Performing multimodal inference  
- Understanding how visual features are integrated into LLM prompts  

---

## 🖼 Multimodal Inference Demo

The model takes an **image + text prompt** and generates a natural language response.

### Example

**Input**
- Image: `classroom.jpg`  
- Prompt: *"What is happening in this image?"*

**Output**
> "Students are attending a lecture in a university classroom."

### Inference Screenshot

![Inference Result](images/inference_demo.png)

---

## 💻 Inference Script

Main script: `inference_demo.py`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen-VL-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

query = tokenizer.from_list_format([
    {"image": "demo.jpg"},
    {"text": "Describe this image."}
])

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
