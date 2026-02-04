# LLaMA-Factory Fine-Tuning Guide for Qwen-VL

This document summarizes my learning and exploration of the **LLaMA-Factory** framework and the fine-tuning workflow for **Qwen-VL multimodal models**.

---

## 1. What is Model Fine-Tuning?

Fine-tuning is the process of continuing training from a pretrained model on task-specific data.  
It allows a general model to adapt to new domains or tasks with fewer resources than training from scratch.

### Typical Fine-Tuning Steps

1. **Data Preparation** – Collect and format task-related data  
2. **Model Selection** – Choose a suitable pretrained model  
3. **Transfer Learning** – Continue training on new data  
4. **Hyperparameter Tuning** – Adjust learning rate, batch size, etc.  
5. **Evaluation** – Test performance and iterate  

### Advantages of Fine-Tuning

- More resource-efficient than training from scratch  
- Faster deployment for new tasks  
- Significant performance gains in specific domains  
- Stronger domain adaptation ability  

---

## 2. Introduction to LLaMA-Factory

LLaMA-Factory is an open-source framework designed to simplify the customization and fine-tuning of large language models.

### Core Features

- Supports multiple mainstream models (LLaMA, Mistral, Qwen, etc.)
- Multiple training strategies (Full fine-tuning, LoRA, DPO, PPO)
- User-friendly WebUI interface
- Built-in monitoring tools such as TensorBoard

### Key Characteristics

- Easy to use via graphical interface
- Efficient fine-tuning methods (SFT, DPO, ORPO)
- Flexible hyperparameter configuration
- Multi-language UI support

---

## 3. Environment Installation

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n llamafactory python=3.11 -y
conda activate llamafactory
pip install -e ".[torch,metrics]"

### Step 1: Launch WebUI

After running `llamafactory-cli webui`, the training interface can be accessed in the browser.

![WebUI Home](images/webui_home.png)

### Step 2: Model Selection

Selecting the base model (e.g., Qwen-VL) inside the WebUI.

### Step 3: Dataset Configuration

Loading instruction-style dataset for training.

![Dataset Config](images/dataset_config.png)

### Step 4: Training Parameters

Setting hyperparameters such as learning rate, epochs, and batch size.

![Training Config](images/training_config.png)
