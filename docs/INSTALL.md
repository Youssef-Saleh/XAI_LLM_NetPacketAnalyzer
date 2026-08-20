# Installation Guide

This document walks through the complete setup of the XAI LLM Network Packet Analysis framework.

## 1. System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.11 |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA 8 GB VRAM | NVIDIA 12 GB VRAM |
| Storage | 5 GB | 10 GB |

## 2. Clone the Repository

```bash
git clone https://github.com/Youssef-Saleh/XAI_LLM_NetPacketAnalyzer.git
cd XAI_LLM_NetPacketAnalyzer
```

## 3. Python Environment

```bash
conda create -n xai-llm python=3.11
conda activate xai-llm
pip install -r requirements.txt
```

## 4. Datasets

The datasets are **not** included in this repository due to their size.

### NSL-KDD (~30 MB)

1. Download from <https://www.unb.ca/cic/datasets/nsl.html>
2. Extract into `Data/NSL_KDD/`
3. Expected files: `KDDTrain+.txt`, `KDDTest+.txt`

### UNSW-NB15 (~850 MB)

1. Download from <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
2. Extract into `Data/UNSW_NB15/`
3. Expected files: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`, `NUSW-NB15_features.csv`

### Verify the layout

```bash
ls Data/NSL_KDD/KDDTrain+.txt Data/UNSW_NB15/UNSW_NB15_training-set.csv
```

## 5. Local LLM (Mistral-7B)

The framework uses a locally deployed Mistral-7B-Instruct model in GGUF format.

```bash
pip install llama-cpp-python
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --local-dir models/
```

Edit `llm_explanation_generator_gguf.py` and set `MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"`.

## 6. Verify the Setup

```bash
python -c "import torch, shap, lime, llama_cpp; print('OK')"
```

## 7. First Run

```bash
python improved_main_workflow.py
```

The first run takes ~30 minutes on an RTX 4070 Laptop.