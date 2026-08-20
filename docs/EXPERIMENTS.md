# Reproduction Guide

This document describes how to reproduce every result reported in the thesis.

## Pipeline Overview

```
Raw datasets (NSL-KDD + UNSW-NB15)
        |
        v
[1] Preprocessing        (data_preprocessing.py)
        |
        v
[2] CNN-LSTM training    (improved_cnn_lstm_classifier.py + improved_main_workflow.py)
        |
        v
[3] XAI attribution      (xai_visualization.py)
        |
        v
[4] LLM explanation      (llm_explanation_generator_gguf.py)
        |
        v
[5] Metrics evaluation   (evaluation_metrics.py)
```

## Stage 1: Preprocessing

```bash
python data_preprocessing.py
```

Loads datasets, constructs the 37-feature canonical schema, applies Z-score normalization, one-hot encoding, SMOTE oversampling (0.6 ratio). Runtime: ~5 minutes.

## Stage 2: CNN-LSTM Training

```bash
python improved_main_workflow.py
```

Trains the 522,144-parameter CNN-LSTM with the refined recipe (Adam lr=1e-3, class-weighted BCE, label smoothing ε=0.05, 50 epochs, batch=256). Runtime: ~15 minutes on RTX 4070 Laptop.

## Stage 3: XAI Attribution

```bash
python xai_visualization.py
```

Generates SHAP beeswarm, SHAP interaction heatmap, baseline comparison, cost-uncertainty tradeoff, intersection ablation, and architecture diagrams in `visualizations/`. Runtime: ~5 minutes.

## Stage 4: LLM Explanation

```bash
python llm_explanation_generator_gguf.py
```

Runs Mistral-7B-Instruct (Q4_K_M, ~4.5 GB) on top SHAP/LIME attributions, producing 4 stakeholder-tailored explanations per packet. Runtime: ~10 minutes for 20 packets on RTX 4070 Laptop.

## Stage 5: Metrics

```bash
python evaluation_metrics.py
```

Computes accuracy/F1 with 95% bootstrap CIs, per-dataset breakdown, baseline comparison, cost-effectiveness ratios, XAI fidelity metrics, SHAP-LIME top-k agreement, and LLM technical metrics. Runtime: ~2 minutes.

## Key Hyperparameters

| Parameter | Value |
|---|---|
| CNN filters | 128 |
| LSTM hidden size | 128 |
| Attention heads | 8 |
| Sequence length | 10 packets |
| Train/test split | 70/15/15 |
| SMOTE ratio | 0.6 |
| Positive class weight | 2.5 |
| Learning rate | 1e-3 |
| Epochs | 50 |
| SHAP samples | 100 (stratified) |
| LIME samples | 5,000 |
| LLM temperature | 0.0 (deterministic) |
| LLM max tokens | 200 |
| LLM top-p | 0.9 |

## Expected Outputs

```
visualizations/
  baseline_comparison.png
  cost_uncertainty_tradeoff.png
  fig_3_1_architecture.png
  intersection_ablation.png
  shap_beeswarm_plot.png
  shap_interaction_heatmap.png
  temporal_attention_weights.png

llm_explanations.json     # Role-tailored narratives
metrics_results.json       # All numerical results
trained_model.pth         # Trained CNN-LSTM
```

## Validation

To verify your reproduction matches the thesis:

1. **Accuracy on combined test set should be 96.76% ± 0.4%** (95% CI)
2. **CNN-LSTM loses to RF/XGBoost by ~2%** on raw accuracy (expected — contribution is XAI+LLM, not accuracy)
3. **SHAP top-1 feature should be `srv_diff_host_rate`** on NSL-KDD
4. **LLM should produce 100% valid JSON within 200 tokens**
5. **Pipeline runs in ~30 minutes total** on RTX 4070 Laptop