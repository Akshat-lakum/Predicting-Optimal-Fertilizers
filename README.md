# 🧪 Fertilizer Recommendation using Tabular ML & DL Models

### 🎯 Kaggle Playground Series - Season 5 Episode 6 (June 2025)

> **Objective**: Predict the top 3 most suitable fertilizers for given weather, soil, and crop conditions using tabular data.

---

## 📂 Repository Structure

This repository contains a comprehensive notebook (`fertilizer_prediction.ipynb`) implementing and experimenting with **over 40+ machine learning and deep learning pipelines**. It includes both traditional models and advanced ensemble/meta-learning techniques.

---

## 🔍 Problem Overview

The dataset is synthetic and mimics real-world fertilizer recommendation systems. The goal is to predict the **top-3 fertilizers** out of multiple possible classes for each input condition using **MAP@3 (Mean Average Precision @3)** as the evaluation metric.

---

## 📊 Models & Techniques Used

### ✅ Traditional Models
- LightGBM (basic and tuned versions)
- CatBoost
- XGBoost
- Logistic Regression (One-Hot Encoding + Feature Cross)
- KMeans Cluster Feature Engineering
- Class Balancing (sample weighting)

### ✅ Ensemble & Meta-Learning
- Bagging and Boosting with LightGBM/CatBoost
- Stacking (LGBM + CatBoost + XGBoost + NN Meta)
- Weighted Ensemble with Optuna-Tuned Weights
- Pseudo-Labeled Stacking
- Dynamic Cluster-Based Blending
- Base + Meta Neural Net Stacker

### ✅ Deep Learning & Neural Architectures
- Wide & Deep using TensorFlow/Keras
- TabPFN (Transformer-based)
- Hybrid TabNet + LightGBM Ensemble
- PyTorch-Based Meta Model

### ✅ Special Preprocessing & Features
- Quantile Transformer + Smart Feature Crosses
- Mutual Information-based Feature Interactions
- Augmented Data using External CSV
- Class Rebalancing with Sample Weights
- Original Dataset Incorporation with Fractional Weighting

---

## 📁 Dataset

- `train.csv`: Training data with features and fertilizer labels.
- `test.csv`: Test data for inference.
- `Fertilizer Prediction.csv`: Augmented original dataset used for pseudo-labeling and logistic regression enrichment.

---

## 📈 Evaluation Metric

**MAP@3 (Mean Average Precision at 3)**  
This metric measures if the correct fertilizer label appears in the top-3 predictions and rewards correct early rankings.

---

## 📌 Results Summary

- **Best LB Score**: `0.358`  
- Goal: Improve to `0.37+` via advanced ensemble and augmentation strategies

Strategies tested so far:
- Logistic Regression + Feature Cross + One-Hot Encoding
- Class Rebalancing + Original Dataset Augmentation
- Deep Neural Nets (Wide & Deep, TabNet, TabPFN)
- Optuna-Tuned Weighted Ensemble Blends

---

## 🚀 How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fertilizer-prediction.git
   cd fertilizer-prediction
