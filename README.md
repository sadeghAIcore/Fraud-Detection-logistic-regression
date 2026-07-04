# Credit Card Fraud Detection using Logistic Regression (From Scratch)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Linter: Flake8](https://img.shields.io/badge/code%20style-flake8-black.svg)](https://flake8.pycqa.org/en/latest/)

An end-to-end, production-grade machine learning project implementing **Logistic Regression from scratch** (using NumPy) to detect fraudulent credit card transactions. This project demonstrates modular software engineering principles, rigorous evaluation metrics for imbalanced datasets, and robust pipeline structures.

---

## 📌 Project Overview
Fraud detection is a classic highly-imbalanced classification problem. In this repository, we:
1. Implement **Logistic Regression** mathematically using only NumPy (including Gradient Descent, L2 Regularization, and custom loss tracking).
2. Establish a modular preprocessing and training pipeline.
3. Evaluate model performance using metrics suited for extreme class imbalance ($F_1$-Score, Precision-Recall AUC) rather than simple Accuracy.

---

## 📐 Mathematical Formulation

The core Logistic Regression model is built from scratch based on the following equations:

### 1. Hypothesis Function (Sigmoid)
The probability that a given transaction $x^{(i)}$ is fraudulent ($y=1$) is modeled by:
$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

### 2. Regularized Loss Function (Log-Loss with $L_2$ Regularization)
To prevent overfitting on the majority class, we minimize the binary cross-entropy loss with Ridge ($L_2$) regularization:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

### 3. Gradient Update Rule
The parameters $\theta$ are updated iteratively using batch gradient descent:
$$\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m} \theta_j \right] \quad (\text{for } j \ge 1)$$

---
