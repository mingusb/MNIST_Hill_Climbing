# 🧪 MNIST Hill Climbing

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge" alt="PyTorch">
</p>

<p align="center"><b>Implementing and evaluating hill climbing optimization on the MNIST dataset.</b></p>

---

## 📑 Index

- [Overview](#-overview)

---

## 🚀 Overview

The first implementation of hill climbing on MNIST was done in 2017 using the SGDClassifier in scikit-learn. This was followed up with an implementation in PyTorch in 2022. In 2025, a ResNeXt gating network was added to route MNIST samples to N hill climbing experts.

The hill climbing method adds uniform noise to the weights and then runs inference on the entirety of MNIST. If the performance improved the weight increase is kept, otherwise it is disregarded and a new sample is tried. In the plot shown here accuracy of 89.5% is achieved during which time inference was run on the entirety of MNIST 10 million times.

![alt text](mingushc.png)

In `mingus_hc_resnext.py` a ResNeXt network is taught to gate MNIST samples to N trained hill climbing models (with a separate mingushc implementation in `train_expert.py`) in order to test the hypothesis that the models learned with hill climbing can serve as experts. This prototype code was written by GPT-5 and Gemini Pro and it uses a sophisticated cross-validation scheme. The current best performance by this model with 37 hill climbing experts is 92.6%.

---
