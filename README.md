# 🔋 2026 MCM Problem A: Adaptive Physics-Based Battery Simulation
> A high-fidelity, continuous-time Lithium-Ion battery model with adaptive state & parameter estimation.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![MCM](https://img.shields.io/badge/MCM-2026-orange)

## 📖 Introduction (项目介绍)

This repository contains the solution code for **2026 MCM Problem A**. 
We developed a **hybrid model** combining physical kinetics with data-driven adaptive control to simulate smartphone battery depletion under various loads.

The core framework integrates:
1.  **KiBaM (Kinetic Battery Model):** Describes the charge recovery effect and capacity rate dependence.
2.  **Shepherd Model:** Simulates the voltage-discharge curve.
3.  **Adaptive BMS (Battery Management System):** An online observer that corrects SOC drift and estimates SOH (Internal Resistance) in real-time.

本项目建立了基于物理机理的锂离子电池连续时间模型。通过结合 KiBaM 动力学模型与自适应 BMS 算法，实现了对电池 SOC（荷电状态）和 SOH（健康状态/内阻）的高精度跟踪与预测，解决了传统模型在老化阶段拟合误差大的问题。

## 🌟 Key Features (核心特性)

* **🧪 White-Box Physics (白盒物理机理):** * Fully interpretable differential equations for charge diffusion ($y_1, y_2$).
    * Thermodynamic modeling (Entropic heat & Arrhenius aging).
* **🧠 Dual-Loop Adaptation (双闭环自适应):** * **Fast Loop:** Real-time SOC correction based on voltage innovation.
    * **Slow Loop:** Online parameter estimation for Internal Resistance ($R_{base}$) using Gradient Descent.
* **📊 Data-Driven Validation:** * Calibrated using NASA Battery Prognostics Dataset.
    * Achieved low RMSE across 70+ charge/discharge cycles.

## 📂 Project Structure (文件结构)

```text
.
├── BatteryV3.py        # 🧱 Physics Engine: KiBaM, Shepherd, Thermal models
├── AdaptiveBMS.py      # 🧠 The Brain: SOC Observer & Resistance Estimator
├── main.ipynb          # 🔬 Experiment: Simulation loop & Visualization
├── get_data.py         # 🧹 Data Loader: Preprocessing NASA datasets
├── data/               # (Optional) Place dataset files here
└── README.md           # This file