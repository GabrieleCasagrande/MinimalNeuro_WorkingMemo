# eQIF Network Dynamics, Hysteresis, and Bifurcation Analysis

This repository provides a computational framework for simulating and analyzing **exponential Quadratic Integrate-and-Fire (eQIF)** neurons. It includes tools for large-scale Spiking Neural Network (SNN) simulations, mean-field theoretical comparisons, and the exploration of neural robustness through phase-plane and bifurcation analysis.

## 🚀 Overview

The project is structured to bridge the gap between individual neuron dynamics and collective network behavior. It allows for the exploration of:
1. **Network-Level Analysis**: Comparing SNN simulations (`Brian2`) with adjusted mean-field models.
2. **Robustness and Perturbation**: Analyzing how the network responds to distractor stimuli.
3. **Dynamical Systems Theory**: Visualizing the mathematical foundations of the model through phase-plane trajectories and bifurcation diagrams.

---

## 🛠️ Key Features

### 1. Network Simulation & Mean-Field Comparison
* **eQIF Model**: Implementation of the exponential Quadratic Integrate-and-Fire model with quenched heterogeneity.
* **Synchrony & Variability**: Calculation of the time-resolved synchrony measure $\chi(t)$ and Coefficient of Variation (CV) for firing intervals.
* **Validation**: Comparative plotting of population firing rates between SNN and derived mean-field equations.

### 2. Advanced Dynamical Analysis
* **Bifurcation Studies**: Tools to generate bifurcation diagrams that illustrate transitions between resting and spiking states.
* **Phase-Plane Trajectories**: Visualization of nullclines and system trajectories in the $(V, w)$ plane to understand the stability of neural states.
* **Distractor Robustness**: Evaluation of network stability and response when subjected to external distractor signals.

### 3. Hysteresis & Bifurcation Mapping
* **Hysteresis Loops**: Analysis of "Up" vs. "Down" frequency responses to identify bistable regions where the neuron's state depends on its history.

---

## 📂 Project Structure

### 📓 Notebooks
* `network_analysis.ipynb`: Core pipeline for SNN vs. Mean-Field comparison.
* `single_neuron_hysteresis.ipynb`: Exploration of single-cell dynamics and bistability.
* `distractor_analysis.ipynb`: Analysis of network response and robustness to external distractors.

### 🐍 Python Scripts
* `PhasePlane_trajectories.py`: Generates phase-space visualizations ($V$ vs $w$) including nullclines.
* `BifFig.py`: Scripts for plotting bifurcation diagrams to identify critical parameter transitions.
* `single_eQIF_utils.py`: Utility functions for simulation and rate computation.

---

## ⚙️ Installation & Requirements

Ensure you have a Python environment (3.8+) with the following dependencies:

```bash
pip install numpy matplotlib scipy brian2
