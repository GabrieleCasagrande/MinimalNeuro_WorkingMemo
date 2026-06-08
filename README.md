# eQIF Dynamics, Hysteresis, Bifurcation Analysis and Network Implementation

This repository provides a computational framework for simulating and analyzing **self-excitatory Quadratic Integrate-and-Fire (eQIF)** neurons. It includes tools for large-scale Spiking Neural Network (SNN) simulations, mean-field theoretical comparisons, and the exploration of neural robustness through phase-plane and bifurcation analysis.
Additionally, the repository also porvide the script to simulate self-excitatory Leaky Integrate-and-Fire (eLIF), which is also shown in the manuscript.

## 🚀 Overview

This repository provides a computational framework for simulating and analyzing an extension of the adaptive-QIF in the self-excitatory model case (that we term eQIF), presented in **"A minimal model of working memory in neural systems and neuromorphic circuits"**. 
It includes tools for large-scale Spiking Neural Network (SNN) simulations, mean-field theoretical comparisons, and the exploration of single-neuron bistability and hysteresis.
It allows for the exploration of:
1. **Dynamical Systems Theory**: Visualizing the mathematical foundations of the model through phase-plane trajectories and bifurcation diagrams.
2. **Robustness and Perturbation**: Analyzing how the network responds to distractor stimuli.
3. **Network-Level Analysis**: Comparing SNN simulations (`Brian2`) with adjusted mean-field models.

---

## 🛠️ Key Features

### 1. Numerical Simulation and  Dynamical Analysis
* **Model Simulation**: Visualization of the time evolutionof the single-neuron model, subjected to an external current.
* **Phase-Plane Trajectories**: Visualization of nullclines and system trajectories in the $(V, w)$ plane to understand the stability of neural states.
* **Hysteresis & Bifurcation Mapping**: Analysis of "Up" vs. "Down" frequency responses to identify bistable regions.
* **Distractor Robustness**: Evaluation of network stability and response when subjected to external distractor signals.

* ### 2. Network Simulation & Mean-Field Comparison
* **Spiking Neural Network**: Numerical implementation of a network of eQIF neurons with quenched heterogeneity.
* **Synchrony & Variability**: Calculation of the time-resolved synchrony measure $\chi(t)$ and Coefficient of Variation (CV) for firing intervals.
* **Validation**: Comparative plotting of population firing rates between SNN and derived mean-field equations.

---

## 📂 Project Structure

### Notebooks
* `network_analysis.ipynb`: Core pipeline for SNN vs. Mean-Field comparison.
* `eQIF_hysteresis.ipynb`: Exploration of single-cell dynamics and bistability.
* `eQIF_perturbation.ipynb`: Analysis of the single_neuron response and robustness to external distractors.
* `eQIF_3d_bifurcation.ipynb`: Bifurcation analysis of eQIF model with respect to the external current.


### Python Scripts
* `eQIF_PhasePlane.py`: Generates phase-space visualizations ($V$ vs $w$) including nullclines of the eQIF model.
* `eQIF_utils.py`: Utility functions for simulation and analysis of the single-neuron eQIF model.
* `eQIF.py`: Simulation of the eQIF model with a time-variable external current. 
* `eLIF.py`: Simulation of the eLIF model with a time-variable external current.
* `Izhikevich.py`: Simulation of the Izhikevic model with a time-variable external current.
* `AdEx.py`: Simulation of the AdEx model with a time-variable external current.

---

## ⚙️ Installation & Requirements

Ensure you have a Python environment (3.8+) with the following dependencies:

```bash
pip install numpy matplotlib scipy brian2
```

---

## 📈 Usage Information

### Exploring eQIF Single-Neuron Model and Hysteresis
Run `eQIF.py` to simulate and plot th time evolution of the eQIF model with a time varying external current.
Run `eQIF_hysteresis.ipynb` to explore single-neuron dynamics and bistability. With that notebook you can:
1. Generate heatmap to show the amplitude and frequency response in the hysteresis region for different values of the parameters $a$ and $I_{ext}$ (Fig2c and Fig9b)
2. Generate heatmap to show the amplitude and frequency response in the hysteresis region for different values of the parameters $b$ and $I_{ext}$ (Fig9a)
3. Generate hysteresis loop by simulating the neuron with different initial conditions ($w_0, v_0$) (Fig2b)

### Visualizing State Stability & Mapping Transitions
Use `eQIF_PhasePlane.py` to observe how the neuron's state evolves relative to its nullclines. This is essential for understanding why certain parameters lead to sustained oscillations versus steady states (Fig3).
Use `eQIF_3d_bifurcation.ipynb` to perform the bifurcation analysis of eQIF model with respect to the specific current values ($I_{ext}$). Also, it allows to represent both in 3d and 2d where the system undergoes a qualitative change in behavior (Fig4).

### Testing Network Robustness
Run `eQIF_perturbation.ipynb` to simulate the single-neuron model to current perturbations. This demonstrates the stability of the working memory capacity of the single-neuron eQIF model. With that notebook you can:
1. Plot the response of the firing rate to the stability of either the persistent and quiescent state in the hysteresis region to external perturbations. Perturbation can be varied in amplitude, duration and strenght of the noise (Fig11)
2. Generate heatmap to show the stability of both the persistent and quiescent states to a perturbation for different combination of amplitude and duration (Fig10)
3. Generate a plot which highlight the effect of the introduction of white noise to the external perturbation on the stability of the persistent and quiescent states.

### Running the Network Analysis
Run `network_analysis.ipynb` to simulate the activity of a network of eQIF neurons and its mean-field reduction (Fig8). With that notebook you can:
1. Define the network of all-to-all coupled eQIF neurons and the respective mean-field reduction and run simulation of both.
2. Compute distinct measure which characterized the behavior of the dpiking neural network.
3. Generate comparison plots showing the behavior of the spiking neural network and its mean-field reduction (Fig8).

### Extra
Run `eLIF.py` to simulate and plot th time evolution of the eLIF model with a time varying external current. By running this script is possible to show the appearence of a bistable regime in the self-excitatory Leaky Integrate-and-Fire.
The same can be done with `Izhikevich.py` and `AdEx.py` for the respective models.
