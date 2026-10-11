# STG-ProtoNet
STG‑ProtoNet, a multi‑scale spatial‑temporal graph prototype network for EEG‑based VR motion‑sickness recognition.
<p>
<img src="https://img.shields.io/badge/python-3.8+-blue?style=flat-square">
<img src="https://img.shields.io/badge/pytorch-1.8+-red?style=flat-square">
<img src="https://img.shields.io/badge/pytorch_geometric-2.0+-green?style=flat-square">
</p>
<h2>📢 News </h2>

- 🚀 Model Optimization: We optimize the inference pipeline of STG-ProtoNet, integrating prototype learning and MCTS for interpretable neural subgraph mining.
- 🌍 Cross-subject Generalization: STG-ProtoNet achieves competitive performance on both intra-subject and cross-subject VRMS EEG classification tasks.

This repository contains the implementation of our paper "STG-ProtoNet: Multi-Scale Spatial-Temporal Graph Prototype Network for Interpretable EEG Analysis of VR Motion Sickness", introducing a novel spatio-temporal graph prototype network to decode EEG signals and reveal brain network plasticity during VR motion sickness adaptation.

## Key Features
- Dual-branch encoding to capture temporal dynamics and spatial topology of EEG brain networks.
- Combine prototype learning and Monte-Carlo Tree Search to extract discriminative interpretable neural subgraphs.
- Superior classification accuracy and stability for intra-subject and cross-subject VRMS recognition.

## Installation </h2>
## Prerequisites
- Python 3.8+
- PyTorch 2.0+
- torch-geometric==1.8+ (for graph neural network)
- CUDA 11.1+ (for GPU acceleration)
- scikit-learn, numpy, scipy (for EEG signal processing)
## Dagtaset Prepatation
Download our dataset from Baidu Drive: https://pan.baidu.com/s/1SMBTGtB-1mS8QMvD7gjPoA  (extraction code: EEGD)
## Quick Start
1. Clone the repository:
```bash
git clone https://github.com/xxx/STG-ProtoNet.git
cd STG-ProtoNet
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Prepare EEG dataset and run training:
```
python train.py
```


