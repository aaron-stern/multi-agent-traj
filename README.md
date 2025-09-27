# Multi Agent Trajectory Inference

This repo is a work in progress experimenting with multi-agent trajectory inference using a toy
implementation of the multi-agent transformer architecture from Seff, et al (2023) [https://arxiv.org/abs/2309.16534].

## Approach

Using simulated trajectories of 2d rigid bodies, I train a transformer to predict trajectories. The training loop is defined in `models/train.py`.


## Installation

### Prerequisites
- Anaconda or Miniconda installed on your system
- Git (for cloning the repository)

### Setup Instructions

```bash
conda env create -f environment.yml
conda activate multi_agent_traj
```
