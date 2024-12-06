# KKT-hPINN: Physics-Informed Neural Networks with Hard Linear Equality Constraints for Surrogate Modeling
Welcome to the KKT-hPINN paper page. This code belongs to the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0098135424001820) published in Computers & Chemical Engineering.

If you found this work useful, please cite this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0098135424001820) ❤️:
```bibtex
@article{CHEN2024108764,
title = {Physics-informed neural networks with hard linear equality constraints},
journal = {Computers & Chemical Engineering},
volume = {189},
pages = {108764},
year = {2024},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2024.108764},
url = {https://www.sciencedirect.com/science/article/pii/S0098135424001820},
author = {Hao Chen and Gonzalo E. Constante Flores and Can Li},
keywords = {Surrogate modeling, Physics-informed neural network, Artificial intelligence}
}
```

# Overview
In this work, we develop a novel **hard-constrained PINN architecture, KKT-hPINN, that always satisfies hard linear equality constraints in both training and testing processes**. The architecture consists of a traditional neural network and two non-trainable projection layers that orthogonally project model predictions onto a feasible region of predefined linear equality constraints.

In all case studies, KKT-hPINN demonstrates enhanced predictive capability as surrogate models, while strictly satisfying hard linear equality constraints without introducing additional hyperparameters and computational cost.

# Installation
```pip install -r requirements.txt```

# Quick Start
To run the KKT-hPINN and compare it with non-constrained NN, soft-constrained PINN and equality completion NN, 

```python main.py --model_id MODELID --dataset_type plant --dataset_path benchmark_plant.csv --job experiment```
