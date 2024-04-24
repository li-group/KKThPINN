# KKT-hPINN: Physics-Informed Neural Networks with Hard Linear Equality Constraints for Surrogate Modeling
Welcome to the KKT-hPINN paper page. This code belongs to a paper currently under review, and the preprint can be found at arxiv: [https://arxiv.org/abs/2402.07251](https://arxiv.org/abs/2402.07251)

If you found this work useful, please cite this [preprint](https://arxiv.org/abs/2402.07251) ❤️:
```bibtex
@misc{chen2024physicsinformed,
      title={Physics-Informed Neural Networks with Hard Linear Equality Constraints}, 
      author={Hao Chen and Gonzalo E. Constante Flores and Can Li},
      year={2024},
      eprint={2402.07251},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Overview
In this work, we develop a novel **hard-constrained PINN architecture, KKT-hPINN, that always satisfies hard linear equality constraints in both training and testing processes**. The architecture consists of a traditional neural network and two non-trainable projection layers that orthogonally project model predictions onto a feasible region of predefined linear equality constraints.

In all case studies, KKT-hPINN demonstrates enhanced predictive capability as surrogate models, while strictly satisfying hard linear equality constraints without introducing additional hyperparameters and computational cost.

# Installation
```pip install -r requirements.txt```

# Test it on your device
To run the KKT-hPINN and compare it with non-constrained NN and soft-constrained PINN

```python main.py --model_id MODELID --dataset_type plant --dataset_path benchmark_plant.csv --job experiment```

For other setups: --dataset_type can be either cstr, plant or distillation; --dataset_path can be either benchmark_CSTR.csv, benchmark_plant.csv or benchmark_distillation.csv; Adjust the hyperparameters correspondingly.

```mkdir data models```
```mkdir data/learning_curves data/tables```
Create their sub-folders for the specific dataset, model, val_ratio in the following order if they do not exist.
```mkdir data/learning_curves/dataset/model/val_ratio```
```mkdir data/tables/dataset/model/val_ratio```
```mkdir models/dataset/model/val_ratio```

