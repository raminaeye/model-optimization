# Model Optimization

Toy example notebooks on applying model optimization techniques to computer vision and LLM models: quantization, knowledge distillation, and pruning.

## Repo structure

- `quantization/quantization.ipynb` - dynamic quantization, static post-training quantization (PTQ), and quantization-aware training (QAT) on a small MNIST CNN, plus static quantization of a ResNet18 on STL10.
- `quantization/resnet.py` - ResNet definitions adapted for static quantization (named ReLU modules for layer fusion, `FloatFunctional` skip connections).
- `knowledge_distillation/knowledge_distillation.ipynb` - distilling a fine-tuned ResNet18 teacher into small student CNNs on STL10, with both logit-based and feature-based distillation.
- `pruning/pruning.ipynb` - unstructured L1 pruning of a ResNet18 on STL10, accuracy measurement before and after, and fine-tuning to recover performance.

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Open any notebook and run the cells in order.

The notebooks download datasets (MNIST, STL10) from torchvision on first run and train small models, so they need an internet connection the first time. Training is CPU-friendly but faster on a GPU. Some cells reference trained checkpoints under `data/` (gitignored); run the training cells in order to produce them.

## What you learn

- Dynamic vs static post-training quantization vs quantization-aware training, and when to use each.
- Layer fusion, observers, and calibration for static quantization.
- Logit-based and feature-based knowledge distillation with temperature-scaled KL divergence.
- Unstructured pruning with masks, measuring sparsity, and fine-tuning to recover accuracy.
