# About

Encoder tranformer model with optional token learning. 

Architecture is based off the paper AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (https://arxiv.org/abs/2010.11929)

Implemented using https://github.com/karpathy/nanoGPT as a base.

# Prerequisites

- Pytorch
- pip install transformers
- pip install datasets
- pip install wandb

### How to train the model
- Run python train.py config/{config name}.py
- Model parameters can be overwritten in the cmd by typing python train.py config/{config name}.py --{parameter}={value}

