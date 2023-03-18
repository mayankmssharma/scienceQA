# scienceQA
Welcome to our Git repository for building and training PyTorch models on the ScienceQA dataset! This repository contains the codebase and approaches that we used to train and evaluate PyTorch models for answering scientific questions. The ScienceQA dataset is a challenging benchmark for machine learning models, as it contains questions from various scientific domains and requires both general knowledge and specific domain expertise to answer accurately. In this repository, we provide a detailed description of our approach, along with the codebase and trained models, to enable other researchers to reproduce and build upon our work. We hope that this repository will be a valuable resource for the machine learning community and will contribute to the development of more accurate and robust models for scientific question answering.

> **_NOTE:_** This repository currently doesn't contain the data files required to train our model. Please download the dataset from [here](https://scienceqa.github.io/#download). And store them under ```data/scienceqa``` directory for ease of code execution.

## Run
A sample command to run the baseline model with distilroberta-base as pre-trained framework:
```
python train_scienceqa.py --name distilroberta-base --epochs 20
```
