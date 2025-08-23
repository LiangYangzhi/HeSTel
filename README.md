# HeSTel

## Introduction

HeSTel is a graph learning-based heterogeneous spatio-temporal entity linking framework that accounts for temporal and spatial heterogeneity in heterogeneous trajectories to enable associate the same moving objects across spatiotemporal trajectories captured by diverse sensors.

## Overall Framework

Here is the overall framework of **HeSTel**, which consists of the following six main modules: 

- **Data Module**: Responsible for loading datasets. Corresponding package name: **dataset**.
- **Preprocessing Module**: Handles data preprocessing operations. Corresponding package name: **preprocessing**.
- **Model Module**: Initializes baseline models and custom models. Corresponding package name: **model**.
- **Evaluation Module**: Evaluates model predictions using multiple metrics and performs visualization. Corresponding package name: **evaluator**.
- **Execution Module**: Manages model training and prediction. Corresponding package name: **executor**.
- **Infrastructure Module**: Includes: Managing framework parameter configurations (package name: **config**). Handling model files and runtime logs (package name: **logs**). Providing general utility functions (package name: **utils**).

This modular design ensures a clear structure for development, testing, and deployment.

## Requirements

Python >=3.8
PyTorch>=2.0.0
numpy==1.22.4
pandas==1.3.5
torch==1.12.1
tqdm==4.66.1
geopy==2.3.0
torch_geometric==2.3.1
faiss-gpu==1.7.2
scikit-learn==1.3.0

## Dataset

The Automatic Identification System (AIS) dataset and T-Drive dataset are open-source datasets.
We provide heterogeneous trajectory data from both datasets.

## Quick-Start

Before run models in LibCity, please make sure you download at dataset and put it in directory.

Startup code:

```shell
python STEL.py
```

In the STEL.py file, program parameters are controlled by reading different configuration files.
