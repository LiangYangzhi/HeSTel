# HeSTel

## Introduction

HeSTel is a graph learning-based heterogeneous spatio-temporal entity linking framework that accounts for temporal and spatial heterogeneity in heterogeneous trajectories to enable associate the same moving objects across spatiotemporal trajectories captured by diverse sensors. This repository of HeSTel aims to facilitate research in the spatiotemporal entity linking by providing standardized data formats and easy-to-use tools for model evaluation and comparison.

## Overall Framework
<img width="3335" height="1570" alt="repository of HeSTel" src="https://github.com/user-attachments/assets/5dfbcaca-31c1-4d02-90c3-3566e030bfc1" />
Here is the overall framework of **HeSTel**, which consists of the following six main modules: 

- **Data Module**: Responsible for loading datasets. Corresponding package name: **dataset**.
- **Preprocessing Module**: Handles data preprocessing operations. Corresponding package name: **preprocessing**.
- **Model Module**: Initializes baseline models and custom models. Corresponding package name: **model**.
- **Evaluation Module**: Evaluates model predictions using multiple metrics and performs visualization. Corresponding package name: **evaluator**.
- **Execution Module**: Manages model training and prediction. Corresponding package name: **executor**.
- **Infrastructure Module**: Includes: Managing framework parameter configurations (package name: **config**). Handling model files and runtime logs (package name: **logs**). Providing general utility functions (package name: **utils**).

This modular design ensures a clear structure for development, testing, and deployment.

## Feature

Unified Data Representation: All HeSTel datasets are represented in a unified format. This standardization allows for easy extension of new datasets into our benchmark.

Highly Efficient Pipeline: HeSTel leveraging its powerful tools and functionalities. Therefore, its code is concise. Specifically, we provide a small .py file and config folder to run all baselines in one click.

Comprehensive Benchmark and Analysis: We conduct extensive benchmark experiments and perform a comprehensive analysis of HeSTel  methods, delving deep into various aspects such as the impact of different models, and the influence of different domain datasets. The DOI of our HeSTel datasets is 10.57760/sciencedb.29423 .

## Our Experiments 

Please check the experimental results and analysis from our paper.

## Package Usage

### Requirements

You can quickly install the corresponding dependencies,

```
pip install -r requirements.txt
```

### Quick-Start

Before run models in HeSTel, please make sure you download at dataset and put it in directory.

Startup code:

```shell
python STEL.py
```

In the STEL.py file, program parameters are controlled by reading different configuration files.

### Acknowledgements

This work was supported by National Key Research and Development Program of China (2020YFA0712500), National Natural Science Foundation of China (No. 62006083), and Major Program of National Language Committee (WT145-39).

