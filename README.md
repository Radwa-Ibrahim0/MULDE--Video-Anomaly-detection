# MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection

PyTorch implementation of **MULDE**, a density-based anomaly detection framework trained using score matching techniques. This method is detailed in the paper:

**Micorek, Jakub, et al. "MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection."** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. [Read the Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Micorek_MULDE_Multiscale_Log-Density_Estimation_via_Denoising_Score_Matching_for_Video_CVPR_2024_paper.html).

Origional Implementation by author can be found [here](https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection)

Following dataset have been used to train and Test the model
![PapersWithCode UCSD-Ped2](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mulde-multiscale-log-density-estimation-via/anomaly-detection-on-ucsd-ped2)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Visualizations](#visualizations)


## Overview

**MULDE** (Multiscale Log-Density Estimation) utilizes denoising score matching to perform effective anomaly detection in video data. By estimating the log-density across multiple scales, MULDE can identify anomalous events with high precision. This repository provides a PyTorch implementation of the MULDE framework, enabling researchers and practitioners to train and evaluate the model on various datasets.

## Installation

It is recommended to use a **Conda** environment to manage dependencies efficiently.

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/divyanshm21/MULDE--Video-Anomaly-detection.git
    cd mulde
    ```

2. **Create and Activate the Conda Environment:**

    ```bash
    conda env create -f environment.yml
    conda activate mulde
    ```

    *Note:* Ensure that [Conda](https://docs.conda.io/en/latest/miniconda.html) is installed on your system. If not, download and install it from the [official website](https://docs.conda.io/en/latest/miniconda.html).
    Miniconda will also work for Linux Users

## Usage

To execute the training and evaluation pipeline along with visualizations of densities and the dataset, use the following command:

```bash
python main.py --plot_dataset --gmm
```
## Visualizations
To visualize the training process and anomaly detection results, utilize TensorBoard:

```bash
tensorboard --logdir=runs/MULDE --samples_per_plugin images=100
```
Scalars Tab:

View anomaly scores for each individual scale under roc_auc_*_individual -> Micro. \
Aggregate metrics such as max, median, mean, and the negative log-likelihood scores of the GMM are available under roc_auc_*_aggregate->Macro.\
The roc_auc_best section highlights the best-performing individual scales or aggregates
