# Robust Deep Signed Graph Clustering via Weak Balance Theory
This repository contains the official implementation of DSGC as described in this paper sumitted to WWW-2025.

**Overview of DSGC:** **D**eep **S**igned **G**raph **C**lustering

![image](https://github.com/yaoyaohuanghuang/DSGC/blob/main/IMG/framework_www.jpg)

# Installation
We have tested our code on Python 3.6.13 with PyTorch 1.8.0, PyG 1.8.1 and CUDA 12.1. Please follow the following steps to create a virtual environment and install the required packages.

Clone the repository:
```bash
git clone xxx
cd DSGC
```

Create a virtual environment:
```bash
conda create --name dsgc python=3.6.13 -y
conda activate dsgc
```

Install dependencies:
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric==1.6.3
pip install -r requirements.txt
```
## Reproduce Results
We provide the source code to reproduce the results in our paper. The results of DSGC can be reproduced by running main.py. 

To train sythetic datasets:
```bash
python main.py --N [node_number] --p [edge_probability] --K [cluster_number] --eta [flip_probability]
```
To train real-world datasets:
```bash
python main.py --dataset [dataset_name]
```
Simple test on the signed stochastic block model SSBM(N=1000, p=0.01, K=5, eta=0.01). Dataset_name can be choosen from sp1500 and rainfall.
