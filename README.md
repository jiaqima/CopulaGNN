# CopulaGNN

This repo provides a PyTorch implementation for the CopulaGNN models as described in the following paper:

[CopulaGNN: Towards Integrating Representational and Correlational Roles of Graphs in Graph Neural Networks](https://arxiv.org/abs/2010.02089)

Jiaqi Ma, Bo Chang, Xuefei Zhang, and Qiaozhu Mei. ICLR 2021.

## Requirements
Most dependency packages are included in `environment.yml`. Run `conda torch_env create -f environment.yml` to install the required packages.

In addition, one also needs to install [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) following the [official installation instructions](https://github.com/rusty1s/pytorch_geometric#installation). 

The code is tested with the following PyTorch-Geometric version.

```
torch-scatter==2.0.5
torch-sparse==0.6.7
torch-cluster==1.5.7
torch-geometric==1.6.1
```

## Run the code
Example: `python main.py --lr 0.001 --hidden_size 16 --dataset wiki-squirrel --model_type regcgcn`.

## Cite
```
@article{ma2020copulagnn,
  title={CopulaGNN: Towards Integrating Representational and Correlational Roles of Graphs in Graph Neural Networks},
  author={Ma, Jiaqi and Chang, Bo and Zhang, Xuefei and Mei, Qiaozhu},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```