# AFECL
Edge Contrastive Learning: An Augmentation-Free Graph Contrastive Learning Model

## Requirements

This code package was developed and tested with Python 3.9


The required dependencies are as follows:

- numpy 1.26.4
- torch 1.12.0
- dgl 1.1.1

## How to run
To run AFECL：

```bash
python train.py --dataset=cora --lr=0.01 --num-heads=4 --num-hidden=32 --tau=1 --weight-decay=1e-4 --in-drop=0.6 --attn-drop=0.5 --epochs=2000 --num-layers=1 --seed=1 --negative-slope=0.2 --rate=1


## Citation
