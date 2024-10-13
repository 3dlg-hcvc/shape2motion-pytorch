# Shape2Motion-PyTorch

## Introduction

The repository is an unofficial PyTorch implementation of the paper [Shape2Motion: Joint Analysis of Motion Parts and Attributes from 3D Shapes](https://arxiv.org/abs/1903.03911), please refer to the [official TensorFlow implementation](https://github.com/wangxiaogang866/Shape2Motion) for more details.


## Project Overview

Shape2Motion is an innovative approach to mobility analysis of 3D shapes. It performs joint analysis for simultaneous part segmentation and motion estimation from a single 3D model input. The method has 3 stages:

1. **Motion Part Proposal Module and Motion Attribute Proposal Module**
2. **Proposal Matching Module**
3. **Mobility Optimization Network**

## Installation

The project requires Python 3+ and PyTorch 1.7+.

```bash
conda create -n shape2motion python=3.8
conda activate shape2motion
pip install -r requirements.txt
pip install -e .
```

## Training

We use hydra to manage the training process, please refer to the [hydra documentation](https://hydra.cc/docs/intro/) for more details.

To train the model, run the following command:

```bash
export HAS_COLOR=true # whether the input 3D model has color information
export HAS_NORMAL=true # whether the input 3D model has normal information
export NUM_CHANNELS=6 # the number of channels of the input 3D model, in addition to xyz coordinates
export SEED=42 # random seed
export OUTPUT_DIR=./outputs # the directory to save the output
export AUGMENTATION=false # whether to use augmentation, by default no augmentation is used
python pipeline.py network.has_color=$HAS_COLOR network.has_normal=$HAS_NORMAL \
network.num_channels=$NUM_CHANNELS network.random_seed=$SEED paths.result_dir="${OUTPUT_DIR}" \
network.augmentation.jitter=$AUGMENTATION network.augmentation.flip=$AUGMENTATION network.augmentation.rotate=$AUGMENTATION \
network.augmentation.color=$AUGMENTATION
```

The training process will automatically train all the 3 stages step by step. You can disable some stages by setting the corresponding `run` field to `false` in the config file [configs/network/default.yaml](configs/network/default.yaml).

For example:
```bash
# disable stage 1 and stage 2
python pipeline.py network.stage1.run=false network.stage2.run=false
```

## Evaluation

Run the evaluation by:

```bash
export OUTPUT_DIR=./evaluations # the directory to save the output
python evaluate.py output_dir=${OUTPUT_DIR}
```

## Results

Compare the results with the original method reported in the paper on the Shape2Motion dataset:


| Method                             | IoU   | EPE   | MD    | OE    | TA    |
|------------------------------------|-------|-------|-------|-------|-------|
| Original Shape2Motion              | **84.70** | **0.025** | **0.010** | 6.875 | 98.00 |
| Shape2Motion re-implementation (ours) | 80.07 | 0.062 | 0.030 | **1.604** | **99.70** |

## Acknowledgements

This repository is built upon the [official TensorFlow implementation](https://github.com/wangxiaogang866/Shape2Motion).
We thank the authors for their great work.

## Citation

If you find this repository useful, please consider citing:

```
@inproceedings{mao2022multiscan,
    author = {Mao, Yongsen and Zhang, Yiming and Jiang, Hanxiao and Chang, Angel X, Savva, Manolis},
    title = {MultiScan: Scalable RGBD scanning for 3D environments with articulated objects},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2022}
}
```

