# Real-Time-Anomaly-Segmentation-for-Road-Scenes
![Python](https://img.shields.io/badge/language-Python-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![Model](https://img.shields.io/badge/model-ERFNet--ENet--BiSeNet-green)

[![ERFNet](https://img.shields.io/badge/IEEE-8063438-b31b1b.svg)](https://ieeexplore.ieee.org/document/8063438)
[![ENet](https://img.shields.io/badge/arXiv-1606.02147-b31b1b.svg)](https://arxiv.org/abs/1606.02147)
[![BiSeNet](https://img.shields.io/badge/arXiv-1808.00897-b31b1b.svg)](https://arxiv.org/abs/1808.00897)

This repository provides a starter-code setup for the Real-Time Anomaly Segmentation project of the Machine Learning Course. It consists of the code base for training ERFNet on the Cityscapes dataset and perform anomaly segmentation.

## Packages

For instructions, please refer to the README in each folder:

- [train](train) contains tools for training the network for semantic segmentation.
- [eval](eval) contains tools for evaluating/visualizing the network's output and performing anomaly segmentation.
- [imagenet](imagenet) Contains script and model for pretraining ERFNet's encoder in Imagenet.
- [trained_models](trained_models) Contains the trained models used in the papers.

## Requirements:

- [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "\_labelTrainIds" and not the "\_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
- [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
- [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0).
- **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)
- **For testing the anomaly segmentation model**: Road Anomaly, Road Obstacle, and Fishyscapes dataset. All testing images are provided here [Link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view).

## Anomaly Inference:

- The repo provides a pre-trained ERFNet on the cityscapes dataset that can be used to perform anomaly segmentation on test anomaly datasets.
- Anomaly Inference Command:`python evalAnomaly.py --input '/home/shyam/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png`. Change the dataset path `'/home/shyam/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png`accordingly.


