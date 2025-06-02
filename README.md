# Real-Time-Anomaly-Segmentation-for-Road-Scenes
![Python](https://img.shields.io/badge/language-Python-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![Model](https://img.shields.io/badge/model-ERFNet--ENet--BiSeNet-green)
[![ERFNet](https://img.shields.io/badge/IEEE-8063438-b31b1b.svg)](https://ieeexplore.ieee.org/document/8063438)
[![ENet](https://img.shields.io/badge/arXiv-1606.02147-b31b1b.svg)](https://arxiv.org/abs/1606.02147)
[![BiSeNet](https://img.shields.io/badge/arXiv-1808.00897-b31b1b.svg)](https://arxiv.org/abs/1808.00897)

This repository contains the code developed for the project **Real-Time Anomaly Segmentation for Road Scenes**, carried out within the Advanced Machine Learning course (01URWOV) in AY 2024/25 at Politecnico di Torino.

## Description
This project focuses on real-time segmentation of anomalies in road scenes, a crucial aspect for autonomous driving applications. Leveraging deep learning models, the system is trained on the *Cityscapes* dataset to identify and segment anomalous objects or situations that may represent hazards or deviations from normal road conditions.

## Packages

For detailed usage instructions, please refer to the README files in the relevant folders.

- [train](train): Tools for training semantic segmentation networks on the *Cityscapes* dataset. The README present in this folder is specific to ERFNet, but the same training procedure can be applied to other architectures used in this project, such as BiSeNetV1 and ENet.
- [eval](eval): Tools for evaluating/visualizing networks' output and performing anomaly segmentation.
- [save](save): Contains a separate folder for each training run. Each folder includes useful output files such as an automated log, weights of the model at the epoch with best validation accuracy, the best IoU achieved during training and the corresponding epoch.
- [plots](plots): Contains evaluation visualizations, including PR and ROC curves for MSP, MaxLogit, and MaxEntropy methods across different anomaly datasets drawn from the *Fishyscapes* and *SMIYC* benchmarks. Also includes visual results showing qualitative comparisons on a selected image from the *Road Anomaly* dataset.
- [imagenet](imagenet): Contains the script and model definition for pretraining ERFNet's encoder on the ImageNet dataset.
- [trained_models](trained_models): Contains the trained models used in the official papers.
- [AML_AnomalySegmentation.ipynb](AML_AnomalySegmentation.ipynb): Colab Notebook containing all scripts for training, validation, and visualization.

## Requirements

- [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "\_labelTrainIds" and not the "\_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
- [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
- [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0).
- **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)
- **For testing the anomaly segmentation model**: Road Anomaly, Road Obstacle, and Fishyscapes dataset. All testing images are provided here [Link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view).

## Contributing
<table>
  <tr>
    <td align="center" style="border: none;">
      <a href="https://github.com/AgneseRe">
        <img src="https://github.com/AgneseRe.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>AgneseRe</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/GiorgioRuvolo">
        <img src="https://github.com/GiorgioRuvolo.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>GiorgioRuvolo</sub>
      </a>
    </td>
  </tr>
</table>

## License
This project is licensed under the MIT License.