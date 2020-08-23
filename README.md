# ImageFiltering

## Introduction

This repository contains code to train, use and evaluate binary classification models to tag images as naturally occurring/photographic or not.
With this code you can train both coarse and fine-grained classification models as described in [our paper](#).

This research is derived from and extends [Houda Alberts](https://github.com/houda96)' master thesis.

## Environment

Create an environment in Python 3.6 and install the requirements as below.

    python3 -m venv venv_name
    pip install -r requirements.txt

## Data

There are two versions of the ImagiFilter dataset: v1.0 and v1.1.
Version v1.0 is the original version and the one you should use in order to reproduce the results from [our paper](#).
In the second version v1.1 we have fixed a few images that have ambiguous/incorrect labels in v1.0, which is around 6 images.
V1.0 contains images and their coarse- and fine-grained annotations, as well as train and validation splits, whereas v1.1 contains the updated file names according to the new assigned coarse and fine-grained classes, which can be used later on by changing the dataset accordingly. 

- Download v1.0 [here](https://surfdrive.surf.nl/files/index.php/s/YeC4eKiFCxHoNIW).
- Download for version v1.1 [coming soon](#).

## Usage

We assume you want to reproduce the results in our paper.

### Data Extraction

First, download [ImagiFilter v1.0](https://surfdrive.surf.nl/files/index.php/s/YeC4eKiFCxHoNIW) if you have not yet done so and untar it. You will create a `data` directory with subdirectories `positive_images` and `negative_images` containing images labelled positive and negative, respectively. You will also find files `train.json` and `test.json`.

    tar zxvf imagi-filtering-data.tgz

### Training (coarse prediction)

To train a LeNet CNN architecture from scratch, it is necessary to set the resize argument to True and run:

    python train.py --model lenet --resize True

To fine-tune a VGG19 CNN architecture pretrained on ImageNet, run:

    python train.py --model vgg

To fine-tune a ResNet-152 CNN architecture pretrained on ImageNet, run:

    python train.py --model resnet

- To ease the experiments, automation scripts in bash are also provided. 

### Training (fine-grained prediction)

Training for fine-grained image classification is very similar to above. To train a LeNet CNN architecture from scratch, it is necessary to set the resize argument to True and run:

    python train_finegrained.py --model lenet --resize True

To fine-tune a VGG19 CNN architecture pretrained on ImageNet, run:

    python train_finegrained.py --model vgg

To fine-tune a ResNet-152 CNN architecture pretrained on ImageNet, run:

    python train_finegrained.py --model resnet

### Evaluation

To be added.

## Citation

If you found this repository useful, please cite our paper.

[Paper bibtex description](#)

