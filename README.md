# Flower Classifier

This repository contains a Convolutional Neural Network (CNN) classifier designed to accurately categorize five different types of flowers.

## Introduction

In this project, we implement a CNN classifier capable of distinguishing between five different types of flowers. The dataset utilized is the [Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html), which originally contains 17 categories of flowers, each with 80 images. However, to optimize GPU usage and training time, we focus on a subset of five categories: daisy, dandelion, rose, sunflower, and tulip. These flowers were chosen as they are common in the UK. The dataset exhibits significant variations in scale, pose, lighting conditions, and intra-class and inter-class similarities. Prior to training, we normalize the images using three methods and apply various affine transformations such as rotation and scaling to augment the dataset.

## Method

For image classification, we employ ResNet50 [ResNet50](https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition), a deep convolutional neural network architecture consisting of 50 layers. ResNet50 addresses the vanishing gradient problem by introducing skip connections, enabling the training of very deep networks. Widely used in tasks like image classification and feature extraction in computer vision, ResNet50 proves to be effective for our classification task. We train the model using approximately 5000 images over 10 epochs.

## Results

Our model achieved high Area Under the Curve (AUC) and accuracy scores on both the training and validation datasets. The progress of increasing accuracy, AUC, and decreasing loss throughout the epochs can be observed sequentially in plots A, B, and C.

a: <img src="https://github.com/Anndischeh/Flower_Images_Classifier/blob/11539b7992da6fb19a40f3ad2cbefb86272d7633/Media/Acc.png" alt="Accuracy Plot" style="width:30%;">
b: <img src="https://github.com/Anndischeh/Flower_Images_Classifier/blob/11539b7992da6fb19a40f3ad2cbefb86272d7633/Media/AUC.png" alt="AUC Plot" style="width:30%;">
c: <img src="https://github.com/Anndischeh/Flower_Images_Classifier/blob/11539b7992da6fb19a40f3ad2cbefb86272d7633/Media/LOSS.png" alt="Loss Plot" style="width:30%;">

## Example

Below is an example of a classified image with its true and predicted labels, demonstrating the accuracy of our method.

<img src="https://github.com/Anndischeh/Flower_Images_Classifier/blob/11539b7992da6fb19a40f3ad2cbefb86272d7633/Media/example.png" alt="Example Image" style="width:60%;">
