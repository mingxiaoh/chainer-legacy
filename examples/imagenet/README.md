# Large Scale ConvNets

## Requirements

- Pillow (Pillow requires an external library that corresponds to the image format)

## Description

This is an experimental example of learning from the ILSVRC2012 classification dataset.
It requires the training and validation dataset of following format:

* Each line contains one training example.
* Each line consists of two elements separated by space(s).
* The first element is a path to 256x256 RGB image.
* The second element is its ground truth label from 0 to 999.

The text format is equivalent to what Caffe uses for ImageDataLayer.
This example currently does not include dataset preparation script.

This example requires "mean file" which is computed by `compute_mean.py`.

## Training
 Configuration suggestions about hyper parameters when training Imagenet with Intel Architectures by `train_imagenet_ia.py`

 * googlenet: --batchsize 96 --epoch 60 --poly_policy 0 --poly_power 0.5 --base_lr 0.012 --iteration 800000 --weight_decay 0.0002 --momentum 0.9

 * alexnet: --batchsize 256 --epoch 50 --poly_policy 0 --poly_power 1 --base_lr 0.07 --iteration 250000 --weight_decay 0.0005 --momentum 0.9

 * resnet50: --batchsize 128 --epoch 32 --poly_policy 0 --poly_power 1 --base_lr 0.1 --iteration 320000 --weight_decay 0.0001 --momentum 0.9

 * vgg16: --batchsize 128 --epoch 32 --poly_policy 0 --poly_power 0.6 --base_lr 0.01 --iteration 320000 --weight_decay 0.0005 --momentum 0.9

