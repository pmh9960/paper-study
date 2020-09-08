# U-Net

# Abstract

Deep networks는 많은 label된 training sample이 필요하다는 것이 지배적이다.  
본 논문에서는 존재하는 annotated samples을 보다 효율적으로 사용할 수 있는 network와 training strategy를 제안한다.  
U-Net의 구조는 context를 찾을 수 있도록 수축하는 경로와 이와 대칭되게 정확한 위치를 찾을 수 있도록 팽창하는 경로로 구성되어있다.  
U-Net은 매우 적은 수의 이미지들로 end-to-end로 훈련되었고, ISBI challenge에서 기존에 있던 모델들 중 가장 좋은 성능을 보이던 모델(a sliding-window convolutional network)보다 더 좋은 성능을 보여주었다.

> 종목은 electron microscopic stacks이었는데 이후 같은 네트워크를 transmitted light microscopy images로 다시 훈련시켜, ISBI cell tracking challenge 2015에서 큰 차이를 보이며 우승하였다.
> 심지어 이 네트워크는 빠르다. (512x512 image를 segmentation하는데에 1초도 걸리지 않는다.)

# 1. Introduction

## History of convolutional networks

Convolutional networks have already existed for a long time, but their success was limitied due to the size of the available training sets and the size of the considered networks.  
This is broken by Krizhevsky who use 8 layers and millions of parameters on the ImageNet dataset with 1 million training images. Since then, researchers have trained larger and deeper networks in classification tasks.

However, there are not only classification tasks, but also have other tasks such as localization. (especially, in biomedical tasks have)  
There are just thousands of training images for biomedical tasks.  
Hence, Ciresan et al.

1. Train a network in a **_sliding-window setup_**
2. Predict **_the class label of each pixel_**
3. Provide **_a local region(patch)_** around that pixel as input.

There are two noticeable points.

1. The network can localize.
2. The training data in terms of patches is much larger than the number of training images.

Obviously, the strategy in Ciresan et al. has two drawbacks

1. It is quite slow
   > The network must be run separately for each patch, and there is a lot of redundancy due to overlapping patches
2. There is a trade-off between localization accuracy and the use of context.
   > Larger patch vs. small patch  
   > Larger patch require more max-pooling layers which reduce the location information.  
   > Smaller patch allow the network to see only little context.

# U-Net

## Fully Convolutional Network

U-Net is built upon **_Fully Convolutional Network._**  
It is modified and extended for the works with very few training images and yield more precise segmentations.  
The main idea in FCN is upsampling layers which make contracted images to precise images.

## Upsampling part

### A large number of feature channels.

A large number of feature channels allow the network to propagate context information to higher resolution layers.  
As a result, the upsampling path is almost symmetric to the contracting path, and yield a u-shaped architecture.

![](unet_imgs/unet_figure1.png)

## Overlap-tile strategy

Overlap-tile strategy allows the seamless segmentation of arbitrarily large images.

![](unet_imgs/unet_figure2.png)

To predict the pixels in the border region of image, the missing context is extrapolated by mirroring tha input image.  
Tiling strategy applys the network to large images, and also images' resolution is not limited by the GPU memory.

## Data augmentation

If there is very little training data, data augmentation is important.  
They apply elastic defromations to training images.  
Elastic deformation is important in biomedical segmentation, because the tissue is well distorted.  
Dosovitskiy et al. suppose that the data augmentation help to learn invariance of data.

## Separation of touching objects of the same class

Use of a weight loss, where the separating background labels between touching cells obtain a large weight in the loss function.

### This network is applicable to various biomedical segmentation problems

# 2. Network Architecture

U-Net consists of a contracting path, expansive path, and $1\times1$ convolution.  
In total, the network has 23 convolutional layers.

## Contracting path

1. Two $3\times3$ convolutions. (unpadded convolutions, followed by ReLU)  
   Unpadding with mirroring input image is better option than zero-padding, I think.
2. $2\times2$ max pooling (stride: 2, double the number of feature)

## Expansive path

1. Up-convolution ($2\times2$ convolution, halves the number of feature channels)
2. Concatenation with the correspondingly cropped feature map from contracting path. (cropping is necessary, because unpadding has loss of border pixels)
3. Two $3\times3$ convolutions. (unpadded convolutions, followed by ReLU)

## $1\times1$ convolution

The final layer of U-Net.  
The first layer use 64-feature map, but the desired number of classes is not 64.  
$1\times1$ convolution can change the number of feature map simply.

_What is the correation between the number of channel in first layer and the real number of classes which we desire._

**_To allow a seamless tiling of the output segmentation map, selecting the input tile size is important._**  
e.g. $2\times2$ max-pooling operations are applied to a layer with an even x- and y-size.

_Does U-Net patches are seamless while they use patchwise convolution?_
