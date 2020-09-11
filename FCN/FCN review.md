# Fully Convolutional Networks for Semantic Segmentation

# Abstract

- Convolutional networks는 feature hierarchies를 산출하는데 탁월한 능력을 가진 모델이다.
- 본 논문의 모델은 [Semantic segmentation](#semantic-segmentation) 가장 좋은 결과를 보여주며, **_end-to-end, pixels-to-pixels_**로 훈련하였다.
- **_Fully Convolutional Networs_**
  > Fully Convolutional Networks는 입력값으로 임의의 크기(input of arbitrary)를 가질 수 있고, 이에 따라서 연관된 크기의 출력물(correspondingly-sized output)을 출력하는 장점이 있다. (Semantic segmentation에서 강점)  
  > $\Leftrightarrow$ 기존 분류 네트워크들은 입력 사이즈가 고정되어 있었다.
- 현대의 Classification Networks(AlexNet, VGG net, GoogLeNet)를 Fully Convolutional Network로 적용시킨 후에 semantic segmentation에 적합하도록 세부적인 부분을 fine-tune 하는 transfer learning 방식으로 진행하였다.
  > Contemporary classification networks : pre-trained models (pre-training) <!-- 사전에 학습되는 모델 -->  
  > Fully Convoutional Network : fine-tuned model <!-- 이를 활용하여 새로운 모델을 학습하는 과정 -->
- 본 네트워크의 특별한 구조.
  > Semantic information은 deep, coarse한 layer에서 추출.  
  > Appearance information은 shallow, fine한 layer에서 추출.  
  > $\Rightarrow$ 정확하고 세부적인 segmentation을 구성하였다.
- PASCAL VOC, NYUDv2, SIFT Flow의 Dataset에서 좋은 성능을 보여주었고, 추론에 걸리는 시간은 1/5초 이하이다.

# 1. Introduction

## Convolutional Networks

### Performance

Fully Convolutional Network improve for the two tasks below.

1. Whole-image classification
   1. Cat or dog
2. Local tasks with structed output
   1. bounding box object detection
   2. part and keypoint prediction
   3. local correspondence
   <!-- TODO What are they? -->

**_Why convnets work? (reference VGG-16)_**

[VGG16 review](VGG16%20review%20review.md)

## The next task, **_Coarse to fine_**

### Semantic segmentation

Pixel 단위로 어떤 object인지 classification 하는 것.

![](fcn_imgs/semantic_segmentation.jpg)

### **_Semantics vs. Location_** 의 균형이 중요하다.

[What vs. Where](#skip-architecture)

## Fully Convolutional Networks

이름 그대로 처음부터 끝까지 convolutional layer만을 사용.  
특히, classification part를 fully connected layer에서 convolutional layer로 변경.

### Compare with prior networks

기존에도 semantic segmentation을 위해서 convnets을 이용하려는 연구가 있었다. 하지만 ...  
(구체적인 단점은 나와있지 않음. 아마도 성능 문제)

**_FCN은 추가적인 machinery 요소 없이 state-of-the-art로 넘어섰다._**  
기타 자잘한 잡기술이 아닌 큰 변화를 통한 발전이라는 뜻으로 해석.

### The first End-to-end training...

1. Pixelwise prediction
2. From supervised pre-training

### Advantage

1. Input size가 자유롭다.
2. 학습과 추론 모두 전체 사진을 한 번에 계산하였다. (by [dense feedforward computation](#fully-connected-layers-can-also-be-viewed-as-convolutions-with-kernels-that-cover-their-entire-input-regions))
3. 본 네트워크에 있는 upsampling layers가 subsampled하는 pooling이 있음에도, pixelwise prediction과 학습 모두 가능하게 해준다.
4. Efficiency <!-- TODO for what? -->  
   For what ? Train ? Inference ? Build architecture ?
   1. [Asymptotically efficient](<https://en.wikipedia.org/wiki/Efficiency_(statistics)#Asymptotic_efficiency>) and absolutely efficient (정확한 뜻은 모르겠다.) <!-- Asymtotically : N이 커진다던지 무언가 변화가 있을 것임. 이 때 이 변화에 따라서 서서히 최적의 값으로 접근한다면 해당된다.  -->
   2. No patchwise training
   3. Pre- and post-processing을 복잡하게 만들지 않는다.
      1. No [superpixels](fcn_imgs/superpixel.jpeg)
      2. [region proposals](fcn_imgs/region_proposals.png) : selective search와 Edge boxes가 주로 사용
      3. post-hoc refinement (사후정제처리)
   4. 학습되어있는 최근의 network의 classification 부분을 fully convolutional과 fine-tuning으로 재해석하였다.
      > 기존 연구들은 작은 convnet에서 학습되어있지 않은 상태로 진행하였다.

### Skip architecture

Deep, coarse semantic information과 shallow, fine, appearance information이 섞인 결과를 낼 수 있었다.  
Architecture design vs. Dense prediction tradeoff  
**_What vs. Where_**

# 2. Related work

## Prior Fully Convolutional Networks

- Matan : Extending a convnet to arbitrary-sized input idea (1D)
- Wolf and Platt : Expand convnet outputs to 2D
- Ning : Define a convnet for coarse multiclass segmentation
- Sermanet : Sliding window detection
- Pinheiro and Collobert : Semantic segmentation
- Eigen : Image restoration
- Tompson : Fully convolutional training effectively
- He : Discard the non-conv portion of classification nets to make a feature extractor.  
  Combine (region) proposals and Spatial pyramid pooling은 국부적이고 고정된 길이의 feature를 classification을 위해 산출해낼 수 있다.
  빠르고 효율적이나 hybrid이기 떄문에 end-to-end training이 불가능하다.

**_Then,_**

- FCN : Draw on recent successes of deep nets for image classification and transfer learning.

## Dense prediction

It is called dense prediction, because pooling subsample images. <!-- TODO is it right? -->

- Ning, Farabet, Pinheiro and Collobert : Recent works have applied convnets to dense prediction problems, including semantic segmentation.
- Ciresan : boundary prediction for electron microscopy
- etc.

There are common elements.

1. Small models restricting capacity and receptive fields
2. Patchwise training
3. Post-processing by superpixel projection, random field regularization, filtering, or local classification
4. Input shifting and output interlacing for dense output as introduced by OverFeat
5. Multi-scale pyramid processing
6. Saturatingtanhnonlinearities
7. Ensembles

**_In summary, the exist methods are not learned end-to-end._**  
**_FCN DOES WITHOUT THIS MACHINERY_** (in fact, their ideas are used as FCN perspective)

# 3. Fully Convolutional Network

## The whole architecture

![](fcn_imgs/fcn_figure1_modified.jpg)

### Notation

Each layer of data in a convnet : $h\times w\times d$  
$h, w$ : spatial dimensions.  
$d$ : feature or channel dimension.

1. Feature extraction with trained classifier(e.g. VGG16)
2. Feature-level classification (it was fc-layer before FCN)
3. Upsampling
4. Segmentation

## Receptive field

The receptive field is **_a portion of sensory space_** that can elicit neuronal responses when stimulated. ([wiki](https://en.wikipedia.org/wiki/Receptive_field), Alonso and Chen(2008))

<img src="fcn_imgs/receptive_field.png" width=50%>

## What is Convnet

### Basic components

1. Convolution
2. Pooling
3. Activation function

operate on local input region

### Convnet is built on translation invariance

Invariance is important in detection part of AI.

- Type of invariance  
  ![](fcn_imgs/translation_invariance.png)

### Depend only on relative spatial coordinates

**_Convolution_**

$$
y_{ij} = f_{ks}(\{x_{si+\delta i, sj+\delta j}\}_{0 \le \delta i, \delta j \le k})
$$

$k$ : kernel size  
$s$ : stride  
$f_{ks}$ : layer type (conv, pool, activate)

**_Transformation rule_**

Therefore, general deep net computes a general nonlinear function.

$$
f_{ks} \circ g_{k's'} = (f \circ g)_{k' + (k-1)s', ss'}
$$

**_Loss function which defines a task._**

Custom-made loss function called Pixel-Wise Loss.

$$
l(x;\theta) = {\Sigma}_{ij} l'(x_{ij};\theta)
$$

# 3.1. Adapting classifiers for dense prediction <!-- What is the different with feature extraction -->

## How to change fc-layer to conv layer

### Typical recognition nets

![](fcn_imgs/cnn_fc_layer.png)

1. Take fixed-sized inputs
1. Produce nonspatial outputs

This is because fully connected layers. (fixed dimensions and throw away spatial coordinates)

### Fully connected layers can also be viewed as convolutions with kernels that cover their entire input regions.

![](fcn_imgs/conv_impliementation_of_sliding_windows.png)

- Faster

  5 times faster than AlexNet  
  both the forward and backward passes are straightforward. (take advantage of the inherrent computational efficiency)

- Subsample

  The classification nets subsample to keep filters small.  
  This coarsens the output.  
  Receptive fields의 pixel stride에 따라서 output unit의 크기를 줄일 수 있다.

- No more depend on input size

  convolution impliementation of sliding windows (OverFeat)

# 3.2. Shift-and-stitch is filter rarefaction

Input shifting and output interlacing

## Few years ago... Inpterpolation

: 보간법

![](fcn_imgs/Comparison_of_1D_and_2D_interpolation.svg)

## OverFeat : Shift-and-stitch trick

If the outputs are downsampled by $f$, the input is shifted(by left and top) $x$ pixels to the right and $y$ pixels down, once for every value below ($f^2$ times).

$$
(x, y) \in \{0, ..., f-1\} \times \{0, ..., f-1\}
$$

- What does it do (OverFeat)

  Less diminution of resolution.

  ![](fcn_imgs/input_shifting.png)

### Shift-and-stitch trade-off

|                 pros                  |                 cons                  |
| :-----------------------------------: | :-----------------------------------: |
| denser w/o decreasing receptive field | can't finer scale than their original |

**_Thus, FCN does not use the shift-and-stitch trick._**

## FCN : Changing only the filters and layer strides of a convnet

$$
f'_{ij} = \begin{cases} f_{i/s, j/s} & \text{if } s \text{divides both } i \text{ and } j;\\0 & \text{otherwise,} \end{cases}
$$

### Why Pooling? (Decreasing subsampling)

If not pooling, we don't have to reproduce.

**_Trade-off_**

|         without pooling          |              with pooling              |
| :------------------------------: | :------------------------------------: |
| Filter can see finer information | The nets have smaller receptive fields |

# 3.3. Upsampling is backwards strided convolution

## Bilinear interpolation

reference [3.2. Interpolaration](#few-years-ago-inpterpolation)

## Backward convolution (Deconvolution)

**_Upsampling with factor $f$ = Convolution with stride $s = 1/f$_**

Thus, end-to-end learning is possible.

Moreover, deconvolutional filter could learn any effective upsampling.  
e.g. bilinear upsampling, non-linear upsampling, etc...

### Transposed Convolution Matrix

How upsample a small size image with learnable parameters.  
First, let redraw image of convolution as a below figure.

<img src="fcn_imgs/conv_reshaped.png" width=80%>

Then, we can transpose the matrix for up-convolution.

<img src="fcn_imgs/conv_transposed.png" width=50%>

In this article, they find more optimized upsampling layer which is in [Section 4.2.](#42-combining-what-and-where)

# 3.4. Patchwise training is loss sampling

## Stochastic optimization

Focus on Computational efficiency

Patchwise training : Uniform (no sample) vs. Random sample patch  
Random sample is more efficient than uniform sampling of patches, because of reducing # of possible batches.

In addition, it is loss sampling which has the effect **_like DropConnect mask._**  
DropConnect, dropout은 하나 또는 몇 개의 노드에 결과값이 너무 많이 의존하는 것을 방지하도록 하여 이미지 인식 성능 개선에 도움을 준다.

![](fcn_imgs/dropout_dropconnect.png)

# Segmentation Architecture

## Use what?

1. Fine-tuning
2. Skip architecture

## What is trained and validated

1. A per-pixel multinomial logistic loss
2. Validate with mean pixel Intersection over Union (IoU, the mean taken over all classes, including background)
3. The training ignores pixels that are masked out in the ground truth.

# 4.1. From classifier

## Begin by proven classification architectures

### Reference networks

Consider the AlexNet, VGG nets, and GoogLeNet $\rightarrow$ select VGG 16-layer net. (be equivalent to the 19-layer net on the task)  
Only the final layer of GoogLeNet is used by loss layer. (discarding the final average pooling)

### Discarding fc-layer $\rightarrow$ to convolutions

Append $1\times1$ convolution with channel dimension 21 (# of PASCAL classes)  
The predict coarse output layer followed by a deconvolution layer. (bilinear... upsampling layer)

### Preliminary validation results

Even the worst model achieved $\sim 75\%$  
FCN-VGG16 already appears 56.0 mean IU

![](fcn_imgs/sample_iou.png)

# 4.2. Combining what and where

### They Define FCN that combines **_feature hierarchy layers_** and **_refines the spatial precision_**.

![](fcn_imgs/fcn_figure3.png)

### Dissatisfyingly coarse output

The result of deep layers is too coarse to segment images.  
For this reason, they combine shallow layers and deep layers as can be seen by figure3.  
It makes sense to make them from shallower net outputs.

# How to make FCN

## FCN architecture

![](fcn_imgs/fcn_architecture.png)

## FCN-32s

Just upsample `conv7` 32x.

- The stride is 32
- Too coarse

## FCN-16s

1. Upsample `conv7` 2x.
2. Add a $1\times1$ convolution layer on top of `pool4`. (to set channel same)
3. Sum both layer. (just sum, backpropagation is easier than max fusion)
4. Upsample 16x.

- The stride is 16
- Less coarse

## FCN-8s

1. Upsample `conv7` 4x.
2. Upsample `pool4` 2x.
3. Take `pool3` (also add a $1\times1$ convolution layer on top as `pool4`)
4. Sum all of them
5. Upsample 8x.

- The stride is 8
- More Less coarse

## No more

### why?

There is not significant result.

## Notice point on FCN architecture

- Initialize 2x upsampling to bilinear interpolation, and allow the parameters to be learned.
- The new params in $1\times1$ conv is initialized with zero. (the net start with unmodified prediction)
- The learning rate is decreased by a factor of 100.

## The skip net (Even FCN-16s) improves performance by 3.0 mean IU to 62.4

![](fcn_imgs/fcn_figure4.png)

Also, find a slight improvement in the smoothness and detail of the output.

## Refinement by other means

### Decreasing the stride of pooling layers

The straight way to obtain finer predictions.  
_Setting the `pool5` layers to have stride 1_ requires _a kernel size of $14\times14$_ in order to maintain its receptive field size.  
Big kernel size means increasing parameters which needs much computational cost.

**_The researchers fail learning such large filters._**

변명 : Initialization from ImageNet-trained weights in the upper layers is important.

### Shift-and-stitch trick

They found that the method is worse than layer fusion.

# 4.3. Experimental framework (Details)

## Optimization

- SGD with momentum
- Learning rate : $10^{-3}, 10^{-4}, and 5^{-5}$ for FCN-AlexNet, FCN-VGG16, and FCN-GoogLeNet, respectively
- Momentum : 0.9
- Zero-initialize the class scoring convolution layer. (Random is worse perfomance and slow)
- Dropout was included in the original classifier nets.

## Fine-tuning

**_Fine-Tune all layers._**  
Fine-tuning only output part yields only $70\%$ of the full fine-tuning.

1. Take 3 days for coarse FCN-32s version.
2. Take 1 day to upgrade to the FCN-16s
3. Take 1 day to upgrade to the FCN-8s

## Patch sampling

<img src="fcn_imgs/fcn_figure5.png" width=50%><br/>

**_Use full image training_**  
By contrast, prior works randomly sampled patches.  
In this research, they recovered that **_random sampling does not have a significant effect on convergence rate_** compared to whole image training, but **_takes significantly more time_** due to **_the larger number of images_** that need to be considered per batch.

## Class Balancing

Unneccessary

## Dense Prediction

- Final layer deconvolutional filters are fixed to bilinear interpolation.
- Intermediat upsampling layers are initialized to bilinear interpolation, but it can be learned.

## Augmentation

- Randomly mirror data
- Jittering (?)

No noticeable improvement.

## More Traning Data

- PASCAL VOC 2011 : 1112 images
- Hariharan's data : 8498 PASCAL training images

Improve 3.4 points to 59.4 mean IU

## Implementation

All models are trained and tested with **_Caffe_**

# 5. Results

## Datasets <!-- TODO what are they -->

1. PASCAL VOC
2. NYUDv2
3. SIFT

## Metircs

from common semantic segmentation.

### Notation

$n_{ij}$ : # of pixels of class $i$ predicted to belong to class $j$  
$n_{cl}$ : # of classes  
$t_i = \Sigma_j n_{ij}$ : the total number of pixels of class $i$

### The 4 metrics

1. pixel accuracy : $\Sigma_i n_{ii} / \Sigma_i t_i$
2. mean accuracy : $(1/n_{cl}) \Sigma_i n_{ii} / t_i$
3. mean IU : $(1/n_{cl})\Sigma_i n_{ii}/(t_i+\Sigma_j n_{ji}-n_{ii})$
4. frequency weight IU : $(\Sigma_k t_k)^{-1} \Sigma_i t_i n_{ii}/(t_i+\Sigma_j n_{ji}-n_{ii})$

## Results of PASCAL VOC

![](fcn_imgs/fcn_table3.png)
![](fcn_imgs/fcn_figure6.png)

## Results of NYUDv2

The images in NYUDv2 are RGB-D images. However, FCN-32s, 16s, 8s are trained with just RGB images. To add depth information, input of the model upgraded to take 4D.

1. Train unmodified FCN-32s model on RGB images.
2. Upgrade input dimension of the model and train on RGB-D images.

![](fcn_imgs/fcn_table4.png)

- HHA is an 3-dims encoding way which transform from RGB-D.
- Training as _late fusion_ of RGB and HHA
- Both nets are summed at the final layer
- Upgrade _late fusion_ to 16 stirde version

## Results of SIFT Flow

SIFT Flow is a dataset of 2,688 images with pixel labes.

- 33 semantic categories ("bridge", "mountain", "sun" ... )
- 3 geometric categories ("horizontal", "vertical", and "sky")

![](fcn_imgs/fcn_table5.png)

- This model performs as well on both tasks as two independent trained models.
- Learning and inference speeds are as fast as each independent models.

# 6. Conclusion

## Future works

If combination with Multi-resolution layer, the Fully Convolutional Networks improves dramatically in segmentation part, while simplifying and speeding up learning and inference.

# A. Upper Bounds on IU

While their mean IU has great performance, there is an upper bounds on mean IU.

| factor | mean IU |
| -----: | :------ |
|    128 | 50.9    |
|     64 | 73.3    |
|     32 | 86.1    |
|     16 | 92.8    |
|      8 | 96.4    |
|      4 | 98.5    |

This result is predicted by downsampling ground truth images and then upsampling them again.  
Conversely, mean IU is not a good measure of fine-scale accuracy.

# B. More results

While PASCAL-Context provides whole scene annotations of PASCAL VOC 2010, and there are over 400 distinct classes. Thus, the researchers do same thing on new dataset.  
In more results part, they experiment 59 classes dataset.  
They train and evaluate. The result is table6.

<!-- conclusion for me -->

<!--
In semantic segmentation problems,
Deep layers make coarse output, shallow layers make fine output.
Coarse output let us know what is it. (global topic) Because it has big receptive field.
Shallow layers let us know where is it. (details) Because it has local information.

The case of deep layers, how about big size of windows?
3X3 windows can also have big receptive field with less parameters which we sholud train. (VGG)
-->
