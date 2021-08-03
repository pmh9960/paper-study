# Deeplab (v1, v2, (v3))

저자 :  
Liang-Chieh Chen

출처 :
Deeplab paper [v1](v1/1412.7062.pdf), [v2](v2/1606.00915.pdf), [v3](v3/1706.05587.pdf), [v3+](v3+/1802.02611.pdf)  
[Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](v2/Fully%20Connected%20CRFs/fall2016_slide15.pdf)

### Contribution

1. DCNNs 때문에 coarse 한 feature map으로부터 segmentation이 가능할 정도로 fine하게 하는 새로운 방법 제안.
2. Version 별
   1. v1에서는 **atrous convolution**,
   2. v2에서는 **ASPP**와 **fully connected CRF**,
   3. v3에서는 atrous convolution을 이용하여 **더 깊은 network** 구현,
   4. v3+에서는 **Encoder-Decoder**와 결합하였다.
3. 비교
   1. 기존(FCN, U-Net ...) : pooling 있음. 이후 upsample. / Encoder-Decoder. 둘 다 같은 맥락.
   2. Deeplab : **pooling 제거. upsample 제거.** 하지만 field of view 유지.

Upsample 자체가 무슨 방법으로도 효과가 좋지 않아서 만든 기법으로 보임. (물론 upsample 보다 #parameters도 감소.)

# 전체 프로세스 (v2)

![][v2_fig1]

DCNN : VGG-16 or ResNet-101

# Atrous convolution

이해하는데 두 가지 접근 방식이 존재한다.

1. 필터 사이즈가 5X5, 7X7인데 사이사이에 0으로 고정된 parameter 존재.
2. 넓게 떨어져있는 파라미터를 계산한다.

2번 방식이 계산량이 적다. 아래 방식이 2번 방식.

$$
y[i] = \sum_{k=1}^{K}{x[i + r \cdot k]w[k]}
$$

![][v3_fig1]

파란색 네모가 field of view. 대략적으로 이런 식으로 field of view가 유지된다는 뜻.  
이러한 atrous convolution을 DCNN의 마지막 conv, pool을

# Bi-linear interpolation

![][wiki_bilinear_interpolation]

# Atrous Spatial Pyramid Pooling (ASPP)

R-CNN의 SPP에서 아이디어를 받음.

![][v2_fig4]

### Multiple scales objects problem

다양한 크기의 물체가 존재하면 한 가지 크기의 필터로는 한계가 있다.  
너무 작으면 찾지 못하고 너무 크면 쪼개버리는 문제 발생.

![][v3_fig2]

**해결법 4종류 (v2, v3)**

1. 이미지 크기를 변화시킨다.
2. Encoder-Decoder를 이용하면서 큰 feature map부터 작은 feature map들이 영향을 주도록 한다.
3. 몇 번의 atrous convoltion을 더 적용하여(deeper) 큰 field of view를 획득한다. (v3)
4. 필터 사이즈를 다양하게 하는 ASPP를 적용한다. (v2, v3)

_필터 사이즈 다양 vs. Encoder-Decoder ?_  
=> v3+에서는 결합하였음

# Fully Connected CRF

### Energe function

$$
E({\textbf x}) = \sum_{i}{\theta_{i}(x_{i})} + \sum_{ij}{\theta_{ij}(x_{i}, x_{j})}
$$

$\theta_{i}(x_{i}) = -\log P(x_i)$ : Unary potential  
$\theta_{ij}(x_{i}, x_{j})$ : Pairwise potential

Pairwise potential이 원래 모든 i, j에 대해서 계산하기에 계산량이 너무 많아 지엽적으로만 진행하였었음.

|   Grid CRF    | Fully connected CRF |
| :-----------: | :-----------------: |
| ![][crf_grid] |   ![][crf_fully]    |

이 경우를 이전의 segmentation들이 smoothing에 사용하였다. 우리가 원하는 것은 그 반대.  
_Efficient Inference on Fully connected CRF_ 라는 논문이 계산 속도를 ~50,000 nodes에 대해 0.2초로 줄여주었다.  
CRF 분포에 대해서 Mean field approximation에 기반으로 계산하여 가우시안 convolution으로 표현할 수 있다고 하는데... 잘 모르겠음.

![][v2_pairwise_potential_equation]

$p$ : Position  
$I$ : Intensity
$\mu$ : 다르면 1, 같으면 0

이렇다 하고 넘어가겠음.

사용한 결과 :

![][v2_fig5]

### 왜 잘 작동하는가?

<!-- reference -->

[v3_fig1]: img/v3_fig1.png
[v2_fig1]: img/v2_fig1.png
[wiki_bilinear_interpolation]: https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Bilinear_interpolation_visualisation.svg/170px-Bilinear_interpolation_visualisation.svg.png
[v2_fig4]: img/v2_fig4.png
[v3_fig2]: img/v3_fig2.png
[v2_pairwise_potential_equation]: img/v2_pairwise_potential_equation.png
[v2_fig5]: img/v2_fig5.png
[crf_grid]: img/crf_grid_crf.png
[crf_fully]: img/crf_fully_connected_crf.png
