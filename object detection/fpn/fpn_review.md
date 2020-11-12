# Feature Pyramid Networks for Object Detection

저자 :  
Tsung-Yi Lin

[FPN 원문](1612.03144.pdf)

### Contribution

1. Region proposal network에 FPN 모델 제안.
2. 연산량을 크게 늘리지 않으면서 다양한 크기의 물체들을 찾는 작업의 정확도 높임

# 1. Introduction

## 다양한 크기의 물체를 찾는 작업

한 종류의 이미지 크기와 한 종류의 필터 사이즈로는 아주 작은 물체나 아주 큰 물체를 찾기 힘들다.  
해결책은 총 4종류가 존재한다. (FPN 포함)

![][fig1_img]

### a. Featurized image pyramid

가장 고전적으로 사용했던 방법.  
이미지의 크기를 다르게 입력하여 같은 크기의 필터라도 효과가 다르게 작동한다.  
하지만 같은 작업을 복수로 진행하니 computation cost가 몇 배가 된다.

### b. Single feature map

DCNN이 발전하면서 다양한 크기의 물체에 대한 robustness가 좋아졌다.  
하지만 그럼에도 한계가 존재.  
(a)와 함께 쓰여서 어느 정도 극복하지만, 이 역시 cost가 증가함.

### c. Pyramidal feature hierarchy

(a) 처럼 다양한 이미지를 넣는 효과를 주면서 cost는 늘리고 싶지 않다.  
Subsample 되면서 작아지는 feature map을 각각 prediction.  
두 가지 방법이 있다.

1. 기존 네트워크에 그림의 layer 추가.
   1. 높은 해상도의 feature map을 사용하지 못하여 acc. 낮음. (실험적으로 보임)
2. 기존 네트워크 feature map 사용하여 prediction.
   1. _Low-level feature map에 대해서 semantic 정보가 부족하여 acc. 낮음._ (설명이 없어 내 생각)

### d. Feature Pyramid Network

Coarse한 feature map의 semantic 정보를 가져오면서 높은 해상도(작은 물체 찾기 좋은)의 feature map도 이용하여 prediction

## U-Net 계열과의 차이점

![][fig2]

U-Net 계열은 최종적으로 하나의 높은 해상도 feature map을 얻고 싶어했다.  
FPN은 각 층별로 독립적으로 prediction 하는 것이 목표이다.

# 3. Feature Pyramid Networks

FPN은 다양한 곳에 사용할 수 있지만 본 논문에서는 RPN으로 사용하였다. _(RPN외에 어디에 쓸 수 있을까?)_  
Instance segmentation에도 사용할 수 있음. (Sec. 6)

## FPN의 구성 요소

![][fig3]

1. Bottom-up pathway
2. Top-down pathway
3. Lateral connections

### 1. Bottom-up pathway

Backbone ConvNet의 feed-forward computation과 같다.  
ResNet으로 실험하였는데, 가장 처음 convolution layer인 $conv1$을 제외하고 $conv2$, $conv3$, $conv4$, $conv5$를 이용하였다. ($\because$ 메모리 문제)  
_(모든 layer를 FPN으로 만들면 어떨까?)_

### 2. Top-down pathway

Coarse한 높은 level의 layer를 upsample을 통해 해상도를 높이는 과정이다.  
본 논문은 구조에 대한 논문이라 upsampling은 nearest neighbor upsampling으로 간단하게 진행.  
Bottom-up 과정에서의 feature map과 해상도는 같더라도 FOV(Field of view, or receptive field)를 생각해보면 보다 semantically stronger 한 feature map임을 알 수 있다.

### 3. Lateral connections

_(Upsampling은 뭔짓을 해도 완벽하게 convolution의 역연산을 수행할 수 없기 때문에 완전하지 않아서 필요하다고 생각.)_

Bottom-up pathway에서 거쳐온 feature map들을 $1 \times 1$ convolution을 통해 dimension을 맞추고 element-wise로 더해준다. _(Inception 처럼 차원을 이어 붙여주는건 어떤가?)_  
각각의 더해진 feature map은 층에 관계없이 같은 classifier/regressor에 들어갈 것이기 때문에 dimension은 모두 256으로 맞춰준다.

<!-- reference -->

[fig1]: img/fig1.png
[fig1_img]: img/fig1_only_img.png
[fig2]: img/fig2.png
[fig3]: img/fig3.png
