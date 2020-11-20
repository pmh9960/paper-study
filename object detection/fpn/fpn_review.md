# Feature Pyramid Networks for Object Detection

저자 :  
Tsung-Yi Lin

[FPN 원문](1612.03144.pdf)

### Contribution

1. Region proposal network에 FPN 모델 제안.
2. 연산량을 크게 늘리지 않으면서 **다양한 크기의 물체들을 찾는 작업**의 정확도 높임

# 1. Introduction

## 다양한 크기의 물체를 찾는 작업

한 종류의 이미지 크기와 한 종류의 필터 사이즈로는 아주 작은 물체나 아주 큰 물체를 찾기 힘들다.  
해결책은 총 4종류가 존재한다. (FPN 포함)

![][fig1_img]

### a. Featurized image pyramid

가장 고전적으로 사용했던 방법.  
이미지의 크기를 다르게 입력하여 같은 크기의 필터라도 효과가 다르게 작동한다.  
하지만 같은 작업을 복수로 진행하니 **computation cost**가 몇 배가 된다.  
_하나의 네트워크가 서로 다른 해상도에 대해서 잘 작동할지도 의문. (내 생각)_

### b. Single feature map

**DCNN이 발전**하면서 다양한 크기의 물체에 대한 robustness가 좋아졌다.  
하지만 그럼에도 한계가 존재.  
(a)와 함께 쓰여서 어느 정도 극복하지만, 이 역시 cost가 증가함.

### c. Pyramidal feature hierarchy

(a) 처럼 다양한 이미지를 넣는 효과를 주면서 cost는 늘리고 싶지 않다.  
Subsample 되면서 작아지는 feature map을 각각 prediction.  
두 가지 방법이 있다.

1. 기존 네트워크에 그림의 layer 추가.
   1. 높은 해상도의 feature map을 사용하지 못하여 작은 물체에 대해서 여전히 잘 찾지 못함. acc. 낮음. (실험적으로 보임)
2. 기존 네트워크 feature map 사용하여 prediction.
   1. _Low-level feature map(fine-layer)에 대해서 **semantic 정보가 부족**하여 acc. 낮음._ (설명이 없어서 내 생각)

### d. Feature Pyramid Network

**Coarse한 feature map의 semantic 정보를 가져오면서 높은 해상도(작은 물체 찾기 좋은)의 feature map도 이용하여 prediction**

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

# 4. Applications

## 4.1. RPN

### 다시 보는 RPN의 구조

![][rcnn architecture with loss]

### FPN 도입

Anchors를 기존에는 3-scale, 3-aspect ratios로 총 9개의 anchors 사용.  
DCNN, Feature Extractor 부분을 FPN으로 변경하였음.

![][fpn+rpn]

FPN을 통해 서로 다른 크기의 feature map에서 같은 크기 ($7 \times 7$) anchor를 이용. (Aspect ratio는 그대로 {1:2, 1:1, 2:1})  
깊은 feature map에서는 같은 크기라도 feature map이 훨씬 작기 때문에 RoI pooling시 크기는 훨씬 크다. (e.g. $\{P_2, P_3, ... P_6\}$에서 각각 anchor의 크기는 $\{32^2, 64^2, ... 512^2\}$의 효과를 가짐.)  
위 그림은 설명을 위한 예시이고 실제로는 총 ResNet의 **5**개의 층을 FPN을 이용하여 만들었음. = **15 anchors.**  
**각각의 층에서 만들어낸 feature map을 같은 binary classifier, bounding box regressor에 집어넣는다.** (share parameters, 다르게도 진행해 보았는데 의미있는 차이를 보이지 않았다고 한다.)

## 4.2. Fast R-CNN

이제 RPN을 만들어 RoI를 찾아내었으니 RoI pooling, 물체들 간의 classsification과 bounding box regression을 진행해야 한다.  
기존 R-CNN계열은 RoI pooling을 DCNN의 출력인 conv4에 대해서 진행하였는데, feature map이 다양한 현재의 상황에서 어느 feature map에 projection을 진행해야 할까.

$$
k = [k_0 + \log_2(\sqrt{wh}/224)]_{floor}, \text{where} \ k_0 = 4
$$

$k_0$가 4이고 224의 의미는 기존 Fast R-CNN이 224의 크기로 conv4의 layer에서 진행했기에 이를 기본값으로 정의한 것이다.  
이를 기준으로 크기가 더 작다면 더 작은 $k$, 즉 finer-layer에서 RoI pooling을, 더 크다면 coarser-layer에서 RoI pooling을 진행한다는 뜻이다. _(그냥 RoI를 찾은 layer에서 진행하면 안되나..?)_

# 5. Experiments on Object Detection

Experiments 부분은 항상 어떻게 읽어야 할지 잘 모르겠다.

본 논문에서는

1. Baseline을 잡았고(기존 모델 or F.E.의 feature map을 conv4에서 conv5등으로 수정한 모델),
2. Top-down의 중요성 (bottom-up만 있는 것보다 정확도 올라감)
3. Lateral connection의 중요성 (bottom-up, top-down은 존재하나 lateral connection의 유무로 실험)
4. Pyramid의 중요성 (U-Net처럼 $P_2$만 뽑아서 봄.)

을 기준으로 실험하였고 모두 정화도가 올라갔다.

# 6. Extensions: Segmentation Proposals

Mask R-CNN에서 다루겠다.

# 7. Conclusion

**_Scale variation에 robust한 ConvNet 모델 제안하였다._**

<!-- reference -->

[fig1]: img/fig1.png
[fig1_img]: img/fig1_only_img.png
[fig2]: img/fig2.png
[fig3]: img/fig3.png
[rcnn architecture with loss]: https://miro.medium.com/max/700/1*Fg7DVdvF449PfX5Fd6oOYA.png
[fpn+rpn]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FDFL4S%2FbtqEeX5IAp0%2FCbvO9zsvHU9Z6fNcrFkf8K%2Fimg.jpg
