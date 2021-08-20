# Fast R-CNN

저자 : Ross Girshick
출처 :

1. https://arxiv.org/abs/1504.08083
2. [갈아먹는 머신러닝](https://yeomko.tistory.com/15)
3. [Arun Mohan](https://medium.com/datadriveninvestor/review-on-fast-rcnn-202c9eadd23b)

**_Contribution : 새로운 학습 알고리즘으로 R-CNN, SPPNet의 단점들을 해결하였다._**

1. R-CNN, SPPNet보다 검출 성능이 좋아졌다. (mAP)
2. Multi-task loss를 이용하여 하나의 stage로 end-to-end 학습하였다.
3. Training으로 모든 네트워크 layers(include conv)를 update 할 수 있다. (R-CNN, SPPNet은 fc-layer만 가능했음.)
4. Feature caching으로 인한 space resource를 절약할 수 있다.

## Fast R-CNN Architecture

![][fast r-cnn figure 1]

1. 전체 이미지를 미리 학습된 CNN을 통과시켜 feature map을 추출.
2. Selective Search로 찾은 각각의 RoI에 대하여 feature map에 projection.
   1. RoI pooling 진행
   2. 고정된 크기의 feature vector를 얻는다.
3. Feature vector를 fc-layers에 통과시킨다.
4. 이후 두 branches로 나뉜다.
   1. Softmax를 통과하여 해당 RoI가 어떤 물체인지 클래시피케이션 합니다. 더 이상 SVM은 사용되지 않습니다.
   2. Bouding box regression으로 selective search로 찾은 박스의 위치 조정.

SPPNet과 굉장히 유사하다.  
차이점은 Pyramid 구조가 한 층으로 바뀌어서 end-to-end로 학습할 수 있고, SVM 대신 softmax를 사용하여 classify 하였다. 또한 BB regressor도 SVM뒤에 오지 않고 RoI feature vector 바로 뒤에 왔다.

## RoI projection

![][roi projection]

CNN을 통해서 나온 feature map은 원래 이미지와 크기가 다르다. (Subsampling ratio : 한 변의 길이가 줄어든 비율로 계산. e.g. 18X18 => 3X3일 때 1/6)  
단순하게 subsampling ratio를 곱한 위치로 RoI를 Projection 시킨다. (딱 떨어지지 않는건 대충 반올림 or 내림)

## RoI Pooling

Convolution 이후 나온 임의의 크기인 feature map의 크기가 $h\times w$이고, RoI의 목표 window가 $H\times W$라면, $h/H\times w/W$의 크기로 max-pooling을 진행해야 한다.  
이는 사실 SPPNet에서 한 층 짜리 SPP layer와 같다.

## Train Fast R-CNN

### 왜 SPPNet은 SPP layer 이전으로 backpropagation을 통한 update를 진행할 수 없는가?

SPPNet의 역전파가 RoI가 서로 다른 이미지에서 나올 떄 매우 비효율적이다. (왜 비효율적 ? 왜 Fast의 RoI pooling은 괜찮은가 ?)  
RoI가 매우 큰 receptive field를 가질 수 있기 때문이라고 한다. (하지만 잘 이해 안됨. 이미지의 concept을 너무 많이 담았기 때문이라는 것인가 ?)

### Multi task Loss

Fast R-CNN에는 두 개의 output layers가 존재한다. 즉, loss가 두 개 나온다. Backpropagation을 하기 위해서는 두 loss를 적절하게 엮어 하나로 만들어주어야 한다.

$$
L(p, u, t^u, v)=L_{\text{cls}}(p, u) + \lambda [u \geq 1]L_{\text{loc}}(t^u, v)
$$

$p = (p_0, ..., p_K)$ : RoI에 대해서 예측한 softmax 값. (K+1개의 categories)  
$u$ : 실제 ground truth 값.

$$
L_{\text{cls}}(p, u) = - \log{p_u}
$$

$t^k = (t^k_x, t^k_y, t^k_w, t^k_h)$ : k번째 categori에 대해서, 이렇게 Bounding Box Regression(BBR)을 옮겨라 하는 값.  
$v = (v_x, v_y, v_w, v_h)$ : 실제 BBR 수정하는 값. (target)  
$[u \geq 1]$ : Iverson bracket, 참이면 1. 거짓이면 0. (배경(0)이면 localization loss = 0)

$$
L_{\text{loc}}(t^u, v) = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L_1} (t^u_i - v_i) \\
\text{smooth}_{L_1} (x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x|-0.5 & \text{otherwise,} \end{cases}
$$

Smooth를 하는 이유 :  
부분적으로 L1, L2 norm이 섞여 있는데, L2 norm 만을 사용하였을 때보다 outliers에 robust하다. (L1이 원래 더 robust)  
이전에는 exploding gradient 문제가 존재했는데 이를 해결함.

## Train

### Backpropagation through RoI pooling layers

Fast R-CNN은 R-CNN, SPPNet에 비하여 end-to-end로 학습시킬 수 있다는 장점이 있다.  
하지만 모든 layer를 학습시키기에는 당시 컴퓨터나 데이터 수로는 한계가 존재하였다. (앞서 언급했던 gradient exploding, vanishing도 문제로 포함이 되는 것 같다.)  
그렇기 때문에 **_몇 번쨰 layer까지 backpropagation을 진행할지_**가 다음 과제이다.

<!-- TODO 맑은 정신으로 다시 읽어볼 것 -->

![][which layers to fine-tune]

결과적으로는 위 표와 같이, **_깊게 하면 할 수록 성능이 좋아졌다._** 즉 CNN(Feature extraction)단을 Object detection에 맞게끔 fine-tuning 하는 것이 성능 향상에 많은 도움을 주었다. (R-CNN, SPPNet 대비)

## Conclusion

### 장점

End-to-end

1. 학습 단계 간소화
2. 성능 향상
3. Inference time = ~0.3초

### 단점

1. 여전히 region proposal을 selective search로 수행. (CPU로만 수행 가능) **_왜?_**

### Faster

Region proposal 역시 전체 네트워크의 일부로 끌어온다!

### 기타

> SVD (Singular Vector Decomposition, 특이값 분해)를 통해서 Fully Connected Layer 들의 파라미터를 줄이는 방법 등이 소개되었지만 이후의 연구들에서는 사용되어 지지않고, 지나치게 어렵기 때문에 쿨하게 넘기겠습니다. - 갈아먹는 머신러닝

<!-- reference -->

[fast r-cnn figure 1]: ../R-CNN%20계열/Fast%20R-CNN/img/figure_1.png
[roi projection]: https://miro.medium.com/max/616/1*dPHlfhhy2PlfD9Y5LHBa6w.png
[which layers to fine-tune]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbadZIp%2FbtqAVIwqRP6%2FW9hTlTIcKm6JNlFDTsWf4K%2Fimg.png
