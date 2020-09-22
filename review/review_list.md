# Review paper list

## Table of Contents

### Paper

1. [R-CNN](#r-cnn)
2. [Spatial Pyramid Pooling Network](#spatial-pyramid-pooling-network)
3. [Fast R-CNN](#fast-r-cnn)
4. [Faster R-CNN](#faster-r-cnn)

### Reference

- Metric
  - [Precision & Recall](#precision--recall)
  - [mAP (Mean Average Precision)](#map-mean-average-precision)

# R-CNN

저자 : Ross et al.  
출처 :

1. [갈아먹는 머신러닝](https://yeomko.tistory.com/13)

**_Contribution : CNN을 사용하여 object detection task의 정확도와 속도를 획기적으로 향상시켰다._**

## R-CNN Architecture

![r-cnn model][r-cnn model2]

1. 입력 이미지에 Selective Search 알고리즘을 적용하여 물체가 있을만한 박스 2천개를 추출한다.
2. 모든 박스를 227 x 227 크기로 리사이즈(warp) 한다. 이 때 박스의 비율 등은 고려하지 않는다.
3. 미리 이미지 넷 데이터를 통해 학습시켜놓은 CNN을 통과시켜 4096 차원의 특징 벡터를 추출한다.
4. 이 추출된 벡터를 가지고 각각의 클래스(Object의 종류) 마다 학습시켜놓은 SVM Classifier를 통과한다.
5. 바운딩 박스 리그레션을 적용하여 박스의 위치를 조정한다.

## 1. Region Proposal

### Selective search

![selective search][selective search]

Region Proposal 중 고전적인 방법.  
이후 Neural Network가 end-to-end로 학습할 수 있게 발전되었으니 간단하게만 짚고 넘어간다.

구체적인 정보 : [라온피플](https://m.blog.naver.com/laonple/220918802749)

## 2. Feature Extraction

저자들은 이미지넷 데이터(ILSVRC2012 classification)로 미리 학습된 CNN 모델을 가져온 다음, fine tune하는 방식을 취했습니다.  
Classification의 마지막 레이어를 Object Detection의 클래스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로 맞춰주었습니다.

## 3. Classification

CNN Classifier를 쓰지 않고 SVM Classifier를 사용하여 별도로 학습시킨다.  
이제는 더 이상 사용되지 않는 기법.

## 4. Non-Maximum suppression

![non-maximum suppression][non-maximum suppresion]

동일한 물체에 여러 개의 박스가 쳐져있다면, 가장 스코어가 높은 박스만 남기고 나머지는 제거.  
가장 confidence가 높은 박스와 IoU가 일정 이상인 박스를 제거.  
논문에서는 IoU가 0.5보다 크면 동일한 물체를 대상으로 한 박스로 판단하고 Non-Maximum suppression 적용.

## 5. Bounding Box Regression

**_하는 이유 : Selective search를 통해서 찾은 박스 위치가 부정확하다. 교정이 필요._**

방법 : 본 논문 **Appendix C. Bounding-box regression 참고**. MSE error에 L2 normalization한 형태의 loss function.

Loss function

$$
\DeclareMathOperator*{\argmin}{argmin}
\textbf{w}_\star = \argmin_{\hat{\textbf{w}}_\star} {\sum_{i}^{N}(t_{\star}^2 - \hat{\textbf{w}}_{\star}^T \phi_5(P^i))^2 + \lambda ||\hat{\textbf{w}}_\star||^2}
$$

문제점 1. Regularization이 중요하다.  
문제점 2. (P, G)를 골라서 training 시킨다. (너무 거리가 먼 것은 학습하지 않음. IoU >= 0.6; Outlier 때문?)

## 결론

속도 및 정확도면에서 기존 Object Detection 분야에서 획기적인 발전을 이루었다.  
초기 모델이라 전통적인 비전 알고리즘도 함께 사용하여 구조가 복잡하다.  
이후 Fast R-CNN, Faster R-CNN을 같은 저자가 모델을 개선하여 내놓았다.

<!-- Reference -->

[r-cnn model]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbJaTYc%2FbtqANCZbqeK%2FYilKOm42aNYvPcWIjYxCdK%2Fimg.png
[r-cnn model2]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbdmFi2%2FbtqAQ38E2v3%2FJMXznsWZsX3YQAuTkKtpWK%2Fimg.png
[map]: https://better-today.tistory.com/3
[selective search]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FSRNtz%2FbtqAPeQCKIU%2F1JsEHoX4e2bSAgzrgQQCD1%2Fimg.png
[non-maximum suppresion]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fpu1Jo%2FbtqANDX2WUQ%2FdB9pDakTtO57zjZa0CLsa1%2Fimg.png

# Spatial Pyramid Pooling Network

저자 : He et al.  
출처 :

1. [갈아먹는 머신러닝](https://yeomko.tistory.com/14)

Fast R-CNN이 SPPNet의 핵심 아이디어를 많이 차용하였다.  
**_Contribution : 입력 이미지 크기와 상관없이 CNN을 적용할 수 있도록 하는 Spatial Pyramid Pooling 기법을 제안하였다._**

## 핵심 아이디어

![][main idea of sppnet]

기존의 CNN 아키텍쳐들은 convolution layer가 입력 사이즈에 관계 받지 않고 sliding window 형식으로 작동함에도 불구하고, classification 부분의 fully connected layer(fc-layer) 때문에 입력 사이즈를 제한할 수 밖에 없었다. 이 과정에서 원본 사진을 crop이나 warp 해야하기 때문에 원본 이미지에서 정보 손실이 있었다.

저자는 이 문제를 해결하기 위해서 먼저 convolutional layer들을 입력 이미지의 크기를 자유롭게 하여 통과시키고, **_이후 fc-layer 통과 전에 크기를 맞춰주는 pooling을 적용_**한다는 아이디어를 제안합니다.

## SPPNet Architecture

![][sppnet learning structure]

1. 전체 이미지에 Selective Search 적용하여 RoI(Region of Interest) 추출.
2. 전체 이미지를 학습된 CNN을 통과시켜 Feature map을 추출.
3. 크기와 비율이 다양한 RoI에 SPP를 적용하여 고정된 크기의 feature vector 추출.
4. fc-layer를 통과.
5. Binary SVM Classifier
6. Bounding box regressor

5, 6은 R-CNN과 같은 과정인듯함.

## Spartial Pyramid Pooling

![][spatial pyramid pooling structure]

1. 그림에서는 예시로 4x4, 2x2, 1x1 세 가지 영역을 제공한다. (각각이 하나의 피라미드.)
2. bin : 피라미드 한 칸. (e.g. 64x64의 feature map에서 4x4 bin의 크기는 16x16)
3. bin 당 max pooling 후 fc-layer를 위해 이어 붙임.

k : Feature map의 channel 수  
M : bin의 개수  
최종 output은 kM 차원의 벡터가 된다.

실제 실험 : 1x1, 2x2, 3x3, 6x6 총 4개의 피라미드로 SPP 적용

## R-CNN과 비교

![][sppnet vs r-cnn]

**_Q. 모든 RoI가 convolution 후에도 크기가 정확하게 딱 떨어져서 각각에 해당하는 Feature map이 존재하는가 ?_**  
[Fast R-CNN RoI projection](#roi-projection) 참고

### R-CNN

2000개의 RoI를 각각 CNN에 통과시켜서 계산하므로 계산량이 굉장히 많아서 느리다.  
또한, RoI를 crop하고 warp하는 과정이 있다. 이미지를 왜곡한다.

### SPPNet

한 이미지가 한 번의 CNN을 통과하기 때문에 속도가 훨씬 빠르다.

## 한계점

1. end-to-end 방식이 아니기 때문에 여러 단계가 필요하다. (fine-tuning, SVM training, Bounding Box Regression)
2. 여전히 최종 Classification은 binary SVM, Region Proposal은 Selective Search를 이용한다.
3. fine-tuning시에 SPP를 거치기 이전의 Conv layer들을 학습시키지 못한다. (fc-layer만 학습 가능)
   1. 이유는 Fast R-CNN의 [Train](#train-fast-r-cnn) 참고.

**_이후 Fast R-CNN이 위 한계점들을 대폭 개선한다._**

<!-- reference -->

[main idea of sppnet]: ../R-CNN%20계열/SPPNet/img/Figure%201.png
[sppnet learning structure]: https://i.imgur.com/fuIB1bY.png
[spatial pyramid pooling structure]: ../R-CNN%20계열/SPPNet/img/Figure%203.png
[sppnet vs r-cnn]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc3yBHX%2FbtqAThF1y0Z%2FR6ktlMZrYE9skAkGlJiRQk%2Fimg.png

# Fast R-CNN

저자 : Ross et al.  
출처 :

1. [갈아먹는 머신러닝](https://yeomko.tistory.com/15)
2. [Arun Mohan](https://medium.com/datadriveninvestor/review-on-fast-rcnn-202c9eadd23b)

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

# Faster R-CNN

[Full review](../R-CNN%20계열/Faster%20R-CNN/faster_r-cnn_review.md)

# Metric

## Precision & Recall

출처 : [Better Today](https://better-today.tistory.com/1)

### Confusion Matrix

Binary Classfier 의 Prediction 결과를 2x2 Matrix 로 나타낸 것

![](https://t1.daumcdn.net/cfile/tistory/99EEC8335993B47330)

### Definition of Precision & Recall

$$
Precision = {{TP}\over{TP + FP}} \\
Recall = {{TP}\over{TP+FN}}
$$

이를 해석하면 다음과 같다.

**_Precision(정밀도) : 모델이 참이라고 예측했던 것들 중 실제 참인 비율 / 모델이 찾은 결과 중 실제 맞는 경우_**
**_Recall(재현율/검출율) : 실제 참인 경우 중 모델이 참이라고 예측한 비율 / 실제 찾아야 하는 것들 중 모델이 찾은 경우_**

참고 : Accuracy(정확도) $= {{TP + TN}\over{TP + TN + FP + FN}}$

많은 사람들이 이름과 연결짓기는 어렵다고 인정했다.

### Precision-Recall graph

Precision과 Recall은 일반적으로 trade-off가 존재한다.
예를 들어, Threshold를 낮추어 Recall을 높이면 오검출이 늘어날 것이고 이는 Precision 감소로 이어진다.

![][precision recall graph]

<!-- reference -->

[precision recall graph]: https://t1.daumcdn.net/cfile/tistory/214932335869F08E38

## mAP (mean Average Precision)

출처 :

1. [Better Today](https://better-today.tistory.com/3)
2. [Jonathan Hui](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)
3. [다크 프로그래머](https://darkpgmr.tistory.com/162)

### AP (Average Precision)

Precision-Recall graph에서 아래 면적.

![][ap graph]

<!-- reference -->

[ap graph]: https://t1.daumcdn.net/cfile/tistory/220E10365869F5CA34

### Definition of mAP

**_1개의 object당 1개의 AP 값을 구하고, 여러 object-detector 에 대해서 mean 값을 구한 것._**

### 정리

장점

1. 인식 threshold에 의존성 없이 성능 평가가 가능하다.
2. mAP 평가를 통해 최적 threshold를 정할 수 있다.

단점

1. 굉장히 느리다. (Threshold 0 이상의 box를 추철하고 정렬하므로.)

<!-- TODO question -->

**_꼭 조절할 parameter가 threshold 여야 하는가 ?_**
**_정확히 AP를 한 번 구하는 것을 보고 싶다._**

$$
$$
