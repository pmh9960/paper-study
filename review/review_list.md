# Review paper list

## Table of Contents

1. [R-CNN](#r-cnn)
2. [Spatial Pyramid Pooling Network](#spatial-pyramid-pooling-network)

# R-CNN

저자 : Ross et al.  
[갈아먹는 머신러닝](https://yeomko.tistory.com/13?category=888201)

**_Contribution : CNN을 사용하여 object detection task의 정확도와 속도를 획기적으로 향상시켰다._**

## 전체적인 과정

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
Fast R-CNN이 SPPNet의 핵심 아이디어를 많이 차용하였다.  
**_Contribution : 입력 이미지 크기와 상관없이 CNN을 적용할 수 있도록 하는 Spatial Pyramid Pooling 기법을 제안하였다._**

## 핵심 아이디어

![][main idea of sppnet]

기존의 CNN 아키텍쳐들은 convolution layer가 입력 사이즈에 관계 받지 않고 sliding window 형식으로 작동함에도 불구하고, classification 부분의 fully connected layer(fc-layer) 때문에 입력 사이즈를 제한할 수 밖에 없었다. 이 과정에서 원본 사진을 crop이나 warp 해야하기 때문에 원본 이미지에서 정보 손실이 있었다.

저자는 이 문제를 해결하기 위해서 먼저 convolutional layer들을 입력 이미지의 크기를 자유롭게 하여 통과시키고, **_이후 fc-layer 통과 전에 크기를 맞춰주는 pooling을 적용_**한다는 아이디어를 제안합니다.

## 전체적인 과정

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

**_Q. 모든 RoI가 convolution 후에도 크기가 정확하게 딱 떨어져서 각각에 해당하는 Feature map이 존재하는가 ?_** <!-- TODO -->

### R-CNN

2000개의 RoI를 각각 CNN에 통과시켜서 계산하므로 계산량이 굉장히 많아서 느리다.  
또한, RoI를 crop하고 warp하는 과정이 있다. 이미지를 왜곡한다.

### SPPNet

한 이미지가 한 번의 CNN을 통과하기 때문에 속도가 훨씬 빠르다.

## 한계점

1. end-to-end 방식이 아니기 때문에 여러 단계가 필요하다. (fine-tuning, SVM training, Bounding Box Regression)
2. 여전히 최종 Classification은 binary SVM, Region Proposal은 Selective Search를 이용한다.
3. fine-tuning시에 SPP를 거치기 이전의 Conv layer들을 학습시키지 못한다. (fc-layer만 학습 가능)
   1. 왜 ? <!-- TODO -->

**_이후 Fast R-CNN이 위 한계점들을 대폭 개선한다._**

<!-- reference -->

[main idea of sppnet]: ../R-CNN%20계열/SPPNet/img/Figure%201.png
[sppnet learning structure]: https://i.imgur.com/fuIB1bY.png
[spatial pyramid pooling structure]: ../R-CNN%20계열/SPPNet/img/Figure%203.png
[sppnet vs r-cnn]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc3yBHX%2FbtqAThF1y0Z%2FR6ktlMZrYE9skAkGlJiRQk%2Fimg.png
