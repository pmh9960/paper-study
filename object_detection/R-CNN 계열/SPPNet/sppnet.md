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
