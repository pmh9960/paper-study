# VGG16

[VGG16 논문 리뷰 -강준영](https://medium.com/@msmapark2/vgg16-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-very-deep-convolutional-networks-for-large-scale-image-recognition-6f748235242a)

# Topic

1. 깊이가 깊은 신경망 모델의 학습 성공 방법

1. 모든 Convolutional layer에서 3x3만을 사용한 이유

# 깊이가 깊은 신경망 모델의 학습 성공 방법 = 모든 Convolutional layer에서 3x3만을 사용한 이유

## 모든 Convolutional layer에서 3x3을 사용했기 때문에 성공했다.

|                 |     7x7 Filter      |     3x3 Filter      |
| :-------------: | :-----------------: | :-----------------: |
|      image      | ![][7x7 filter img] | ![][3x3 filter img] |
| reception field |         7x7         |         7x7         |
|   # of layers   |       1 layer       |      3 layers       |
| # of parameters |     49 = 7 \* 7     |  27 = 3 \* 3 \* 3   |

## Result

- 장점

  1. 모델의 비선형성 증가  
     모델의 비선형성 증가는 모델의 특징 식별 능력을 증가시킨다.

  1. 학습 파라미터수의 감소  
     같은 reception field를 가졌음에도 학습 파라미터의 수가 크게 감소한 것을 알 수 있다.

- 단점  
  Layer가 깊어짐에 따라서 같은 reception field이더라도 더 추상적인 feature map이 출력된다.

# 학습 방법

## 가중치 초기화

한 번에 모든 값을 초기화하여 학습하는 것은 학습 속도 면에서 비효율적이므로 VGG16에서는 조금 다른 방식을 채택하였다.

A, A-LRN, B, C, D, E 순서로 layer를 서서히 늘려가면서 학습하였다.

<div style="text-align:center"><img src="https://miro.medium.com/max/1400/1*gU5m4XO2awEM6Zp4DkirFA.png" width=70%></div>

## 학습 이미지 크기

생략

# 결론

1. AlexNet에 비해 깊이를 깊게 하였더니 이미지 분류 정확도가 높아졌다.

1. VGG-19까지만 layer를 늘렸는데, 오차율이 19 layers에서 수렴하였다. 학습 데이터 셋이 크다면 더 깊은 layer도 학습할 수 있다.

<!-- reference link or img -->

[7x7 filter img]: https://miro.medium.com/max/1400/1*Cb8p7EzcWYDHUzMBYI-yyw.png
[3x3 filter img]: https://miro.medium.com/max/1400/1*E9DiwjWyLU-aQU-knOtv3g.png
