# Learning Deconvolution Network for Semantic Segmentation

[Learning Deconvolution Network for Semantic Segmentation -바이오메디컬랩@모두의연구소](https://modulabs-biomedical.github.io/Learning_Deconvolution_Network_for_Semantic_Segmentation)

## FCN의 문제점 지적

1. 크기에 약하다.  
   큰 물체는 파편화되고, 작은 물체는 배경으로 무시하는 경향이 있다.
1. 디테일에 약하다.  
   deconvolution이 너무 단순하다.

## Deconvolution을 개선

![](https://modulabs-biomedical.github.io/assets/images/posts/2018-01-03-Learning_Deconvolution_Network_for_Semantic_Segmentation/fig2.jpg)

Convolution한 만큼 Deconvolution하면 된다.

1. Unpooling & Deconvolution

   ![](https://modulabs-biomedical.github.io/assets/images/posts/2018-01-03-Learning_Deconvolution_Network_for_Semantic_Segmentation/fig3.jpg)

1. Result

   ![](https://modulabs-biomedical.github.io/assets/images/posts/2018-01-03-Learning_Deconvolution_Network_for_Semantic_Segmentation/fig4.jpg)

   (b) -> (c) : Unpooling  
   (c) -> (d) : Deconvolution

_그렇다고 줄어든 차원이 제대로 복구가 되겠는가?_

## 특별한 학습 방식

생략

## 결론

FCN의 문제점을 보완한다.

하지만 FCN이 잘 해결하는 문제를 실수할 때가 있다.

둘을 앙상블하여 `Conditional random field`로 후처리하면 두 모델보다 뛰어난 모델이 나올 것이라고 예상.
