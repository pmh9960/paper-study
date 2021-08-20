# R-CNN

저자 : Ross et al.  
출처 :

1. [갈아먹는 머신러닝](https://yeomko.tistory.com/13)
2. [R-CNN](https://arxiv.org/abs/1311.2524)

**_Contribution : CNN을 사용하여 object detection task의 정확도와 속도를 획기적으로 향상시켰다._**

### Reference

- Metric
  - [Precision & Recall](#precision--recall)
  - [mAP (Mean Average Precision)](#map-mean-average-precision)

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

<!-- Reference -->

[r-cnn model]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbJaTYc%2FbtqANCZbqeK%2FYilKOm42aNYvPcWIjYxCdK%2Fimg.png
[r-cnn model2]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbdmFi2%2FbtqAQ38E2v3%2FJMXznsWZsX3YQAuTkKtpWK%2Fimg.png
[map]: https://better-today.tistory.com/3
[selective search]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FSRNtz%2FbtqAPeQCKIU%2F1JsEHoX4e2bSAgzrgQQCD1%2Fimg.png
[non-maximum suppresion]: https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fpu1Jo%2FbtqANDX2WUQ%2FdB9pDakTtO57zjZa0CLsa1%2Fimg.png
