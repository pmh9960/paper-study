# Review paper list

## Table of Contents

1. [R-CNN](#r-cnn)

# R-CNN

[갈아먹는 머신러닝](https://yeomko.tistory.com/13?category=888201)

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
