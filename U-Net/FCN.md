# Fully Convolutional Networks for Semantic Segmentation

[Fully Convolutional Networks for Semantic Segmentation -바이오메디컬랩@모두의연구소](https://modulabs-biomedical.github.io/FCN)

## Semantic Segmentation

Pixel 단위로 어떤 object인지 classification 하는 것.

![](https://modulabs-biomedical.github.io/assets/images/posts/2017-12-21-FCN/fig1.jpg)

_그렇다면 왜 항상 Semantic segmenation을 쓰지 않는가 ?_

## Network architecture

![](https://modulabs-biomedical.github.io/assets/images/posts/2017-12-21-FCN/fig3.jpg)

1.  Feature Extraction  
    일반적인 CNN 사용 (VGG-19)
1.  Feature-level Classification  
    FC 대신 1x1 conv layer
    > 1x1 conv layer의 의미
    >
    > 1. Height, Width는 유지
    > 1. 비선형성 추가
1.  Upsampling (backwards strided convolution)  
    작은 차원에서 큰 차원으로.  
    이 때문에 디테일이 다 뭉개진다.  
    ![](https://modulabs-biomedical.github.io/assets/images/posts/2017-12-21-FCN/fig5.jpg)
1.  Segmentation  
    Skip Combining (Solution)  
     의도적으로 차원이 줄기 전 feature map을 사용한다. (ResNet의 Residual block과 비슷)  
    ![](https://modulabs-biomedical.github.io/assets/images/posts/2017-12-21-FCN/fig9.jpg)
