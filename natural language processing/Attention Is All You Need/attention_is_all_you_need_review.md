# Attention Is All You Need

- **Author:** A. Vaswani et al. (Google Brain, Google Research)
- **Reference:**
  - [Original Paper](1706.03762.pdf)
  - [Wikidocs Attention Mechanism](https://wikidocs.net/22893)
  - [Wikidocs Transformer](https://wikidocs.net/31379)

## Contribution

0. Attention mechanism만을 이용한 Transformer 모델 제안. _(Recurrence와 convolutions을 제공한다?)_
1. Input과 output간의 global dependencies를 이용하기 위해서 Recurrence를 피하고 attention mechanism만을 이용하였다.
2. Parallelization이 크게 높아져서 속도 향상에 도움이 되었다.
3. 높은 translation quality를 가졌음에도 학습 또한 효율적이다. (8개의 GPU로 12시간만 학습하였다.)
<!-- TODO Self attention도 들어가지 않나 ? -->

# 1. Introduction

## RNN 계열의 한계

Language modeling과 machine translation 분야에서 기존에는 **_recurrent하게 encoder와 decoder의 구조로_** 이루어진 RNN 계열의 모델 (RNN, LSTM, GRU...)등이 유행하였다. _(Language modeling이란?)_

간단하게 설명하면 $h_{t-1}$과 $x_t$를 이용하여 $h_t$를 생성하는 모델이다. 하지만 이러한 RNN 계열의 고질적인 문제는

1. 한 벡터에 모든 문장의 정보를 담기 힘들다. (한 줄로 구성된 encoder-decoder의 문제)
2. Vanishing Gradient
3. 긴 길이에 대해서 메모리의 부족 (긴 길이의 예제 batching 제한)
4. 계산과정의 비효율 ($h_{t-1}$이 나올 때 까지 $h_t$를 계산할 수 없어 기다려야 함)

이다.

이를 고치기 위해 많은 노력을 기울였지만 구조적으로 불가능했다.

## Attention mechanism

Attention mechanism은 RNN의 한계 중 중요한 다음 두 문제를 상당 부분 해결하였다.

1. 한 벡터에 모든 문장의 정보를 담기 힘들다.
2. Vanishing Gradient

이 두 문제는 특히 문장이 길어질 수록 더 심각한데 이를 해결하므로써 input, output의 길이에 관계 없이 강력한 모델을 만들 수 있게 되었다.

- **More Info:** [Wikidocs Attention Mechanism](https://wikidocs.net/22893) <!-- TODO 나중에 정리 예정-->

## Transformer

본 논문에서 제안하는 Transformer는 recurrence를 모델 구조에서 제거하고 대신 input과 output의 global dependencies를 그리는([Attention Visualization](#appendix-attention-visualization)) attention mechanism만을 이용하였다.

1. Attention mechanism이 1, 2를 해결해준다.
2. Transformer 구조가 computation을 줄이고 병렬화를 가능하게 한다. (3, 4 해결)

# 2. Background

# 3. Model Architecture

# 4. Why Self-Attention

# 5. Training

# 6. Results

# 7. Conclusion

# Appendix Attention Visualization

![](imgs/fig3_attention_visualization.png)
