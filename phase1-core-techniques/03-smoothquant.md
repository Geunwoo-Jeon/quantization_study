# SmoothQuant

## 한 줄 요약
> Activation의 채널별 outlier를 수학적 등가 변환으로 weight 쪽에 넘겨서, W8A8 양자화를 가능하게 하는 기법.

## 문제 상황

Weight-only 양자화(W4A16)는 메모리만 줄이고 연산 속도는 향상시키지 못한다. 실제 **연산 속도**를 높이려면 weight와 activation **둘 다** INT8로 양자화하여 INT8 GEMM을 사용해야 한다(W8A8).

그런데 LLM의 activation에는 **특정 채널에 outlier가 집중**되는 문제가 있다:

```
hidden dim = 4096인 Transformer에서:
- 대부분 채널: 값 범위 [-1, 1]
- 특정 채널 몇 개: 값 범위 [-100, 100]
```

이를 per-tensor INT8 양자화하면:

```
scale = 100 / 127 ≈ 0.787
→ [-1, 1] 범위인 채널은 겨우 2 levels만 사용
→ 대다수 채널의 정보가 날아감
```

Per-channel activation quantization은 이론적으론 가능하지만, 행렬곱 `Y = X @ W`에서 X의 column(채널) 방향은 reduction dimension이라 GEMM 커널이 효율적으로 지원하지 못한다.

**결론**: naive한 방법으로는 LLM에서 W8A8이 불가능했다.

## 핵심 알고리즘

### 핵심 아이디어

> "Activation이 양자화하기 어렵고, Weight는 쉽다. 그러면 어려움을 weight 쪽으로 옮기자."

### 등가 변환

행렬곱 `Y = XW`에 채널별 스케일링 벡터 **s**를 끼워넣는다:

```
Y = XW = (X @ diag(s)⁻¹) @ (diag(s) @ W)
         ─────────────────   ──────────────
          X_smooth (쉬워짐)    W_smooth (약간 어려워짐)
```

수학적으로 완전히 동일한 변환이다. 출력 Y는 변하지 않는다.

- `X_smooth = X / s` → outlier 채널을 s로 나눠서 줄인다
- `W_smooth = W * s` → weight에 s를 곱해서 그 어려움을 흡수시킨다

Weight는 원래 분포가 고르니까, 약간의 어려움을 더 떠안아도 괜찮다.

### 스케일링 팩터 s 결정

```
s[j] = max|X[:, j]|^α  /  max|W[j, :]|^(1-α)
```

- `α = 1.0` → activation의 어려움을 전부 weight로 이전 (weight가 너무 힘들어짐)
- `α = 0.0` → 아무것도 안 함
- **`α = 0.5`** → 반반 나눠 갖기 → 대부분의 모델에서 잘 동작

### Python 의사코드

```python
import numpy as np

def smooth_quant(X_sample, W, alpha=0.5):
    """
    X_sample: calibration data로부터 얻은 activation [num_tokens, channels]
    W: 해당 레이어의 weight [channels, out_channels]
    alpha: migration strength (0~1)
    """
    # 1. 채널별 activation 최댓값 (calibration data에서)
    act_max = np.max(np.abs(X_sample), axis=0)  # [channels]

    # 2. 채널별 weight 최댓값
    weight_max = np.max(np.abs(W), axis=1)      # [channels]

    # 3. 스케일링 팩터 계산
    s = (act_max ** alpha) / (weight_max ** (1 - alpha))

    # 4. Weight에 미리 적용 (오프라인, 한 번만)
    W_smooth = W * s[:, np.newaxis]

    # 5. 추론 시 activation에 적용
    # X_smooth = X / s  → 이전 레이어의 LayerNorm에 흡수 가능!

    return W_smooth, s

# 추론 시:
# X_smooth = X / s                    ← LayerNorm 파라미터에 fuse
# X_q = per_tensor_quantize(X_smooth) ← outlier가 없으니 잘 됨
# W_q = per_tensor_quantize(W_smooth) ← 약간 어려워졌지만 여전히 잘 됨
# Y = int8_gemm(X_q, W_q)            ← INT8 GEMM으로 속도 향상
```

### LayerNorm에 s 흡수 (zero-cost 트릭)

Transformer에서 Linear 레이어 앞에는 항상 LayerNorm이 있다:

```
LayerNorm(X) → Linear(W)
```

LayerNorm 출력은 `gamma * norm(X) + beta`이므로:

```python
# SmoothQuant 적용 시:
# X_smooth = X_ln / s = (gamma/s) * norm(X) + (beta/s)
# → gamma_new = gamma / s, beta_new = beta / s 로 교체하면 끝
```

**추가 연산 비용이 0이다.** 오프라인에서 LayerNorm 파라미터만 바꿔놓으면 된다.

## 효과

### 정확도

| 모델 | FP16 PPL | SmoothQuant W8A8 PPL | 차이 |
|------|---------|---------------------|------|
| OPT-175B | 8.34 | 8.42 | +0.08 |
| OPT-66B | 9.34 | 9.45 | +0.11 |
| OPT-30B | 9.56 | 9.67 | +0.11 |
| BLOOM-176B | 8.11 | 8.24 | +0.13 |

SmoothQuant 없이 naive W8A8을 적용하면 perplexity가 수십~수백 올라가거나 발산한다.

### 속도 및 메모리

- NVIDIA A100에서 FP16 대비 **최대 1.56x 속도 향상**
- 메모리 **약 2x 절감** (FP16 → INT8)
- INT8 Tensor Core 활용으로 실제 throughput 증가

## 한계 / 트레이드오프

| 한계 | 설명 |
|------|------|
| INT8까지만 | W4A4 같은 더 낮은 비트로의 확장은 이 기법만으론 부족 |
| Calibration 필요 | activation 통계를 위해 소량의 calibration 데이터 필요 (보통 128~512 샘플) |
| α 튜닝 | 모델마다 최적 α가 다를 수 있음 (0.5가 기본, GLM 등 극단적 outlier 모델은 0.75 필요) |
| 극단적 outlier | GLM-130B 등 outlier가 너무 극심한 모델은 smoothing만으로 부족할 수 있음 |
| 구조 의존 | LayerNorm fuse 트릭은 Transformer 구조 전제. 다른 아키텍처에서는 추가 설계 필요 |

## vs 비교

### SmoothQuant vs GPTQ

| | GPTQ | SmoothQuant |
|---|------|-------------|
| 양자화 대상 | Weight만 (W4A16) | Weight + Activation (W8A8) |
| 비트 수 | 4bit (혹은 3bit) | 8bit |
| 메모리 절감 | ~4x | ~2x |
| 연산 속도 향상 | 없음 (FP16 연산) | 있음 (INT8 GEMM) |
| 핵심 문제 | Weight 양자화 오차 최소화 | Activation outlier 처리 |
| 접근법 | Hessian 기반 오차 보상 | 등가 변환으로 난이도 이전 |
| 칩 요구 | FP16 연산 유닛 | INT8 연산 유닛 (Tensor Core 등) |

**핵심 차이**: 목적이 다르다.
- **메모리가 병목**이면 → GPTQ (4bit로 더 많이 압축)
- **연산 속도가 병목**이면 → SmoothQuant (INT8 GEMM으로 실제 가속)
- 실무에서는 둘을 조합하기도 한다 (예: W4A8)

## 우리 SDK에서의 의미

(본인 작성 영역)

## 참고 자료

- 논문: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2023
- 공식 repo: https://github.com/mit-han-lab/smoothquant
