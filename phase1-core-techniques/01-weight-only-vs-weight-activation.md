# Weight-only vs Weight-Activation Quantization

## 한 줄 요약
> Weight만 양자화하면 메모리 절감 + 정확도 유지, Activation까지 양자화하면 연산 자체가 빨라지지만 outlier로 정확도 위험 — 둘 중 뭘 선택할지는 HW와 사용 시나리오에 달려있다.

## 문제 상황

모델을 양자화할 때 가장 먼저 부딪히는 근본적인 질문:

> Weight만 양자화할 것인가, Activation까지 양자화할 것인가?

CNN 양자화에서는 보통 W+A를 함께 양자화했다(W8A8 등). 그런데 LLM 시대에 와서 **Weight-only quantization**(W4A16 등)이 주류 기법 중 하나로 떠올랐다. 이유는 LLM 추론의 특성(decode 단계의 memory-bound 성질) 때문이다.

## 핵심 알고리즘

두 방식의 추론 과정 비교:

```python
# === Weight-only Quantization (예: W4A16) ===
def weight_only_forward(x_fp16, W_int4, scale, zero_point):
    # Weight는 INT4로 저장되어 있지만, 연산 전에 FP16으로 복원
    W_fp16 = (W_int4 - zero_point) * scale   # dequantize: INT4 → FP16
    output = x_fp16 @ W_fp16.T               # FP16 matmul (연산 자체는 그대로)
    return output
    # 핵심: Weight 메모리가 4배 줄어듦 → 메모리에서 읽는 양이 4배 감소
    # 단점: matmul 자체는 여전히 FP16 연산

# === Weight-Activation Quantization (예: W8A8) ===
def weight_activation_forward(x_fp16, W_int8, w_scale, w_zp):
    # Activation도 즉석에서 INT8로 양자화
    a_scale, a_zp = compute_scale_zeropoint(x_fp16)  # 동적 calibration
    x_int8 = round(x_fp16 / a_scale) + a_zp          # FP16 → INT8

    output_int32 = x_int8 @ W_int8.T                  # INT8 matmul (HW 가속)
    output_fp16 = output_int32 * (a_scale * w_scale)   # dequantize: INT32 → FP16
    return output_fp16
    # 핵심: INT8 matmul은 FP16 대비 2~4배 빠름 (HW 지원 시)
    # 단점: activation 양자화에서 오차 발생 (outlier 문제)
```

**Weight-only가 LLM에서 통하는 직관:**

LLM 추론의 decode 단계(토큰을 하나씩 생성)에서:
- 배치 사이즈가 1이고, 매 스텝마다 거대한 weight 행렬을 메모리에서 읽어야 함
- 연산량 자체는 적고, **weight를 읽는 시간이 병목** (memory-bound)
- Weight를 4bit로 줄이면 → 읽는 양이 4배 줄어듦 → 거의 4배 빨라짐
- 굳이 activation까지 양자화해서 정확도를 희생할 필요가 없음

반면 prefill 단계(프롬프트 전체를 한 번에 처리)에서는:
- 토큰이 많아서 연산량이 큼 → **compute-bound**
- INT8 matmul이 실제로 빠르므로 W+A가 유리

## 효과

| 방식 | 대표 기법 | LLaMA-7B perplexity 영향 | 메모리 절감 | 속도 이득 |
|------|----------|-------------------------|-----------|----------|
| W4A16 | GPTQ, AWQ | FP16 대비 +0.1~0.5 | ~3.5x | decode에서 ~3x (memory-bound) |
| W8A8 | SmoothQuant | FP16 대비 거의 동일 | ~2x | prefill에서 ~1.5x (compute-bound) |
| W4A8 | 최근 연구 | +0.5~2.0 | ~3x | 둘 다에서 이득 |
| W4A4 | 연구 단계 | 상당한 손실 | ~4x | 이론적 최대이나 정확도 문제 |

## 한계 / 트레이드오프

| | Weight-only | Weight-Activation |
|---|------------|-------------------|
| **장점** | 정확도 손실 적음, 구현 단순 | 연산 자체가 빨라짐 (HW 가속) |
| **단점** | compute-bound 구간에서는 속도 이득 제한 | activation outlier로 정확도 하락 위험 |
| **적합한 상황** | 작은 배치, decode 위주, 엣지 | 큰 배치, prefill 위주, 서버 |
| **HW 요구사항** | 특별한 HW 지원 불필요 | INT8/INT4 matmul 유닛 필요 |

**엣지 디바이스 관점 핵심 판단 기준:**
- 자사 NPU가 INT8 matmul을 지원하는가? → 지원하면 W8A8 고려 가치 있음
- 주 사용 시나리오가 decode 위주인가? → 그렇다면 W4A16이 가성비 최고
- 둘 다 지원해야 한다면? → SDK에서 두 경로를 모두 제공하는 게 이상적

## vs 비교

이 주제 자체가 두 접근법의 비교이므로 위 표 참조.
이후 학습할 GPTQ/AWQ(Weight-only 대표)와 SmoothQuant(W+A 대표)에서 각각 구체적으로 다룬다.

## 우리 SDK에서의 의미
(본인 작성 영역)

## 참고 자료
- GPTQ: Frantar et al., "Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2022
- AWQ: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023
- SmoothQuant: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2023
