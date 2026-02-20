# GPTQ

## 한 줄 요약
> Hessian 정보를 이용해, 한 weight를 양자화할 때 발생한 오차를 나머지 weight들이 최적으로 보상하게 하는 4bit PTQ 기법.

## 문제 상황

Weight를 FP16 → 4bit로 양자화할 때, 단순 반올림(RTN)은 오차가 너무 크다. 4bit는 16개 값만 표현 가능하므로 오차가 누적되면 모델이 망가진다.

GPTQ의 핵심 아이디어: **하나의 weight를 양자화할 때 발생한 오차를, 아직 양자화하지 않은 나머지 weight들을 조정해서 보상한다.** "얼마만큼, 어느 방향으로" 보상할지를 Hessian이 알려준다.

## 핵심 알고리즘

### Hessian (H)의 의미

Linear layer `y = W @ x`에서, calibration 데이터로 Hessian을 계산한다:

```
H = X @ X.T / n_samples    # shape: [ch_in, ch_in]
```

H[j, k]는 **"입력 feature j와 k가 calibration 데이터에서 얼마나 같이 활성화되는가"**를 뜻한다.

```
H[j, j] = mean(x_j²)    → feature j의 활성화 크기 (민감도)
H[j, k] = mean(x_j·x_k) → feature j와 k의 동시 활성화 정도 (결합도)
```

W의 shape이 `[ch_out, ch_in]`일 때 H의 shape은 `[ch_in, ch_in]`이다. W의 각 행(출력 채널)은 독립적으로 양자화되며, 모두 같은 H를 공유한다. H 계산은 한 번만 하면 된다.

### Hessian Inverse (H_inv)의 의미

H_inv는 H의 역행렬이지만, **각 원소의 역수가 아니다.** 행렬 전체를 뒤집는 것이므로 의미가 달라진다.

```
H_inv[j, j] = weight j의 "자유도"
               → j를 양자화해도 loss가 얼마나 적게 증가하는가
               → 클수록 양자화해도 안전 (보상자가 있거나 활성화가 작음)
               → 작을수록 양자화가 위험 (혼자 버텨야 함)

H_inv[j, k] = weight k가 weight j의 오차를 보상하는 "강도와 방향"
               → 크기: 보상 효율 (클수록 k가 j의 오차를 잘 흡수)
               → 부호: 보상 방향 (음수면 반대 방향으로 조정)
```

직관: x1 ≈ x2 이면 `y ≈ (w1 + w2)·x`이므로, 출력은 w1, w2 개별 값이 아니라 **합**에 의존한다. w1을 올리고 w2를 내려도 출력은 거의 안 변한다. 이것이 "자유도"이고, 이것이 "보상"이다. 독립적인 feature의 weight는 보상자가 없어서 자유도가 낮다.

### 구체적 예시

w1 = 0.73을 q1 = 0.75로 양자화하는 상황 (양자화 오차 = 0.02):

보상 수식을 먼저 정의한다:

```
e = q - w                     # 양자화 오차 (양자화된 값 - 원래 값)
c = e / H_inv[j, j]           # 보상 계수 (오차를 자유도로 스케일링)
W[k] += c × H_inv[j, k]       # k번째 weight 보상
```

**Case 1: 강한 상관 + 큰 활성화** — x1, x2가 항상 같이 크게 활성화

```
H = [[1.0, 0.9],  [0.9, 1.0]]
H_inv = [[5.26, -4.74],  [-4.74, 5.26]]

자유도 = 5.26 (높음)     보상 강도 = -4.74 (강함, 반대 방향)
e = 0.75 - 0.73 = 0.02               # 양자화 오차
c = 0.02 / 5.26 = 0.0038             # 보상 계수 (자유도가 크니까 작음)
w2 += 0.0038 × (-4.74) = -0.018      # w2가 0.018 감소
loss 증가: 0.0000380
```

y ≈ (w1+w2)·x 이므로, w1이 0.02 올라간 걸 w2를 0.018 내려서 거의 상쇄.

**Case 2: 약한 상관 + 큰 활성화** — x1, x2가 가끔만 같이 활성화

```
H = [[1.0, 0.3],  [0.3, 1.0]]
H_inv = [[1.10, -0.33],  [-0.33, 1.10]]

자유도 = 1.10 (낮음)     보상 강도 = -0.33 (약함)
e = 0.02
c = 0.02 / 1.10 = 0.0182             # 자유도가 작으니까 보상 계수가 큼
w2 += 0.0182 × (-0.33) = -0.006      # w2가 0.006 감소
loss 증가: 0.000182  (Case 1의 5배)
```

보상은 되지만 효과가 약함. 대부분의 오차가 그대로 남음.

**Case 3: 강한 상관 + 작은 활성화** — x1, x2가 항상 같이 움직이지만 값이 작음

```
H = [[0.1, 0.09],  [0.09, 0.1]]
H_inv = [[52.63, -47.37],  [-47.37, 52.63]]

자유도 = 52.63 (매우 높음)  보상 강도 = -47.37 (매우 강함)
e = 0.02
c = 0.02 / 52.63 = 0.00038           # 자유도가 매우 크니까 보상 계수 극히 작음
w2 += 0.00038 × (-47.37) = -0.018    # w2가 0.018 감소
loss 증가: 0.0000038  (Case 1의 1/10)
```

활성화 자체가 작으니 weight를 틀려도 출력에 거의 영향 없음. 자유도 극대.

**Case 4: 독립 + 큰 활성화** — x1, x2가 크게 활성화되지만 서로 무관

```
H = [[10, 0],  [0, 10]]
H_inv = [[0.1, 0],  [0, 0.1]]

자유도 = 0.1 (매우 낮음)   보상 강도 = 0 (보상 불가)
e = 0.02
c = 0.02 / 0.1 = 0.2                 # 자유도가 극히 작으니 보상 계수 폭발
w2 += 0.2 × 0 = 0                    # 보상 불가!
loss 증가: 0.002  (Case 1의 50배!)
```

독립이면 아무도 도와줄 수 없다. 양자화에 가장 위험한 케이스.

**요약:**

| 케이스 | H_inv[0,0] (자유도) | w2 보정량 | loss 증가 | 해석 |
|--------|---------------------|----------|----------|------|
| 1. 강한 상관 + 큰 활성화 | 5.26 | -0.018 | 0.0000380 | 보상 잘 됨 |
| 2. 약한 상관 + 큰 활성화 | 1.10 | -0.006 | 0.000182 | 보상 약함 |
| 3. 강한 상관 + 작은 활성화 | 52.63 | -0.018 | 0.0000038 | 애초에 안 중요 |
| 4. 독립 + 큰 활성화 | 0.10 | 0 | 0.002 | 보상 불가, 위험 |

### 보상 원리 정리

w1을 양자화하면 출력에 오차가 발생한다. GPTQ는 이 오차를 아직 양자화하지 않은 weight들을 조정해서 흡수한다.

```
e = q - w                      # 양자화 오차 (양자화로 인한 변화량)
c = e / H_inv[j, j]            # 보상 계수 (오차를 자유도로 스케일링)
W[k] += c × H_inv[j, k]        # k번째 weight 보상
```

**보상 계수 c의 의미**:
- 자유도(H_inv[j,j])가 크면 → c가 작음 → "별 문제 아니니 조금만 보상"
- 자유도가 작으면 → c가 큼 → "심각한 문제, 많이 보상해야 함"

**보상 방향**:
- 양의 상관 feature면 H_inv[j,k] < 0
- w1이 올라갔으면(e > 0) → c > 0 → `c × (음수)` → w2 감소
- 즉, **w1이 올라갔으니 같이 활성화되는 w2를 내려서 보상** → 직관과 일치

### 의사코드 (공식 repo `IST-DASLab/gptq`의 `gptq.py` 기반)

```python
def gptq_quantize(W, H, blocksize=128):
    """
    W: [ch_out, ch_in] — 양자화할 weight 행렬
    H: [ch_in, ch_in]  — Hessian (= X @ X.T / n, calibration 데이터에서 계산)
    """
    H_inv = torch.linalg.cholesky_inv(H)   # [ch_in, ch_in]
    Q = torch.zeros_like(W)

    for col_start in range(0, ch_in, blocksize):
        col_end = min(col_start + blocksize, ch_in)

        W_block = W[:, col_start:col_end].clone()
        H_block = H_inv[col_start:col_end, col_start:col_end]
        C = torch.zeros_like(W_block)       # 보상 계수 누적용

        for j in range(col_end - col_start):
            w = W_block[:, j]
            d = H_block[j, j]                       # 자유도

            q = quantize_to_int4(w)                  # 반올림
            Q[:, col_start + j] = q

            e = q - w                                # 양자화 오차
            c = e / d                                # 보상 계수
            C[:, j] = c

            # 블록 내 보정: 아직 처리 안 한 열들을 즉시 조정
            W_block[:, j+1:] += c.unsqueeze(1) * H_block[j, j+1:].unsqueeze(0)

        # 블록 간 보정: 이 블록에서 쌓인 보상 계수를 나머지 열에 한꺼번에 전파
        W[:, col_end:] += C @ H_inv[col_start:col_end, col_end:]

    return Q
```

> 참고: 공식 repo(`IST-DASLab/gptq`)에서는 `e = w - q` 규약에 `-=` 연산을 사용. 수학적으로 동일하다.

블록 내 보정 vs 블록 간 보정:
- **블록 내** (`W_block[:, j+1:]`): 열 하나 양자화할 때마다 같은 블록의 나머지 열을 즉시 보정
- **블록 간** (`W[:, col_end:]`): 한 블록이 끝나면, 그 블록에서 쌓인 모든 오차를 나머지 전체 열에 행렬곱 한 번으로 전파

원래는 매 열마다 나머지 전체를 보정해야 하지만, 블록으로 묶어서 한꺼번에 처리하는 것이 GPTQ의 속도 트릭이다 (정확도 손실 거의 없이 ~1000배 빠름).

## 효과

| 모델 | 비트수 | Perplexity (WikiText2) | FP16 대비 | 양자화 시간 |
|------|-------|----------------------|----------|-----------|
| LLaMA-7B | 4bit (g128) | ~5.68 | 거의 동일 | ~4분 (1 GPU) |
| LLaMA-13B | 4bit (g128) | ~5.12 | 거의 동일 | ~8분 |
| LLaMA-7B | 3bit | ~6.6 | +0.9 | ~4분 |
| LLaMA-7B | 2bit | ~100+ | 사용 불가 | - |

g128 = group size 128 (128개 weight마다 별도의 scale/zero_point 사용).

## 한계 / 트레이드오프

- **Calibration 데이터 필요**: ~128개 샘플로 H를 계산. 데이터 분포가 실제 사용과 다르면 보상이 최적이 아닐 수 있음
- **Weight-only**: compute-bound 상황(큰 배치, prefill)에서는 속도 이득 제한적
- **2bit 이하에서 붕괴**: 보상할 수 있는 한계를 넘어섬. Phase 3의 QuIP, AQLM이 이걸 해결하려는 시도
- **열 순서 고정**: 가장 민감한 것부터 처리하는 greedy 방식(OBQ)보다 이론적으로 열등하지만, GPU 병렬화가 가능해서 ~1000배 빠름

## vs 비교 (AWQ)

| | GPTQ | AWQ |
|---|------|-----|
| **전략** | 오차를 나머지 weight에 보상 | 중요 weight를 스케일링으로 보호 |
| **필요 정보** | Hessian (weight 간 상호 의존성) | Activation 크기 (어떤 weight가 중요한지) |
| **4bit 정확도** | 매우 좋음 | 매우 좋음 (약간 우위 보고도 있음) |
| **양자화 속도** | 빠름 (~분 단위) | 더 빠름 |

## 자사 quantizer 코드베이스 구현

> OPTQ = GPTQ. 원래 논문 이름이 GPTQ였으나 개정판에서 "Optimal PTQ"로 변경. 커뮤니티에서는 GPTQ가 더 널리 쓰임.

### 파일 구조

```
src/quantizer/OPTQ/
├── optqCore.h/cc      # Hessian 계산 (H = X @ X.T 점진적 누적)
├── optqRunner.h/cc    # 레이어별 OPTQ 인스턴스 관리 및 오케스트레이션
├── utils.h/cc         # 핵심 양자화 알고리즘 (error compensation 루프)
└── wrapper.cc         # 모델 레벨 통합 훅
```

### 호출 스택

```
Model.quantize(calTensorList)
│
├─ 1) registerLayersOPTQ()                    [wrapper.cc]
│     └─ OPTQRunner::registerLayer()          [optqRunner.cc]
│        Config에 따라 Conv/TConv 레이어에 OPTQ 인스턴스 생성
│
├─ 2) 각 calibration 샘플마다:
│     └─ Model::updateHessian()               [wrapper.cc]
│        └─ OPTQRunner::updateHessian()       [optqRunner.cc]
│           └─ OPTQ::addBatch(batch)          [optqCore.cc:15-57]
│              x = unfold(x)                  # Conv: (b,c,h,w) → (c*k*k, b*L)
│              x = x * sqrt(2/(n + batch))    # 정규화
│              H += x @ x.t()                 # 점진적 Hessian 누적
│
├─ 3) adjustHessian()                         [wrapper.cc]
│     └─ OPTQRunner::adjustHessian()          [optqRunner.cc]
│        └─ OPTQ::adjustHessian()             [optqCore.cc]
│           input 양자화 scale 반영 → Hessian을 레이어로 전달
│
└─ 4) 각 레이어에서:
      └─ ConvolutionLayer::updateFusedWeightOPTQ()  [convolution.cc:748-750]
         └─ updateFusedWeightOPTQ_()                [utils.cc:75-197]
            ├─ (a) Dead neuron 처리             [utils.cc:95-100]
            ├─ (b) Activation ordering          [utils.cc:102-109]
            ├─ (c) Damping + Cholesky 역행렬    [utils.cc:111-141]
            ├─ (d) 블록 단위 양자화 + error 보상  [utils.cc:143-185]
            └─ (e) 원래 열 순서 복원             [utils.cc:186-188]
```

### 핵심 코드 매핑

**Hessian 계산** (`optqCore.cc:addBatch`):

```cpp
// 논문: H = X @ X.T / n
void OPTQ::addBatch(const at::Tensor& batch) {
    x = torch::nn::functional::unfold(x, unfoldOption);  // Conv 지원
    x = x * sqrt(2.0 / (mNumSamples + x.size(-1)));      // 정규화
    auto hessianBatch = torch::matmul(x, x.t());          // H += X @ X.T
    mHessian = mHessian * (mNumSamples / (mNumSamples + x.size(-1)))
             + hessianBatch;                               // 가중 누적
}
```

**양자화 + 보상 루프** (`utils.cc:updateFusedWeightOPTQ_`):

```cpp
// 논문 의사코드와 1:1 대응
for (i = 0; i < n_cols; i += blockSize) {       // blockSize=128
    for (j = 0; j < blockSize; j++) {
        q = downresol(w[j], weightBits, scale);  // 양자화
        err = (w - q) / H_inv[j][j];             // 자유도로 나눔
        W[:, j+1:] -= err * H_inv[j, j+1:];      // 블록 내 보상
    }
    W[:, col_end:] -= Err @ H_inv[block, rest];   // 블록 간 보상
}
```

### 논문 대비 자사 구현의 추가 기능

#### (a) Dead Neuron 처리 (`utils.cc:95-100`)

calibration 데이터에서 한 번도 활성화되지 않은 feature를 처리:

```python
dead = (diag(H) == 0)       # 한 번도 활성화 안 된 feature
H[dead, dead] = 1.0          # Cholesky 실패 방지
W[:, dead] = 0.0              # 활성화 안 되니 weight도 0으로
```

이게 없으면 `H[j,j] = 0` → `H_inv[j,j]` 계산 불가 → Cholesky 실패.

#### (b) Activation Ordering (`utils.cc:102-109`, `actOrder=true`)

열을 H 대각이 큰 순서(민감한 열 우선)로 재정렬:

```python
order = argsort(diag(H), descending=True)  # 민감한 열 먼저
W = W[:, order]
H = H[order][:, order]
# ... GPTQ 양자화 ...
W = W[:, inverse_order]                     # 원래 순서 복원
```

민감한 열을 먼저 처리하면 보상에 쓸 수 있는 나머지 열이 많아서 더 효과적. 3bit 이하에서 유의미한 개선.

#### (c) Damping (`utils.cc:111-141`, `percDamp=0.01`)

Hessian이 singular/ill-conditioned일 때 Cholesky 실패를 방지:

```python
damp = percDamp * mean(diag(H))   # 대각 평균의 1%
H += damp * I                      # H' = H + λI
```

모든 feature에 약간의 독립적 활성화를 인위적으로 추가하여 역행렬을 존재하게 만듦. Cholesky 실패 시 damping을 10배 올려서 최대 3회 재시도.

### 설정값 (`config.h:1084-1095`)

```cpp
struct OPTQConfig {
    bool apply = false;       // 기본 비활성화
    bool actOrder = true;     // activation ordering 사용
    int blockSize = 128;      // 논문과 동일
    double percDamp = 0.01;   // 대각 damping 1%
    std::set<std::string> applyLayerList;    // 빈 값 = 전체 Conv/TConv
    std::set<std::string> excludeLayerList;  // 특정 레이어 제외
};
```

### 논문 vs 자사 구현 비교

| 기능 | 논문 GPTQ | 자사 구현 |
|------|----------|----------|
| Activation Ordering | 없음 (후속 업데이트에서 추가) | `actOrder=true` (기본 활성화) |
| Dead Neuron 처리 | 없음 | H 대각 0 감지 → weight 0 설정 |
| Damping | 있음 | `percDamp=0.01`, Cholesky 실패 시 3회 재시도 |
| 대상 레이어 | Linear | Conv/TConv (unfold로 확장) |
| Hessian dtype | FP32 | Ampere GPU면 BF16, 아니면 FP16 |

## 참고 자료
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2022
- 공식 repo: https://github.com/IST-DASLab/gptq
- 선행 연구: Frantar & Alistarh, "Optimal Brain Compression" (OBC/OBQ), 2022
