# Edge AI Quantization Study Roadmap

엣지 AI 반도체 SDK 개발자를 위한 양자화 및 관련 지식 학습 로드맵.

## 학습 배경

- **역할**: 엣지향 AI 반도체 회사, 모델 그래프 조작 및 양자화 SDK 팀
- **핵심 업무**: 그래프 구조가 결정된 상태에서 양자화를 통해 정확도와 속도를 최대화
- **기존 지식**: CNN 양자화(Image Compression Network 양자화 논문), DL 기초(convolution, backpropagation 등), Transformer block 구조, Activation Quantization의 어려움(outlier 문제 등)
- **학습 방식**: 출퇴근 시간 + 짬 시간 활용

## 학습 진행 방법

> **새 세션에서 이어서 공부할 때**: 이 README를 읽은 뒤, 진행 상황 표에서 다음 미완료 항목을 찾아 "N번 주제 공부하자"라고 말하면 된다.

### 수업 방식

1. **Claude가 설명**: 각 주제에 대해 아래 구조로 설명
   - **문제 상황**: 이 기법이 왜 필요한가? 어떤 문제를 해결하려고 나왔는가?
   - **핵심 알고리즘**: 수학적 증명 없이, 직관적으로 왜 되는지 설명
   - **효과**: 어떤 조건에서 얼마나 좋아지는가?
   - **한계 / 트레이드오프**: 언제 안 되는가? 어떤 조건에서 무너지는가?
   - **vs 비교**: 유사 기법과의 차이 (해당 시)
2. **Q&A**: 이해 안 되는 부분에 대해 보충 설명
3. **기록**: 대화 내용을 바탕으로 해당 주제의 md 파일 생성 (해당 phase 폴더에 저장)

### 설명 원칙

- 수학적 증명이나 수학 이론 기반 설명은 **불필요**
- 대신 **"왜 효과적인지"가 직관적으로 와닿아야** 함
- 실제 숫자(perplexity, 비트수, 속도 비교 등)를 가능하면 포함

## 진행 상황

| # | 주제 | 상태 |
|---|------|------|
| 01 | Weight-only vs Weight-Activation 트레이드오프 | |
| 02 | GPTQ | |
| 03 | SmoothQuant | |
| 04 | AWQ | |
| 05 | Block / Sub-channel Quantization | |
| 06 | Roofline Model | |
| 07 | KV Cache Quantization | |
| 08 | FP8 (E4M3 vs E5M2) | |
| 09 | Product Quantization (VQ 기초) | |
| 10 | QuIP / QuIP# | |
| 11 | AQLM | |
| 12 | GPTVQ | |
| 13 | SqueezeLLM | |
| 14 | GQA (Grouped Query Attention) | |
| 15 | KV Cache 구조 + Paged Attention | |
| 16 | RoPE | |
| 17 | SwiGLU / GeGLU | |
| 18 | MoE (Mixture of Experts) | |
| 19 | Weight Packing / Bit Packing | |
| 20 | DMA와 Double Buffering | |
| 21 | RMSNorm vs LayerNorm | |
| 22 | Speculative Decoding | |

> 상태: (빈칸)=미시작, `진행중`=학습 중, `완료`=md 파일 작성 완료

## Repo 구조

```
study/
├── README.md                          # 전체 목차, 로드맵, 진행 상황 (이 파일)
├── phase1-core-techniques/            # LLM 양자화 핵심 기법
│   ├── 01-weight-only-vs-weight-activation.md
│   ├── 02-gptq.md
│   ├── 03-smoothquant.md
│   └── 04-awq.md
├── phase2-practical-essentials/       # 업무 직결 필수 지식
│   ├── 05-block-subchannel-quantization.md
│   ├── 06-roofline-model.md
│   ├── 07-kv-cache-quantization.md
│   └── 08-fp8-formats.md
├── phase3-vector-quantization/        # 극저비트 + VQ 계열 최전선
│   ├── 09-product-quantization.md
│   ├── 10-quip.md
│   ├── 11-aqlm.md
│   ├── 12-gptvq.md
│   └── 13-squeezellm.md
├── phase4-architecture-context/       # 양자화 대상 아키텍처 이해
│   ├── 14-gqa.md
│   ├── 15-kv-cache-paged-attention.md
│   ├── 16-rope.md
│   ├── 17-swiglu-geglu.md
│   ├── 18-moe.md
│   └── 19-weight-packing.md
└── phase5-hardware-mechanisms/        # 하드웨어 메커니즘 (후순위)
    ├── 20-dma-double-buffering.md
    ├── 21-rmsnorm-vs-layernorm.md
    └── 22-speculative-decoding.md
```

## 학습 로드맵 상세

### Phase 1 — LLM 양자화 핵심 기법

이 Phase를 마치면 **현재 LLM 양자화의 전체 지형도**가 그려진다.

| # | 주제 | 핵심 논문 | 핵심 아이디어 |
|---|------|----------|-------------|
| 01 | Weight-only vs Weight-Activation 트레이드오프 | (개별 논문 없음, 02~04 읽으며 정리) | "어디까지 양자화할 것인가"의 근본 선택 |
| 02 | GPTQ | Frantar et al., 2022 | Hessian 기반 layer-wise PTQ, OBQ의 확장 |
| 03 | SmoothQuant | Xiao et al., 2023 | Activation outlier를 weight로 이전, W8A8 달성 |
| 04 | AWQ | Lin et al., 2023 | Salient weight 보호, activation-aware 4bit |

### Phase 2 — 업무 직결 필수 지식

양자화의 품질과 속도에 직접 영향을 미치는 실무 지식.

| # | 주제 | 핵심 논문/자료 | 핵심 아이디어 |
|---|------|--------------|-------------|
| 05 | Block / Sub-channel Quantization | AWQ, GPTQ 내 group-wise 섹션 + 자사 칩 스펙 | 양자화 granularity가 정확도와 HW 효율 모두에 직결 |
| 06 | Roofline Model | Williams et al., 2009 | 연산 병목 vs 메모리 병목 판단 프레임워크 |
| 07 | KV Cache Quantization | Hooper et al., "KVQuant", 2024 | LLM 추론 메모리 병목의 주범을 양자화로 해결 |
| 08 | FP8 (E4M3 vs E5M2) | Micikevicius et al., 2022 (NVIDIA/ARM/Intel) | 차세대 수치 포맷, INT8과 다른 트레이드오프 |

### Phase 3 — 극저비트 + Vector Quantization 계열

2bit 이하 극저비트 양자화의 연구 최전선. Codebook/VQ 기반 접근.

| # | 주제 | 핵심 논문 | 핵심 아이디어 |
|---|------|----------|-------------|
| 09 | Product Quantization (VQ 기초) | Jegou et al., 2011 | VQ의 기본 프레임워크, 이후 논문들의 전제 지식 |
| 10 | QuIP / QuIP# | Chee et al., 2023 / Tseng et al., 2024 | Incoherence processing + lattice codebook (E8) |
| 11 | AQLM | Egiazarian et al., 2024 | Additive Quantization, 다중 codebook 조합 |
| 12 | GPTVQ | van Baalen et al., 2024 (Qualcomm) | VQ를 LLM weight에 직접 적용, 엣지 HW 관점 |
| 13 | SqueezeLLM | Kim et al., 2023 | Sensitivity 기반 mixed-precision + sparse outlier 분리 |

### Phase 4 — 양자화 대상 아키텍처 이해

양자화 전략 수립 시 대상 모델의 구조를 이해하기 위한 맥락 지식.

| # | 주제 | 핵심 논문 | 양자화와의 관계 |
|---|------|----------|---------------|
| 14 | GQA (Grouped Query Attention) | Ainslie et al., 2023 | Head 구조가 per-head 양자화 granularity에 영향 |
| 15 | KV Cache 구조 + Paged Attention | Kwon et al., "vLLM", 2023 | KV Cache Quantization(#07)의 전제 지식 |
| 16 | RoPE | Su et al., 2021 | Attention score 분포에 영향 → 양자화 민감도 |
| 17 | SwiGLU / GeGLU | Shazeer, 2020 | FFN activation 분포 특성이 양자화에 영향 |
| 18 | MoE (Mixture of Experts) | Shazeer et al., 2017 / Fedus et al., 2022 | Expert별 분포 차이 → expert별 양자화 전략 필요 |
| 19 | Weight Packing / Bit Packing | (독립 논문 없음, GPTQ/AWQ CUDA kernel 코드) | 저비트 가중치의 실제 메모리 저장 및 읽기 방식 |

### Phase 5 — 하드웨어 메커니즘 (후순위)

직접 업무는 아니지만 알면 설계 판단에 깊이가 생기는 지식.

| # | 주제 | 핵심 논문/자료 | 비고 |
|---|------|--------------|------|
| 20 | DMA와 Double Buffering | 자사 칩 매뉴얼 + 컴퓨터 구조 교재 | 데이터 전송-연산 오버랩 |
| 21 | RMSNorm vs LayerNorm | Zhang & Sennrich, 2019 | Norm 전후 activation 분포 차이 |
| 22 | Speculative Decoding | Leviathan et al., 2023 | 양자화와 조합 시 시너지/충돌 |

## 학습 원칙

- Phase 1~2가 **최우선** (약 2~3주 목표)
- Phase 3은 연구 관심사에 따라 깊이 조절
- Phase 4~5는 필요할 때 참조용으로 학습
- 공부한 내용은 커밋으로 기록하여 진행 상황을 추적

## 각 파일의 템플릿

```markdown
# [주제명]

## 한 줄 요약
>

## 문제 상황
어떤 문제를 해결하려고 나온 기법인가?

## 핵심 알고리즘
(수식 최소화, 직관적 설명 위주)

## 효과
어떤 조건에서 얼마나 좋아지는가?

## 한계 / 트레이드오프
언제 안 되는가? 어떤 조건에서 무너지는가?

## vs 비교
유사 기법 대비 장단점 (해당 시)

## 우리 SDK에서의 의미
(본인 작성 영역)

## 참고 자료
- 블로그, 영상, 코드 링크 등
```
