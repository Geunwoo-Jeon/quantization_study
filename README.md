# Edge AI Quantization Study Roadmap

엣지 AI 반도체 SDK 개발자를 위한 양자화 및 관련 지식 학습 로드맵.

## 학습 배경

- **역할**: 엣지향 AI 반도체 회사, 모델 그래프 조작 및 양자화 SDK 팀
- **핵심 업무**: 그래프 구조가 결정된 상태에서 양자화를 통해 정확도와 속도를 최대화
- **기존 지식**: CNN 양자화(Image Compression Network), DL 기초, Transformer 기본 구조
- **학습 방식**: 출퇴근 시간 + 짬 시간 활용

## Repo 구조

```
study/
├── README.md                          # 전체 목차 및 로드맵 (이 파일)
├── phase1-core-techniques/            # 들어봤다 → 안다 (LLM 양자화 핵심)
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
│   └── 18-weight-packing.md
└── phase5-hardware-mechanisms/        # 하드웨어 메커니즘 (후순위)
    ├── 19-dma-double-buffering.md
    ├── 20-rmsnorm-vs-layernorm.md
    └── 21-speculative-decoding.md
```

## 학습 로드맵

### Phase 1 — "들어봤다"를 "안다"로 (LLM 양자화 핵심 기법)

이미 감이 있는 주제들. 논문 1편 + 구현 코드 훑으면 잡힌다.
이 Phase를 마치면 **현재 LLM 양자화의 전체 지형도**가 그려진다.

| # | 주제 | 현재 상태 | 핵심 논문 | 핵심 아이디어 |
|---|------|----------|----------|-------------|
| 01 | Weight-only vs Weight-Activation 트레이드오프 | 들어봤다 | (개별 논문 없음, 02~04 읽으며 정리) | "어디까지 양자화할 것인가"의 근본 선택 |
| 02 | GPTQ | 들어봤다 | Frantar et al., 2022 | Hessian 기반 layer-wise PTQ, OBQ의 확장 |
| 03 | SmoothQuant | 들어봤다 | Xiao et al., 2023 | Activation outlier를 weight로 이전, W8A8 달성 |
| 04 | AWQ | 들어봤다 | Lin et al., 2023 | Salient weight 보호, activation-aware 4bit |

### Phase 2 — 업무 직결 필수 지식

양자화의 품질과 속도에 직접 영향을 미치는 실무 지식.

| # | 주제 | 현재 상태 | 핵심 논문/자료 | 핵심 아이디어 |
|---|------|----------|--------------|-------------|
| 05 | Block / Sub-channel Quantization | 모른다 | AWQ, GPTQ 내 group-wise 섹션 + 자사 칩 스펙 | 양자화 granularity가 정확도와 HW 효율 모두에 직결 |
| 06 | Roofline Model | 모른다 | Williams et al., 2009 | 연산 병목 vs 메모리 병목 판단 프레임워크 |
| 07 | KV Cache Quantization | 모른다 | Hooper et al., "KVQuant", 2024 | LLM 추론 메모리 병목의 주범을 양자화로 해결 |
| 08 | FP8 (E4M3 vs E5M2) | 들어봤다 | Micikevicius et al., 2022 (NVIDIA/ARM/Intel) | 차세대 수치 포맷, INT8과 다른 트레이드오프 |

### Phase 3 — 극저비트 + Vector Quantization 계열

2bit 이하 극저비트 양자화의 연구 최전선. Codebook/VQ 기반 접근.

| # | 주제 | 현재 상태 | 핵심 논문 | 핵심 아이디어 |
|---|------|----------|----------|-------------|
| 09 | Product Quantization (VQ 기초) | (복습) | Jegou et al., 2011 | VQ의 기본 프레임워크, 이후 논문들의 전제 지식 |
| 10 | QuIP / QuIP# | 모른다 | Chee et al., 2023 / Tseng et al., 2024 | Incoherence processing + lattice codebook (E8) |
| 11 | AQLM | 모른다 | Egiazarian et al., 2024 | Additive Quantization, 다중 codebook 조합 |
| 12 | GPTVQ | 모른다 | van Baalen et al., 2024 (Qualcomm) | VQ를 LLM weight에 직접 적용, 엣지 HW 관점 |
| 13 | SqueezeLLM | 모른다 | Kim et al., 2023 | Sensitivity 기반 mixed-precision + sparse outlier 분리 |

### Phase 4 — 양자화 대상 아키텍처 이해

양자화 전략 수립 시 대상 모델의 구조를 이해하기 위한 맥락 지식.

| # | 주제 | 현재 상태 | 핵심 논문 | 양자화와의 관계 |
|---|------|----------|----------|---------------|
| 14 | GQA (Grouped Query Attention) | 기억 안남 | Ainslie et al., 2023 | Head 구조가 per-head 양자화 granularity에 영향 |
| 15 | KV Cache 구조 + Paged Attention | 모른다 | Kwon et al., "vLLM", 2023 | KV Cache Quantization(#07)의 전제 지식 |
| 16 | RoPE | 기억 안남 | Su et al., 2021 | Attention score 분포에 영향 → 양자화 민감도 |
| 17 | SwiGLU / GeGLU | 모른다 | Shazeer, 2020 | FFN activation 분포 특성이 양자화에 영향 |
| 18 | Weight Packing / Bit Packing | 모른다 | (독립 논문 없음, GPTQ/AWQ CUDA kernel 코드) | 저비트 가중치의 실제 메모리 저장 및 읽기 방식 |

### Phase 5 — 하드웨어 메커니즘 (후순위)

직접 업무는 아니지만 알면 설계 판단에 깊이가 생기는 지식.

| # | 주제 | 현재 상태 | 핵심 논문/자료 | 비고 |
|---|------|----------|--------------|------|
| 19 | DMA와 Double Buffering | 모른다 | 자사 칩 매뉴얼 + 컴퓨터 구조 교재 | 데이터 전송-연산 오버랩 |
| 20 | RMSNorm vs LayerNorm | 들어봤다 | Zhang & Sennrich, 2019 | Norm 전후 activation 분포 차이 |
| 21 | Speculative Decoding | 들어봤다 | Leviathan et al., 2023 | 양자화와 조합 시 시너지/충돌 |

## 학습 원칙

- 각 md 파일에는 **핵심 아이디어, 수식/도식, 논문 요약, 구현 포인트, 업무 연관성**을 기록
- 공부한 내용은 커밋으로 기록하여 진행 상황을 추적
- Phase 1~2가 **최우선** (약 2~3주 목표)
- Phase 3은 연구 관심사에 따라 깊이 조절
- Phase 4~5는 필요할 때 참조용으로 학습

## 각 파일의 템플릿

```markdown
# [주제명]

## 상태
- [ ] 논문/자료 읽기
- [ ] 핵심 아이디어 정리
- [ ] 업무 연관성 메모
- [ ] (선택) 코드/구현 확인

## 핵심 논문
- 제목, 저자, 연도, 링크

## 한 줄 요약
>

## 핵심 아이디어


## 수식 / 도식


## 업무 연관성
이 기법이 우리 SDK/칩에서 어떤 의미를 가지는가?

## 참고 자료
- 블로그, 영상, 코드 링크 등

## 메모

```
