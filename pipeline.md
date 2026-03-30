# 새 프로젝트 브리핑: hair-inpaint

## 배경

기존 프로젝트 `hair-dit`는 헤어살롱용 AI로, 스케치 기반 헤어스타일 생성 모델입니다.
구조: `sketch → DiT (HairControlNet) → hair patch → composite.py → 얼굴에 합성`

**문제**: hair patch만 생성하고 나중에 composite.py로 붙이는 방식은
경계가 부자연스럽고 파이프라인이 분리되어 품질 한계가 명확합니다.

---

## 새 프로젝트 목표

**hair patch 생성 → full image inpainting으로 전환**

- face는 고정, hair 영역만 sketch에 따라 바뀐 full image를 직접 출력
- composite.py 완전 제거
- SD3.5 Medium + Dual ControlNet 기반

---

## 새 아키텍처

### 핵심 아이디어

기존 논문(GAN 기반)에서는 배경 인코더-디코더를 별도로 만들어
마지막 4개 decoder block에서 결합하는 방식을 사용했습니다.
DiT는 encoder-decoder가 없으므로, 이를 **Dual ControlNet**으로 구현합니다.

### 모델 구조

```
[HairControlNet]                    [FaceControlNet]
sketch (RGB 512×512)                masked_face (RGB 512×512)
+ matte (Gray 512×512)              (hair 영역을 0으로 마스킹한 원본 이미지)
역할: hair 구조/색 생성              역할: hair-face 경계 blending
      (identity 보존은 re-injection이 담당)
       ↓                                   ↓
sketch → VAE encode → sketch_latent   masked_face → VAE encode → face_latent
matte  → MatteCNN  → matte_feat
ctrl_cond = sketch_latent + matte_feat
       ↓                                   ↓
  12블록 (SD3.5 Block 0~11 복사)      6블록 (SD3.5 Block 18~23 복사)
  zero-init output proj               zero-init output proj
       ↓                                   ↓
residuals_hair[0..11]              residuals_face[0..5]
```

> 참고: Qwen-Image ControlNet Inpainting (InstantX) — DiT 기반 inpainting ControlNet에서
> pretrained transformer 마지막 6 double block 복사 방식 사용. 동일한 설계 근거 적용.

### DiT block 주입 전략

논문 수식 `F_i = F_i^h · M_i + F_i^BG · (1 - M_i)` 를 DiT residual 주입에 적용:

```
SD3.5 Block  0~17 : hidden += residuals_hair
                    ← 헤어 구조/형태 결정

SD3.5 Block 18~23 : hidden += residuals_hair * matte_tokens
                            + residuals_face * (1 - matte_tokens)
                    ← 논문 수식 그대로: matte 비율로 hair/face residual 혼합
                    ← hair 영역(M=1): residuals_hair만
                    ← face 영역(M=0): residuals_face만
                    ← 경계(0<M<1): 두 residual 보간
```

```python
# matte_tokens 준비
# matte (B, 1, 512, 512) → downsample → (B, 1, 64, 64) → (B, 1024, 1) 으로 reshape
matte_tokens = F.interpolate(matte, size=(64, 64)).flatten(2).permute(0, 2, 1)
```

### 입출력

- **입력 조건**: sketch (RGB) + matte (Gray) + masked_face (RGB)
- **출력**: full image 512×512 RGB (hair 영역만 새로 생성된 전체 이미지)

### SD2-inpaint와의 유사점

SD2-inpaint: `noisy_latent(16ch) + mask(1ch) + masked_image_latent(16ch) = 33ch`
이 프로젝트: HairControlNet(sketch+matte) + FaceControlNet(masked_face) 로 역할 분리

---

## 학습 파이프라인

논문과 동일하게 **hair 영역(matte)에만 노이즈를 주고 그 부분만 복원**하는 방식.
face 영역은 latent 그대로 유지 → 자동 보존, FaceControlNet은 경계 blending 담당.

```
① 데이터 로드
   GT          = original full image
   masked_face = GT * (1 - matte)   ← FaceControlNet 조건
   sketch, matte = 기존 그대로
   → (sketch, matte, masked_face, GT) 4-tuple

② GT → VAE encode → gt_latent (16ch, 64×64)
   masked_face → VAE encode → face_latent (16ch, 64×64)
   matte_down = downsample(matte)  → (1ch, 64×64)

③ Partial noise: hair 영역만 노이즈
   noisy_latent = gt_latent * (1 - matte_down) + σ*noise * matte_down
   (face 영역은 gt_latent 그대로, hair 영역만 노이즈)

④ HairControlNet(noisy_latent, ctrl_cond) → residuals_hair[0..11]
   FaceControlNet(noisy_latent, face_latent) → residuals_face[0..11]

⑤ SD3.5 Transformer:
   Block  0~11: hidden += residuals_hair
   Block 12~23: hidden += residuals_hair + residuals_face
   → v_pred

⑥ Loss (matte 영역에서만 계산):
   - Phase 1: Flow matching * matte_down
   - Phase 2: Flow matching * matte_down
             + LPIPS(pred * matte, GT * matte)
             + Edge alignment(pred * matte, sketch, w=0.05)
```

---

## 추론 파이프라인

```
얼굴 사진 + 헤어 스케치
    ↓
S2M-Net (SketchHairSalon GAN) → matte 자동 생성
    ↓
face_latent = VAE.encode(얼굴 사진)
masked_face = 얼굴 사진 * (1 - matte)
    ↓
순수 노이즈에서 시작 (hair 영역만)
noisy_latent = face_latent * (1-matte_down) + noise * matte_down

20 steps 반복:
    HairControlNet + FaceControlNet → v_pred
    noisy_latent = scheduler.step(v_pred)
    noisy_latent = noisy_latent * matte_down          ← hair 영역: 모델 생성
                 + face_latent * (1 - matte_down)     ← face 영역: 원본 re-injection
    ↓
VAE decode → full image 출력
  - face: 원본과 픽셀 수준 동일 (identity 완벽 보존)
  - hair: 스케치대로 새로 생성
  - 경계: FaceControlNet이 자연스럽게 blending
```

---

## 학습 데이터

기존 `hair-dit`의 dataset 그대로 사용 (새 프로젝트에서 심볼릭 링크 or 경로 공유):

```
dataset/
  braid/
    img/test/       ← 원본 full person image (GT)
    matte/test/     ← hair 영역 matte mask
    sketch/test/    ← hair sketch
  unbraid/          ← unbraid 스타일 데이터
```

- braid: ~1000장
- unbraid: ~3000장
- 총 ~4000장

### Self-supervised 학습 구성

```
GT           = original full image (braid_XXXX.png)
masked_face  = GT에서 matte 영역을 0으로 마스킹
sketch       = 해당 hair region에서 추출된 스케치
matte        = 기존 matte
→ 모델이 (masked_face + sketch + matte) → GT 복원하도록 학습
```

추가 데이터 수집 없이 재학습 가능.

---

## 기존에서 그대로 가져오는 것들

### MatteCNN 구조 (그대로 유지)

```python
Input:  matte (B, 1, 512, 512)

Conv2d(1→16,  k=3, s=2, p=1) + GroupNorm(4) + SiLU   # 512→256
Conv2d(16→32, k=3, s=2, p=1) + GroupNorm(8) + SiLU   # 256→128
Conv2d(32→16, k=3, s=2, p=1) + GroupNorm(4) + SiLU   # 128→64

Output: matte_feat (B, 16, 64, 64)
```

### HairControlNet 초기화 방식 (그대로 유지)

```python
SD3ControlNetModel.from_transformer(transformer, num_layers=12)
# SD3.5 첫 12블록 가중치 복사, output projection만 zero-init
```

### Sketch 처리 (그대로 유지)

```python
sketch (B, 3, 512, 512) → frozen SD3.5 VAE encode → sketch_latent (B, 16, 64, 64)
ctrl_cond = sketch_latent + matte_feat
```

### Null embedding (그대로 유지)

```python
null_encoder_hidden_states: nn.Parameter (333 × 4096)  # trainable
null_pooled_projections:    nn.Parameter (2048)         # trainable
```

### StrokeColorSampler (Phase 2, 그대로 유지)

각 stroke 영역 → target 이미지의 해당 위치 픽셀 중 무작위 1개 샘플링 → stroke 색 할당.
RGB 값 그대로 양자화로 stroke 구분 (grayscale collision 없음).
Phase 2에서 AppearanceJitter 제거, StrokeColorSampler 활성화.

---

## Data Augmentation

### Phase 1 (Unbraid Pretrain)

| Augmentation | 확률 | 역할 |
|---|---|---|
| SketchColorJitter | p=0.8 | 색 과적합 방지 |
| ThicknessJitter (dilation) | p=0.5 | 선 두께 변화에 강건 |
| MattePerturbation (elastic) | p=0.3 | 경계 흔들림에 강건 |
| AppearanceJitter | p=0.5 | 구조-외관 분리 (target에만 적용) |

### Phase 2 (Braid Fine-tune)

- SketchColorJitter 제거, **StrokeColorSampler** 활성화
- AppearanceJitter 제거 (stroke↔target 색 대응 보존)
- ThicknessJitter, MattePerturbation 유지

---

## 학습 전략 (2단계 커리큘럼)

기존과 동일하게 유지.

### Phase 1: Unbraid Pretrain
- 데이터: unbraid 3,000장
- 목표: hair structure 기초 학습 (braid 패턴 없이 일반 hair)
- Loss: Flow matching (v-prediction, Rectified Flow)
- Augmentation: SketchColorJitter + ThicknessJitter + MattePerturbation + AppearanceJitter

### Phase 2: Braid Fine-tune
- 데이터: braid 1,000장
- 시작점: Phase 1 체크포인트
- Loss: Flow matching + **LPIPS** (matte 영역만) + **Edge alignment** (matte 영역만, w=0.05)
- Augmentation: StrokeColorSampler + ThicknessJitter + MattePerturbation

---

## 기존 프로젝트에서 가져오는 것

| 항목 | 처리 |
|------|------|
| `dataset/` | 심볼릭 링크 or 경로 공유 (필수) |
| `src/` SD3.5 VAE/스케줄러 코드 | boilerplate 재활용 |
| `scripts/infer_custom.py` | 추론 흐름 참고 |
| `checkpoints/` | 사용 불가 (아키텍처 변경) |
| `scripts/composite.py` | 불필요 (파이프라인 제거) |

---

## 학습 환경

- GPU: H100 80GB 1장 권장 (A100 80GB도 가능)
- 예상 학습 시간: H100 기준 약 18~24시간 (4000장, 100k steps)

---

## 구현해야 할 것들

1. **FaceControlNet** 신규 설계
   - 초기화: `SD3ControlNetModel.from_transformer(transformer, num_layers=6, start_layer=18)` + zero-init
   - 입력: masked_face → VAE encode → face_latent (16ch, 64×64)
   - residual 주입: SD3.5 Block 18~23 (마지막 6블록만)

2. **HairControlNet 수정**
   - 초기화: `SD3ControlNetModel.from_transformer(transformer, num_layers=12)` (기존 동일)
   - 출력 타겟 변경: `img * matte` → `img` (full image GT)
   - residual 주입: SD3.5 Block 0~23 전체

3. **데이터 로더 수정**
   - `(sketch, matte, masked_face, GT)` 4-tuple
   - `masked_face = GT * (1 - matte)` 런타임 생성

4. **학습 루프 수정**
   - HairControlNet + FaceControlNet 동시 학습
   - Block  0~17: residuals_hair만
   - Block 18~23: residuals_hair + residuals_face
   - loss: matte 영역에서만 계산

5. **추론 스크립트 신규 작성**
   - 입력: sketch + matte + face image
   - 출력: full image (composite 불필요)

---

## 기존 프로젝트 참고 경로

`\\wsl.localhost\Ubuntu-22.04-D\home\agliotomato\hair-dit\`

- `src/pipeline.py` — SD3.5 + HairControlNet 학습 파이프라인
- `src/model.py` — HairControlNet 구조 (MatteCNN, ControlNet 초기화)
- `configs/` — 학습 설정
- `architecture.md` — 현재 아키텍처 전체 상세 설명 (필독)
