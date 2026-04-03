# hair-dit2

### 기존 hair-dit와의 차이

| 항목 | hair-dit | hair-dit2 |
|------|----------|-----------|
| 출력 | hair patch (마스크 영역만) | full image 512×512 |
| 합성 | composite.py로 후처리 | 불필요 (end-to-end) |
| 기반 모델 | SD3.5 + HairControlNet (1개) | SD3.5 + Dual ControlNet (2개) |
| Face 처리 | 없음 | FaceControlNet이 경계 blending 담당 |
| 경계 품질 | 부자연스러운 하드 합성 | 소프트 blending (matte-weighted residual) |

---

## 전체 아키텍처

```
입력 조건
───────────────────────────────────────────────────────────────────────
sketch (B,3,512,512)  ──────────────────────────────────────────┐
matte  (B,1,512,512)  ────────┐                                 │
                              ↓                                 ↓
                    sketch → VAE encode → sketch_latent (B,16,64,64)
                    matte  → MatteCNN   → matte_feat    (B,16,64,64)
                    matte  → interpolate→ matte_latent  (B, 1,64,64)
                    ctrl_cond = cat([sketch_latent + matte_feat, matte_latent])
                             = (B, 17, 64, 64)
                              ↓
                    ┌──────────────────────┐
                    │    HairControlNet    │
                    │   SD3.5 blocks 0~11  │
                    │   (12 blocks)        │
                    └──────────────────────┘
                              ↓
                    residuals_hair[0..11]   ← 각 (B, 1024, inner_dim)

masked_face (B,3,512,512) ──────────────────────────────────────┐
                              ↓                                 │
                    masked_face → VAE encode → face_latent      │
                                                (B,16,64,64)   │
                              ↓                                 │
                    ┌──────────────────────┐                    │
                    │    FaceControlNet    │                    │
                    │   SD3.5 blocks 18~23 │                    │
                    │   (6 blocks)         │                    │
                    └──────────────────────┘                    │
                              ↓
                    residuals_face[0..5]    ← 각 (B, 1024, inner_dim)

                    ┌───────────────────────────────────────────────────┐
                    │          SD3.5 Transformer (frozen, 24 blocks)    │
                    │                                                   │
                    │  Block  0~17: hidden += residuals_hair[i // 2]   │
                    │                                                   │
                    │  Block 18~23: hidden +=                           │
                    │    residuals_hair[i//2] * matte_tokens            │
                    │  + residuals_face[i-18] * (1 - matte_tokens)     │
                    │                                                   │
                    │  matte_tokens: matte → (B,1024,1) token space    │
                    └───────────────────────────────────────────────────┘
                              ↓
                         v_pred (B, 16, 64, 64)
                              ↓
                    VAE decode → full image (B, 3, 512, 512)
```

---

## 컴포넌트 상세

### 1. HairControlNet (`src/models/hair_controlnet.py`)

SD3.5 첫 12블록을 복사해 초기화한 ControlNet. Hair 구조/색/형태를 담당합니다.

#### MatteCNN

```
Input:  matte (B, 1, 512, 512)

Conv2d(1→16,  k=3, s=2, p=1) + GroupNorm(4)  + SiLU  → (B, 16, 256, 256)
Conv2d(16→32, k=3, s=2, p=1) + GroupNorm(8)  + SiLU  → (B, 32, 128, 128)
Conv2d(32→16, k=3, s=2, p=1) + GroupNorm(4)  + SiLU  → (B, 16,  64,  64)

Output: matte_feat (B, 16, 64, 64)
```

#### SD3ControlNetModel (12 blocks)

```python
SD3ControlNetModel.from_transformer(transformer, num_layers=12)
# SD3.5 blocks 0~11 가중치 복사
# extra_conditioning_channels=1 (default) → pos_embed_input: 17ch 기대
```

#### Null Embeddings (학습 가능 파라미터)

```python
null_encoder_hidden_states: nn.Parameter  # (1, 333, 4096)
null_pooled_projections:    nn.Parameter  # (1, 2048)
```

텍스트 없이 학습하므로 학습 가능한 null embedding 사용.

#### Forward 흐름

```python
# 1. sketch → frozen VAE encode
sketch_latent = vae.encode(sketch)           # (B, 16, 64, 64)

# 2. matte → trainable MatteCNN
matte_feat = matte_cnn(matte)                # (B, 16, 64, 64)

# 3. raw matte downsample (explicit spatial mask)
matte_latent = interpolate(matte, (64,64))   # (B,  1, 64, 64)

# 4. 17ch conditioning 구성
ctrl_cond = cat([sketch_latent + matte_feat, matte_latent], dim=1)  # (B, 17, 64, 64)

# 5. SD3ControlNetModel forward
block_samples = controlnet(noisy_latent, ctrl_cond, ...)  # list of 12 tensors

return block_samples, null_enc_hs, null_pooled
```

---

### 2. FaceControlNet (`src/models/face_controlnet.py`)

SD3.5 **마지막** 6블록(18~23)을 복사한 ControlNet. 얼굴-헤어 경계 blending 담당.

설계 근거: InstantX Qwen-Image ControlNet Inpainting에서 pretrained 마지막 6 double block 복사 방식 사용. 후반 블록은 세부 디테일과 경계 일관성에 특화되어 있음.

#### 초기화 특이사항

```python
# 1. 우선 blocks 0~5로 초기화 (SD3ControlNetModel API 제약)
self.controlnet = SD3ControlNetModel.from_transformer(
    transformer, num_layers=6,
    num_extra_conditioning_channels=0,   # face_latent는 순수 16ch
)

# 2. transformer_blocks를 blocks 18~23으로 교체
source_blocks = transformer.transformer_blocks[18:24]
self.controlnet.transformer_blocks = nn.ModuleList([
    copy.deepcopy(blk) for blk in source_blocks
])

# 3. SD3.5 block 23의 context_pre_only=True 패치 (2-tuple return 오류 방지)
for blk in self.controlnet.transformer_blocks:
    blk.context_pre_only = False
```

#### Forward 흐름

```python
# face_latent (16ch)를 직접 conditioning으로 사용 (추가 채널 없음)
block_samples = controlnet(
    hidden_states=noisy_latent,
    controlnet_cond=face_latent,           # (B, 16, 64, 64)
    encoder_hidden_states=null_enc_hs,     # HairControlNet에서 전달받음
    pooled_projections=null_pooled,
    timestep=sigmas,
)
return block_samples  # list of 6 tensors
```

---

### 3. SD3.5 Transformer (frozen)

24개의 Double Transformer Block으로 구성된 backbone. 모든 파라미터 `requires_grad=False`.

#### Residual 주입 전략

논문 수식 `F_i = F_i^hair · M_i + F_i^BG · (1 - M_i)` 를 DiT residual에 적용:

```python
# matte_tokens 준비
# SD3.5 patch_size=2 → 32×32 = 1024 tokens
matte_tokens = interpolate(matte, (32,32)).flatten(2).permute(0,2,1)  # (B, 1024, 1)

mixed_residuals = []

# Block  0~17: HairControlNet residuals만
# (HairControlNet 12개 residual → 각 SD3.5 2블록에 하나씩)
for i in range(18):
    mixed_residuals.append(hair_res_list[i // 2])
# i=0,1  → hair_res[0]
# i=2,3  → hair_res[1]
# ...
# i=16,17→ hair_res[8]

# Block 18~23: matte 비율로 hair + face residual 혼합
for i in range(18, 24):
    r_h = hair_res_list[i // 2]       # hair_res 9,9,10,10,11,11
    r_f = face_res_list[i - 18]       # face_res 0,1,2,3,4,5
    mixed_residuals.append(
        r_h * matte_tokens + r_f * (1.0 - matte_tokens)
    )
    # matte_tokens=1 (hair 영역): hair residual만
    # matte_tokens=0 (face 영역): face residual만
    # 0<matte<1 (경계):           선형 보간
```

---

### 4. VAEWrapper (`src/models/vae_wrapper.py`)

SD3.5 frozen VAE. encode/decode 래퍼.

- `encode(x)`: (B, 3, 512, 512) [0,1] → (B, 16, 64, 64)
- `decode(z)`: (B, 16, 64, 64) → (B, 3, 512, 512) [-1,1]
- `normalize(x)`: [0,1] → [-1,1] (LPIPS 계산용)

---

## 데이터 파이프라인

### Dataset (`src/data/dataset.py`)

**HairInpaintDataset** — 4-tuple 반환:

```
dataset/
  braid/
    img/    {train,test}/   ← 원본 full person image (GT)
    matte/  {train,test}/   ← hair 영역 soft alpha matte
    sketch/ {train,test}/   ← colored hair sketch
  unbraid/
    (동일 구조)
```

각 샘플:

```python
{
  "sketch":      (3, 512, 512)  float32 [0,1]   # colored hair sketch
  "matte":       (1, 512, 512)  float32 [0,1]   # soft alpha matte
  "masked_face": (3, 512, 512)  float32 [0,1]   # GT * (1 - matte) — FaceControlNet 조건
  "target":      (3, 512, 512)  float32 [0,1]   # full GT image
  "img":         (3, 512, 512)  float32 [0,1]   # = target (augmentation 호환)
  "filename":    str
}
```

`masked_face = target * (1 - matte)` 는 런타임에 생성되며 별도 파일 없음.

### Augmentation (`src/data/augmentation.py`)

| Augmentation | Phase 1 | Phase 2 | 역할 |
|---|---|---|---|
| SketchColorJitter (p=0.8) | O | X | 색 과적합 방지 |
| ThicknessJitter / dilation (p=0.5) | O | O | 선 두께 변화 강건성 |
| MattePerturbation / elastic (p=0.3) | O | O | 경계 흔들림 강건성 |
| AppearanceJitter (p=0.5) | O | X | 구조-외관 분리 (target에만 적용) |
| StrokeColorSampler | X | O | stroke ↔ target 색 대응 강제 |

Phase 2에서 AppearanceJitter 제거 + StrokeColorSampler 활성화 → stroke 색과 target 색 대응을 보존하여 정밀한 색 제어 학습.

---

## 학습 파이프라인

### Trainer (`src/training/trainer.py`)

```
학습 가능 파라미터:
  HairControlNet (controlnet + matte_cnn + null_encoder_hidden_states + null_pooled_projections)
  FaceControlNet (controlnet)

고정 파라미터:
  SD3Transformer2DModel (requires_grad=False, eval mode)
  VAEWrapper (frozen, eval mode)
```

#### Train Step 전체 흐름

```python
# ① VAE encode (no_grad)
latents     = vae.encode(target)         # GT latent (B, 16, 64, 64)
face_latent = vae.encode(masked_face)    # Face conditioning latent

# ② Sigma sampling (logit-normal)
u      = sigmoid(N(logit_mean=0, logit_std=1))
sigmas = scheduler.sigmas[(u * T).long()]   # (B, 1, 1, 1)

# ③ Partial noise: hair 영역만 노이즈
matte_latent  = resize_matte_to_latent(matte)  # (B, 1, 64, 64)
noisy_latents = latents * (1 - matte_latent) + sigmas * noise * matte_latent
#               ↑ face: GT latent 유지           ↑ hair: σ*noise

# ④ HairControlNet forward
hair_res, null_enc, null_pooled = hair_controlnet(noisy_latents, sketch, matte, σ)

# ⑤ FaceControlNet forward (HairControlNet의 null embedding 재사용)
face_res = face_controlnet(noisy_latents, face_latent, null_enc, null_pooled, σ)

# ⑥ 24개 mixed residuals 구성
matte_tokens = interpolate(matte, (32,32)).flatten(2).permute(0,2,1)  # (B,1024,1)
mixed = [hair_res[i//2] for i in range(18)]
mixed += [hair_res[i//2]*matte_tokens + face_res[i-18]*(1-matte_tokens) for i in range(18,24)]

# ⑦ Transformer v-prediction
v_pred   = transformer(noisy_latents, mixed, ...)
v_target = noise - latents   # flow matching GT velocity

# ⑧ Loss (matte 영역만)
loss, log_dict = hair_loss(v_pred, v_target, matte_latent, ...)
```

#### Optimizer

```python
AdamW(
    params       = hair_controlnet.parameters() + face_controlnet.parameters(),
    lr           = 1e-4,
    betas        = (0.9, 0.999),
    weight_decay = 1e-2,
)
CosineAnnealingLR(T_max=total_steps - warmup, eta_min=1e-6)
# + linear warmup until warmup_steps
```

#### EMA

```python
EMAModel(hair_controlnet, decay=0.9999)
EMAModel(face_controlnet, decay=0.9999)
```

체크포인트: `ema_hair`, `ema_face` 모두 저장.

---

## 손실 함수 (`src/training/losses.py`)

```
L_total = w_flow × L_flow + w_lpips × L_lpips + w_edge × L_edge
```

### L_flow — FlowMatchingLoss

```python
diff_sq = (v_pred - v_target) ** 2         # (B, 16, 64, 64)
masked  = matte_latent * diff_sq            # hair 영역만
area    = matte_latent.sum() × C + 1e-8    # matte 크기로 정규화
L_flow  = masked.sum() / area
```

### L_lpips — PerceptualLoss (VGG LPIPS)

```python
x0_pred  = x_t - σ × v_pred               # flow matching x0 복원
pred_rgb = vae.decode(x0_pred)             # (B, 3, 512, 512) [-1,1]

L_lpips = LPIPS(pred_rgb × matte, target_rgb × matte)
```

활성화 조건:
- Phase 1: `current_step >= lpips_warmup_frac (30%) × total_steps`
- Phase 2: `lpips_warmup_frac=0.0` → 처음부터 활성화

### L_edge — SketchEdgeAlignmentLoss (Phase 2만)

```python
sketch_mask = (sketch.max(dim=1) > 0.1) × matte   # stroke 있는 영역 ∩ matte

pred_gray = pred_rgb.mean(dim=1)
edge_mag  = Sobel(pred_gray).norm()                # (B, 1, H, W)

# 페널티: stroke 있는데 엣지 없는 곳
L_edge = (sketch_mask × (1 - edge_mag)).mean()
```

### 단계별 손실 가중치

| 손실 | Phase 1 (Pretrain) | Phase 2 (Finetune) |
|------|:-----------------:|:-----------------:|
| w_flow | 1.0 | 1.0 |
| w_lpips | 0.1 (warmup 30%) | 0.1 (즉시) |
| w_edge | 0.0 | 0.05 |

---

## 2단계 커리큘럼 학습

### Phase 1: Unbraid Pretrain (`configs/phase1_unbraid.yaml`)

```yaml
dataset:      unbraid
epochs:       50
batch_size:   2                      # A100 40GB 기준
gradient_accumulation_steps: 4       # effective batch size = 8
lr:           1e-4
warmup_steps: 500
loss_weights: flow=1.0, lpips=0.1 (warmup 30%), edge=0.0
output_dir:   checkpoints/phase1_unbraid/
eval_every:   10 epochs
save_every:   20 epochs
```

- 데이터: unbraid ~3,000장 (steps/epoch: 1,500)
- best checkpoint: epoch 30 (val loss 0.1800)
- SketchColorJitter + AppearanceJitter → 색/구조 분리 학습

### Phase 2: Braid Finetune (`configs/phase2_braid.yaml`)

```yaml
dataset:      braid
resume_from:  checkpoints/phase1_unbraid/best.pth  # epoch 30
epochs:       50
batch_size:   4
gradient_accumulation_steps: 2       # effective batch size = 8
lr:           2e-5                   # Phase 1의 1/5
warmup_steps: 100
loss_weights: flow=1.0, lpips=0.1 (즉시), edge=0.05
output_dir:   checkpoints/phase2_braid/
eval_every:   5 epochs
save_every:   10 epochs
```

- 데이터: braid ~1,000장 (steps/epoch: 250)
- best checkpoint: epoch 5 (val loss 0.1998) → epoch 10 이후 단조 증가 (overfitting)
- StrokeColorSampler + Edge loss → braid strand 구조 충실도 향상

---

## 추론 파이프라인 (`scripts/infer_inpaint.py`)

```
입력: face photo + sketch + matte
출력: full 512×512 (composite 불필요 )
```

### Face Re-injection (identity 보존)

```python
scheduler.set_timesteps(num_steps=20)

# 초기화: face 영역 = GT latent, hair 영역 = noise
face_latent      = vae.encode(face_image)
masked_face_latent = vae.encode(face_image * (1 - matte))
latents = face_latent * (1 - matte_down) + noise * matte_down

for t in scheduler.timesteps:
    # Dual ControlNet → mixed residuals → v_pred
    hair_res, null_enc, null_pooled = hair_controlnet(latents, sketch, matte, σ)
    face_res = face_controlnet(latents, masked_face_latent, null_enc, null_pooled, σ)
    mixed    = build_mixed_residuals(hair_res, face_res, matte_tokens)
    v_pred   = transformer(latents, mixed, ...)

    latents = scheduler.step(v_pred, t, latents)

    # Face re-injection: 매 스텝마다 face 영역을 GT latent로 복원
    latents = latents * matte_down + face_latent * (1 - matte_down)
    #          ↑ hair: 모델 생성     ↑ face: 원본 픽셀 완벽 보존

image = vae.decode(latents)   # (1, 3, 512, 512) [0, 1]
```

역할 분리:
- **FaceControlNet**: hair-face 경계 blending
- **Face Re-injection**: 픽셀 수준 face identity 보존 (VAE 재인코딩 없음)
```

## 데이터셋 통계

| 스타일 | 수량 | 용도 |
|--------|------|------|
| unbraid | ~3,000장 | Phase 1 pretrain |
| braid | ~1,000장 | Phase 2 finetune |
| 합계 | ~4,000장 | |

Self-supervised 구성: `masked_face = GT × (1 - matte)` 런타임 생성 → 추가 데이터 수집 불필요.

---

## 하드웨어 & 학습 환경

| 항목 | 값 |
|------|----|
| GPU | A 100 40GB |
| Precision | bfloat16 (HuggingFace Accelerate) |
| Gradient Checkpointing | Transformer + HairControlNet + FaceControlNet + VAE |
| Logging | TensorBoard |

