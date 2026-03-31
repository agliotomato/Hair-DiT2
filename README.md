## 코드 구조

```
hair-dit2/
├── configs/
│   ├── base.yaml              # 공통 설정 (모델 ID, mixed precision, EMA, loss weights)
│   ├── phase1_unbraid.yaml    # Phase 1 학습 설정 (base.yaml 상속)
│   └── phase2_braid.yaml      # Phase 2 학습 설정 (base.yaml 상속)
├── dataset/
│   ├── braid/
│   │   ├── img/test/          # 원본 full image
│   │   ├── matte/test/        # hair alpha matte
│   │   └── sketch/test/       # colored sketch
│   └── unbraid/
│       ├── img/{train,test}/
│       ├── matte/{train,test}/
│       └── sketch/{train,test}/
├── scripts/
│   ├── train.py               # 학습 진입점
│   └── infer_inpaint.py       # 추론 스크립트 (face re-injection 포함)
└── src/
    ├── data/
    │   ├── dataset.py         # HairInpaintDataset — (sketch, matte, masked_face, target)
    │   ├── augmentation.py    # Phase별 augmentation 파이프라인
    │   └── utils.py           # resize_matte_to_latent 등 유틸리티
    ├── models/
    │   ├── hair_controlnet.py # HairControlNet + MatteCNN (12 blocks)
    │   ├── face_controlnet.py # FaceControlNet (blocks 18~23 초기화)
    │   └── vae_wrapper.py     # SD3.5 VAE encode/decode/normalize
    └── training/
        ├── trainer.py         # Trainer — 학습 루프, mixed residual 주입 로직
        ├── losses.py          # HairLoss — FlowMatching + LPIPS + EdgeAlignment
        └── ema.py             # EMAModel
```
