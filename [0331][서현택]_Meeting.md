# Meeting Notes — hair-dit2

### 1. Context & Problem

**기존 방식의 한계**
현재의 Inpainting(Hole-filling) 방식은 타겟 이미지($I_{target}$)의 배경이나 포즈에 종속됨.
사용자가 그린 스케치가 타겟의 컨텍스트와 맞지 않으면 (예: 정면 얼굴에 복잡한 뒷머리 스케치) 텍스처가 뭉개지거나 '스티커'처럼 부자연스러운 결과가 발생함.

**핵심 질문**
"모델이 해당 스케치를 가장 잘 그릴 수 있는 최적의 환경(포즈/조명)에서 소스를 먼저 만들고, 이를 타겟에 이식할 수는 없는가?"

**결론**
단순 합성이 아닌, 잠재 공간(Latent Space)에서의 화학적 융합을 통해 품질과 자연스러움을 동시에 잡는 아키텍처?

---

### 2. Q&A

**Q1. DiT는 베이스 이미지에 따라 결과가 달라지는가?**

그렇다. DiT는 Self-Attention을 통해 주변 픽셀과의 통계적 연관성을 맞추려 함.
베이스 이미지가 스케치와 '통계적으로 먼' 조건이면 스케치를 무시하거나 저품질 결과를 출력함.

**Q2. 최고의 스케치 결과를 "찾아낼" 수 있는가?**

- Latent Optimization: 스케치 S를 입력으로, Loss를 최소화하는 방향으로 잠재 벡터(z)를 역추적(Inversion)하여 모델이 가장 자신 있게 그려낼 수 있는 '마스터피스 이미지(I_source)'를 역생성.
- Semantic Retrieval: 벡터 DB를 활용해 스케치와 유사도가 높은 학습 데이터셋 내의 최적 Reference를 탐색. 

---

### 3. Proposed Solution: Source-to-Target Latent Transfer

**Phase 1: Finding Optimal Source (I_source)**
목표: 타겟 이미지의 제약(포즈, 각도 등)에서 벗어나, 입력 스케치 S를 기하학적·질감적으로 가장 완벽하게 구현하는 고해상도 헤어 패치 소스를 확보.

**Phase 2: Generative Blending (Feature-Level Injection)**
목표: 추출된 소스를 타겟에 '스티커'처럼 붙이는 것이 아니라, 모델 내부의 특징량 수준에서 융합.

방법:
1. DiT의 중간 레이어(Residual Blocks)에서 I_source의 특징량(Feature Maps)을 추출.
2. 타겟 이미지 생성 과정 중 특정 Diffusion Step (예: T=24 to 18)에서 Masked Attention Injection을 수행.
3. 마지막 Step에서 VAE를 통해 전체 이미지를 복원함으로써 경계면이 물리적으로 닿는 곳 없이 자연스럽게 생성되도록 유도.

---

### 4. Implementation Guidance

핵심 구현 포인트:
- Latent Optimization: 스케치 가이드에 최적화된 z*를 찾는 Optimization Loop 설계.
- Feature Hooking: DiT 내부의 특정 레이어 특징량을 가로채고 저장하는 모듈 구현.
- Seamless Sampler: 역과정(Reverse Process) 중 특정 구간에서만 특징량을 주입하여 타겟과 소스를 '화학적으로' 섞어주는 커스텀 파이프라인.

---