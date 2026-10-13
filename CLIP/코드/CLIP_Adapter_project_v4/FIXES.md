# 수정 내역 (FIXES)

원본(zip) 대비 수정한 파일과 이유 요약.

## 1. `src/coop_model.py`

`CLIPAdapter` 클래스의 docstring과 `forward` 시그니처 docstring을 수정함.
**입력 contract 변경**: 이제 `CLIPAdapter.forward(x)`의 `x`는
*unnormalized* CLIP image feature를 받음. 내부에서 잔차 결합 후 마지막에
`F.normalize`를 적용함. 모델 weight 구조 자체는 동일하므로 기존
checkpoint(`adapter_initialization.pt`, `adapter_final.pt`)를 그대로 로드 가능함.

## 2. `train_blue_target_adapter.py`

- `train_one_epoch_adapter`에서 `model.encode_image(images)` 직후의
  `image_features = F.normalize(image_features, dim=-1)` 호출을 제거.
- `evaluate_adapter`의 비-cached 경로에서도 동일하게
  `img_feat = F.normalize(img_feat, dim=-1)` 호출을 제거.
- `evaluate_zero_shot`은 zero-shot CLIP 자체의 동작이므로 그대로 둠
  (zero-shot은 정규화 후 dot product가 표준).

## 3. `src/eval.py`

- `precompute_image_features`에서 캐시 시점의 `F.normalize`를 제거.
  이제 캐시는 *raw* feature로 저장됨. CLIP-Adapter는 raw 스케일에서
  잔차 결합을 해야 정상 동작하므로 캐시도 raw로 두어야 함.
- `evaluate_classifier`의 cached 경로에서, 캐시가 raw가 되었으므로
  zero-shot 비교 직전에 `F.normalize`를 적용하도록 수정.

## 4. `scripts/run_texture_adapter_100seeds_gpu1.sh`,
   `scripts/run_texture_adapter_gpu1.sh`

- `--adapter-alpha 0.2` → `--adapter-alpha-init 0.6`
  - 기존에는 argparse prefix matching으로 `--adapter-alpha-init`에 매칭되어
    값이 0.2로 설정되고 있었음. 명시적인 인자명을 쓰도록 수정.
  - DTD/texture는 fine-grained dataset이므로 논문 Table 2에 따라 α=0.6이 최적.
    α=0.2는 ImageNet 같은 generic dataset용 값임.

---

## 왜 기존 그래프가 거의 평평했는가 (간단 설명)

CLIP image encoder 출력 magnitude는 대략 5–15 수준이며 unit norm이 아님.
저자 공식 `clip_adapter.py`는 이 raw feature에 adapter를 적용한 뒤
잔차 결합을 하고, 모든 작업이 끝난 후에 정규화함.

기존 코드는 `model.encode_image()` 직후 곧바로 `F.normalize`를 호출해
adapter 입력을 unit sphere(magnitude≈1)로 축소시켰음. 이러면:

- Linear→ReLU→Linear의 출력 magnitude ≈ 0.005 수준으로 축소
- 잔차 결합: `α·(0.005) + (1-α)·1 ≈ (1-α)` → 다시 정규화 시 거의 원본 방향과 동일
- 결과적으로 adapter 기여가 무시할 수준이 되어 학습이 거의 진행되지 않음

여기에 α=0.2까지 곱해지면서 16-shot에서도 ~+2%p 정도밖에 못 가는
관찰된 곡선이 나타남. 이번 수정으로 잔차 결합이 raw feature 스케일에서
이루어지고 α가 올바른 값으로 들어가므로, 논문(예: DTD 16-shot ~66%)에
근접하는 수치가 나올 것으로 기대됨.

## 권장 sanity check

수정 후 1-shot 또는 16-shot 단일 seed로 우선 짧게 돌려서
target accuracy가 zero-shot 대비 명확히 (>= 5–10%p) 올라가는지 확인 후
100 seed 전체 실행을 권장함. 첫 epoch 이후 train loss가 빠르게 감소하면
정상 동작하는 것으로 볼 수 있음.
