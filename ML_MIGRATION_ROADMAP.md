# 🤖 ML 기반 투구 분석 마이그레이션 로드맵

> **문서 목적**: 현재 신호처리 알고리즘 → 머신러닝 모델로 전환하기 위한 상세 계획.
> 앱 출시 후 데이터가 축적되면 이 로드맵에 따라 ML 전환을 진행합니다.

---

## 📋 목차

1. [전환 전략 개요](#1-전환-전략-개요)
2. [데이터 수집 전략](#2-데이터-수집-전략)
3. [라벨링 가이드라인](#3-라벨링-가이드라인)
4. [모델 아키텍처 비교](#4-모델-아키텍처-비교)
5. [학습 파이프라인](#5-학습-파이프라인)
6. [iOS CoreML 배포](#6-ios-coreml-배포)
7. [단계별 전환 일정](#7-단계별-전환-일정)
8. [리스크 및 대응 전략](#8-리스크-및-대응-전략)

---

## 1. 전환 전략 개요

### 왜 바로 ML을 적용하지 않는가?

| 조건 | 현재 상황 | ML 적용 최소 조건 |
|---|---|---|
| 라벨링된 영상 수 | 12개 (라벨 없음) | 200~500개 (라벨 있음) |
| Ground Truth | 없음 | 프레임 단위 라벨 필요 |
| 투구자 다양성 | 1~2명 추정 | 최소 20~30명 |
| 촬영 환경 다양성 | 제한적 | 측면/측후면/측전면 × 다양한 환경 |

### 전체 전환 흐름

```
[Phase 0] 알고리즘 기반 앱 출시 (현재)
    │
    ├── 앱 내 데이터 수집 인프라 구축
    │
[Phase 1] 데이터 축적 (200~500 영상)
    │
    ├── 라벨링 도구 개발 or 외주
    │
[Phase 2] ML 모델 학습 + 검증
    │
    ├── CoreML 변환 + A/B 테스트
    │
[Phase 3] ML 모델 앱 배포
    │
    ├── 알고리즘 → ML 완전 전환
    │
[Phase 4] 지속적 개선 (Online Learning)
```

---

## 2. 데이터 수집 전략

### 2.1 앱 내 자동 수집 (권장)

앱 출시 시점부터 사용자 동의 하에 데이터를 자동 수집합니다.

**수집 데이터:**

| 데이터 | 형식 | 용량(예상) | 용도 |
|---|---|---|---|
| **포즈 시계열** | JSON (프레임별 관절 좌표) | ~50KB/영상 | ML 학습 입력 |
| **분석 결과** | JSON (투구 횟수, 페이즈) | ~5KB/영상 | 알고리즘 결과 vs ML 비교 |
| **사용자 피드백** | Boolean (정확/부정확) | ~1KB | 약한 라벨(Weak Label) |
| **원본 영상** (선택) | MP4 (720p) | ~5MB/영상 | 정밀 라벨링용 |

**수집 인프라 구조 (iOS → 서버):**

```
iOS App
  ├── 분석 완료 후 "정확했나요?" 팝업 → 사용자 피드백 수집
  ├── 포즈 시계열 + 분석 결과 → Firebase/S3 업로드
  └── (선택) 원본 영상 → 사용자 동의 시 업로드
         │
         ▼
Cloud Storage (Firebase / AWS S3)
  ├── raw_poses/          ← 관절 좌표 시계열
  ├── analysis_results/   ← 알고리즘 분석 결과
  ├── user_feedback/      ← 정확성 피드백
  └── videos/             ← 원본 영상 (라벨링용)
```

### 2.2 수집 목표량

| 단계 | 영상 수 | 투구 수 | 투구자 수 | 가능 시점 |
|---|---|---|---|---|
| **최소 학습 가능** | 200 | 600 | 20+ | 출시 후 1~2개월 |
| **안정적 학습** | 500 | 1,500 | 50+ | 출시 후 3~6개월 |
| **고성능 모델** | 2,000+ | 6,000+ | 200+ | 출시 후 6~12개월 |

### 2.3 데이터 다양성 확보 체크리스트

- [ ] 촬영 각도: 측면, 측후면, 측전면 (3종류 균등 분포)
- [ ] 촬영 기기: 다양한 iPhone 모델 (fps/해상도 차이)
- [ ] 투구자: 초보~중급~상급 레벨 포함
- [ ] 투구 스타일: Standard, Side-arm 등
- [ ] 조명 환경: 실내 밝음, 실내 어두움, 야외
- [ ] 복장: 반팔, 긴팔, 두꺼운 옷 등 (포즈 추출 난이도 다양화)

---

## 3. 라벨링 가이드라인

### 3.1 라벨링이 필요한 항목

ML 모델의 태스크별로 필요한 라벨이 다릅니다:

#### Task A: 투구 세분화 (Throw Segmentation)

```
라벨 형식: 프레임 구간 (start_frame, end_frame)
난이도: ★★☆ (중간)

예시:
{
  "video_id": "sample_8.mp4",
  "throws": [
    {"start": 135, "end": 185},
    {"start": 235, "end": 290},
    {"start": 355, "end": 405}
  ]
}
```

**라벨링 기준:**
- **시작**: 팔이 뒤로 움직이기 시작하는 프레임 (테이크백 시작)
- **종료**: 팔이 완전히 펴진 후 안정되는 프레임 (팔로스루 완료)
- **허용 오차**: ±3 프레임 (±100ms at 30fps)

#### Task B: 릴리즈 타이밍 감지 (Release Detection)

```
라벨 형식: 단일 프레임 인덱스
난이도: ★★★ (어려움 — 30fps에서 33ms 단위 판단)

예시:
{
  "video_id": "sample_8.mp4",
  "throws": [
    {"release_frame": 162},
    {"release_frame": 268},
    {"release_frame": 388}
  ]
}
```

**라벨링 기준:**
- 다트가 손에서 분리되는 마지막 프레임
- 30fps에서는 판단이 어려우므로 **"팔꿈치가 가장 빠르게 펴지는 순간"** 을 대안 기준으로 사용
- **허용 오차**: ±2 프레임 (±66ms at 30fps)

#### Task C: 4-페이즈 분류 (Phase Classification)

```
라벨 형식: 프레임별 페이즈 레이블
난이도: ★★★ (어려움 — 경계가 모호)

예시:
{
  "video_id": "sample_8.mp4",
  "throw_1": {
    "address":      [135, 148],
    "takeback":     [149, 162],
    "release":      [163, 170],
    "follow_through": [171, 185]
  }
}
```

### 3.2 라벨링 도구 선택지

| 도구 | 비용 | 특징 | 추천 |
|---|---|---|---|
| **CVAT** (오픈소스) | 무료 | 비디오 프레임별 라벨링, 웹 기반 | ✅ 개인/소규모 |
| **Label Studio** (오픈소스) | 무료 | 시계열 + 비디오 지원, 셀프호스팅 | ✅ 시계열 중심 |
| **Labelbox** | 유료 | 팀 라벨링 + QA 워크플로우 | 대규모 인력 투입 시 |
| **자체 개발** (Python + OpenCV) | 무료 | 영상 재생 + 키보드로 구간 마킹 | ✅ 초기 테스트용 |

### 3.3 반자동 라벨링 파이프라인 (추천)

데이터가 쌓이면 순수 라벨링은 비효율적. 알고리즘 결과를 **초벌 라벨**로 사용하고 사람이 **검수/수정**합니다:

```
1. 영상 → 알고리즘 분석 → 초벌 라벨 자동 생성
2. 라벨링 웹 UI에서 검수자가 확인 + 수정
3. 수정된 라벨 → Ground Truth로 저장
4. GT 축적 → ML 모델 학습
5. 새 ML 모델 → 더 정확한 초벌 라벨 → 검수 부담 감소 (선순환)
```

**핵심 포인트:** 알고리즘 기반 시스템이 초벌 라벨러 역할을 하므로, **지금 알고리즘을 잘 만들어두면 나중에 라벨링 비용이 크게 줄어듭니다.**

---

## 4. 모델 아키텍처 비교

### 4.1 입력 데이터 형식

모든 모델의 입력은 **관절 좌표 시계열**입니다:

```
입력 텐서: [batch, time_steps, features]

features = 관절 8개 × 3D좌표 = 24
         + 손가락 6개 × 3D좌표 = 18
         = 총 42 차원

time_steps = 투구 1개 ~60프레임 (2초 at 30fps)
           → 고정 길이 패딩/잘라내기로 통일
```

### 4.2 후보 모델 비교

#### Model A: 1D-CNN (투구 세분화용 — 추천)

```
목적: 연속 시계열에서 "투구 구간" 검출
구조: 슬라이딩 윈도우로 시계열을 잘라서 "투구/비투구" 이진 분류

[Input: 42ch × 30frames]
    │
    ├── Conv1D(64, kernel=5) → ReLU → MaxPool
    ├── Conv1D(128, kernel=3) → ReLU → MaxPool
    ├── Conv1D(256, kernel=3) → ReLU → GlobalAvgPool
    ├── Dense(64) → Dropout(0.3)
    └── Dense(1, sigmoid)  ← "투구 중?" 확률

장점: 구조 단순, 학습 빠름, CoreML 변환 쉬움
단점: 시간적 컨텍스트가 윈도우 크기로 제한
데이터: ~200 투구 + ~200 비투구 구간
```

#### Model B: LSTM / Bi-LSTM (릴리즈 타이밍용 — 추천)

```
목적: 투구 구간 내에서 릴리즈 프레임을 정확히 지목
구조: Sequence-to-Sequence로 프레임별 "릴리즈 확률" 출력

[Input: 42ch × 60frames]
    │
    ├── Bi-LSTM(128) → Dropout(0.2)
    ├── Bi-LSTM(64) → Dropout(0.2)
    ├── TimeDistributed(Dense(32)) → ReLU
    └── TimeDistributed(Dense(1, sigmoid))  ← 프레임별 릴리즈 확률

장점: 시간 순서 정보를 양방향으로 활용
단점: 학습 느림, CoreML 변환 시 주의 필요 (StatefulModel)
데이터: ~300 투구, 각각 릴리즈 프레임 라벨
```

#### Model C: TCN — Temporal Convolutional Network (통합 모델 — 고급)

```
목적: 투구 세분화 + 릴리즈 + 페이즈를 하나의 모델로
구조: Dilated Causal Convolution으로 넓은 시간 범위 커버

[Input: 42ch × 전체 영상]
    │
    ├── TCN Block: Conv1D(dilation=1) → Conv1D(dilation=2) → ...
    ├── TCN Block: Conv1D(dilation=4) → Conv1D(dilation=8) → ...
    ├── TCN Block: Conv1D(dilation=16) → Conv1D(dilation=32) → ...
    └── Dense(5, softmax)  ← 프레임별 5-class 분류
                              {idle, address, takeback, release, follow_through}

장점: 단일 모델로 전체 분석, 인과적(causal) 구조로 실시간 추론 가능
단점: 학습 데이터 500+ 필요, 구현 복잡
데이터: ~500 영상, 프레임별 5-class 라벨
```

### 4.3 추천 전략: 단계별 모델 도입

```
[Phase 1: 200 영상 도달]
 → Model A (1D-CNN) 학습: 투구 세분화만 ML로 전환
 → 나머지(릴리즈, 메트릭)는 알고리즘 유지

[Phase 2: 500 영상 도달]
 → Model B (Bi-LSTM) 추가: 릴리즈 타이밍도 ML로 전환
 → 메트릭 계산은 알고리즘 유지 (ML 불필요 — 수학적 계산)

[Phase 3: 2000+ 영상 도달]
 → Model C (TCN) 전환: 통합 모델로 단순화
 → 또는 Model A+B 조합 유지 (실용적으로 충분)
```

---

## 5. 학습 파이프라인

### 5.1 개발 환경

```
학습 서버: Google Colab Pro / AWS SageMaker / Apple M-chip Mac
프레임워크: PyTorch (→ CoreML 변환은 coremltools 사용)
관리: MLflow 또는 Weights & Biases (실험 추적)
```

### 5.2 데이터 전처리

```python
# 전처리 파이프라인 (의사 코드)

def preprocess_pose_sequence(raw_keypoints, target_length=60):
    """관절 좌표 시계열을 ML 모델 입력으로 변환"""

    # 1. 핸드헬드 보정: 어깨 중점 기준 상대 좌표 변환
    mid_shoulder = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2
    relative_coords = keypoints - mid_shoulder  # 카메라 흔들림 제거

    # 2. 스케일 정규화: 어깨-엉덩이 거리 기준
    torso_length = ||shoulder - hip||
    normalized = relative_coords / torso_length

    # 3. 길이 통일: 패딩 or 보간
    if len(normalized) < target_length:
        padded = zero_pad(normalized, target_length)  # 제로 패딩
    else:
        resampled = interpolate(normalized, target_length)  # 시간 보간

    # 4. 데이터 증강 (학습 시)
    augmented = apply_augmentation(resampled)

    return augmented  # shape: [target_length, 42]
```

### 5.3 데이터 증강 (소량 데이터 극복)

12개 → 200개 전에도 증강으로 효과를 얻을 수 있습니다:

| 증강 기법 | 설명 | 효과 |
|---|---|---|
| **시간축 스트레칭** | 0.8x~1.2x 속도 변환 | 빠른/느린 투구자 대응 |
| **좌우 반전** | 좌표 X축 반전 | 좌타/우타 데이터 2배 |
| **노이즈 주입** | 각 좌표에 가우시안 노이즈 추가 | 핸드헬드 떨림 내성 |
| **관절 드롭아웃** | 랜덤 관절 좌표를 0으로 마스킹 | 포즈 추출 실패 내성 |
| **시간축 이동** | 시작/끝 위치를 ±5프레임 이동 | 세분화 경계 유연화 |
| **스케일 변형** | 전체 좌표 0.9x~1.1x 스케일링 | 촬영 거리 차이 대응 |

### 5.4 학습 설정

```python
# 1D-CNN 학습 예시 (의사 코드)

model = ThrowDetectorCNN(input_channels=42, num_classes=2)
optimizer = Adam(lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
loss_fn = BCEWithLogitsLoss()

# K-Fold 교차검증 (데이터가 적으므로 필수)
for fold in KFold(n_splits=5, shuffle=True):
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    for epoch in range(100):
        train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics = evaluate(model, val_loader)

        # Early Stopping (patience=15)
        if val_metrics.loss < best_loss:
            save_checkpoint(model)
```

### 5.5 평가 지표

| 태스크 | 주요 지표 | 목표 수치 |
|---|---|---|
| 투구 세분화 | Precision / Recall / F1 | F1 > 0.95 |
| 릴리즈 감지 | Mean Absolute Error (프레임) | MAE < 2 프레임 |
| 페이즈 분류 | Frame-level Accuracy | > 90% |

---

## 6. iOS CoreML 배포

### 6.1 모델 변환 파이프라인

```
PyTorch 모델 (.pt)
    │
    ├── torch.jit.trace() → TorchScript
    │
    ├── coremltools.convert() → .mlmodel / .mlpackage
    │
    └── Xcode 프로젝트에 추가 → on-device 추론
```

### 6.2 변환 시 주의사항

| 레이어 | CoreML 지원 | 대응 방법 |
|---|---|---|
| Conv1D | ✅ 완전 지원 | 그대로 변환 |
| LSTM | ⚠️ 부분 지원 | Stateless LSTM으로 변환 필요 |
| Bi-LSTM | ⚠️ 주의 | Forward/Backward 분리 후 concat |
| BatchNorm1D | ✅ 지원 | 추론 시 folding 적용 |
| LayerNorm | ✅ 지원 | CoreML 5+ |
| Dilated Conv | ✅ 지원 | TCN에 필요 |

### 6.3 온디바이스 성능 예상

| 모델 | 파라미터 수 | 추론 시간 (iPhone 14) | 모델 크기 |
|---|---|---|---|
| 1D-CNN (세분화) | ~100K | ~2ms | ~400KB |
| Bi-LSTM (릴리즈) | ~300K | ~5ms | ~1.2MB |
| TCN (통합) | ~500K | ~8ms | ~2MB |

> 모든 모델이 **실시간 추론 가능** (30fps = 33ms/프레임 예산 내)

### 6.4 앱 내 ML 통합 구조

```swift
// Swift 의사코드
class DartAnalyzer {
    // Phase 1: 알고리즘만 사용
    let algorithmEngine = SignalProcessingEngine()

    // Phase 2: ML + 알고리즘 하이브리드
    let throwDetector = try? ThrowSegmenterML(model: ThrowCNN())
    let releaseDetector = try? ReleaseDetectorML(model: ReleaseLSTM())
    let metricsCalculator = MetricsCalculator()  // 항상 알고리즘

    func analyze(poses: [PoseFrame]) -> SessionResult {
        // 1. 투구 구간 검출 (ML or 알고리즘 fallback)
        let segments = throwDetector?.detect(poses)
                       ?? algorithmEngine.segmentThrows(poses)

        // 2. 각 투구의 릴리즈 감지 (ML or 알고리즘 fallback)
        let phases = segments.map { segment in
            releaseDetector?.detectPhases(segment)
            ?? algorithmEngine.detectPhases(segment)
        }

        // 3. 메트릭 계산 (항상 알고리즘 — 수학적 계산이므로 ML 불필요)
        let metrics = phases.map { metricsCalculator.compute($0) }

        return SessionResult(segments, phases, metrics)
    }
}
```

---

## 7. 단계별 전환 일정

### Phase 0: 앱 출시 + 데이터 수집 인프라 (현재 ~ 앱 출시)

| 작업 | 상세 | 예상 비용 |
|---|---|---|
| 알고리즘 기반 분석 완성 | 지금 진행 중 | - |
| 포즈 데이터 업로드 구현 | 분석 완료 시 관절 좌표 JSON을 서버로 전송 | Firebase 무료 티어 |
| 사용자 피드백 UI | "분석이 정확했나요?" 1-tap 피드백 | - |
| 개인정보 처리 동의 | 데이터 수집 동의서 (앱 내) | - |

### Phase 1: ML 학습 시작 (데이터 200개 도달 시)

| 작업 | 상세 | 예상 기간 |
|---|---|---|
| 라벨링 | 200개 영상의 투구 구간 라벨링 | 1~2주 (반자동) |
| 1D-CNN 학습 | 투구 세분화 모델 | 1주 |
| A/B 테스트 | 알고리즘 vs ML 정확도 비교 | 1주 |
| CoreML 변환 + 앱 업데이트 | ML 모델 앱에 탑재 | 1주 |

### Phase 2: 릴리즈 감지 ML (데이터 500개 도달 시)

| 작업 | 상세 | 예상 기간 |
|---|---|---|
| 릴리즈 프레임 라벨링 | 300+ 투구의 정밀 라벨링 | 2~3주 |
| Bi-LSTM 학습 | 릴리즈 감지 모델 | 1~2주 |
| 통합 테스트 | CNN(세분화) + LSTM(릴리즈) 파이프라인 | 1주 |

### Phase 3: 고도화 (필요 시)

- TCN 통합 모델 검토
- 자세 분류(Standard/Side-arm) 모델 추가
- 온디바이스 학습(On-device Training) 검토 — 개인화

---

## 8. 리스크 및 대응 전략

### 8.1 데이터 부족 리스크

| 리스크 | 발생 조건 | 대응 |
|---|---|---|
| 앱 사용자가 적어 데이터 미축적 | MAU < 100 | 다트 동호회 제휴, 라벨링 알바 모집 |
| 특정 촬영 각도만 편중 | 측면 영상이 90% | 앱 내 촬영 가이드로 다양성 유도 |
| 라벨 품질 불균일 | 라벨러 간 기준 차이 | 라벨링 가이드라인 교육 + 교차검증 |

### 8.2 모델 성능 리스크

| 리스크 | 대응 |
|---|---|
| ML이 알고리즘보다 못할 수 있음 | A/B 테스트로 검증, 성능이 나을 때만 전환 |
| 새로운 투구 스타일에 과적합 | 데이터 증강 + 정기 재학습 |
| CoreML 변환 시 정확도 저하 | Float16 양자화 대신 Float32 유지, 변환 전후 비교 |

### 8.3 현실적 판단 기준

> **알고리즘 시스템의 F1이 0.90 이상이라면, ML 전환의 ROI(투자 대비 효과)가 낮을 수 있습니다.**
> ML 전환은 알고리즘으로 해결하기 어려운 케이스(예: 비전형적 투구 스타일, 극단적 촬영 환경)가
> 빈번할 때 가장 큰 효과를 발휘합니다.

---

## 부록: 참고 연구

| 연구 | 핵심 내용 | 적용 포인트 |
|---|---|---|
| Huang et al., 2024 | 다트 투구의 상지 관절 운동학 분석 | 생체역학 메트릭 기준값 |
| Smirnov, 2019 | 다트 투구의 시각적 피드백 효과 | 피드백 설계 참고 |
| IMU 기반 다트 분석 (MDPI) | IMU 센서로 투구 구간 자동 감지 | 피크 감지 알고리즘 설계 |
| TCN (Lea et al.) | Temporal Convolutional Network | Phase 3 모델 아키텍처 |
| Poze (Sports Technique Feedback) | 소량 데이터에서의 스포츠 자세 피드백 | 데이터 효율적 접근법 |
