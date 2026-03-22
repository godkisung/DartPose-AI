# AI Dart Coach 🎯

다트 투구 자세를 실시간으로 분석하고, 생체역학적 데이터를 기반으로 코칭 피드백을 제공하는 AI 프로젝트입니다.  
**본 프로젝트의 최종 목표는 'iOS 온디바이스(On-device) 애플리케이션' 개발이며, 현재 레포지토리는 핵심 분석 알고리즘 검증을 위한 파이썬 프로토타입입니다.**

## 🎯 주요 기능

- **관절 추출**: MediaPipe Pose로 어깨-팔꿈치-손목 3D 좌표를 프레임 단위 추출 (iOS Vision 프레임워크 전환 대비)
- **자동 투구 분리**: 연속 투구 영상에서 개별 투구를 자동으로 감지·분리 (손목 속도 및 팔꿈치 각속도 Multi-Signal Fusion)
- **페이즈 감지**: 투구의 4단계(Address, Takeback, Release, Follow-through)를 정밀 분석
- **규칙 엔진**: 논문(Huang et al., 2024) 기반 생체역학 지표 분석
  - 팔꿈치 안정성, 테이크백 각도, 릴리즈 속도, 손목 스냅, 상체 흔들림
  - 투구 간 일관성 분석 (테이크백, 팔꿈치 속도)
- **AI 코칭**: 룰 엔진 기반 템플릿 코칭 피드백 (LLM 구조 Skeleton 유지)

---

## 📷 앱 연동을 위한 촬영 가이드 (클라이언트 제어)

AI 모델의 인식 정확도를 극대화하기 위한 앱 내 UI/UX 구현 안내입니다.

### 🟢 권장 각도 (Best Practice)
투구 팔이 카메라 방향을 향하고, 몸에 가려지지 않게 촬영해야 합니다. (투구 100% 탐지 보장)
- **오른손잡이 (Right-handed)**: 플레이어 기준 우측 30~90도 대각선 또는 측면에서 촬영.
- **왼손잡이 (Left-handed)**: 플레이어 기준 좌측 30~90도 대각선 또는 측면에서 촬영.

### 🟡 피해야 할 각도 (Occlusion 현상)
투구 팔이 플레이어의 몸통 뒤에 가려지는 화각은 피해야 합니다.
- 예: 오른손잡이 투구자를 좌측에서 촬영할 경우 어깨와 몸통 모델이 겹쳐 인식률이 급감할 수 있습니다.
- 분석 엔진 내부에서 "손목 속도 + 팔꿈치 각속도 융합" 알고리즘을 통해 가림 현상을 보정하도록 되어 있으나, 원활한 피드백을 위해선 클라이언트가 올바른 방향에서 촬영하도록 유도해야 합니다.

*(추가 자료인 `output/midframes/`의 중간 프레임 스냅샷을 앱 내 안내 가이드에 첨부하여 고객이 직관적으로 이해할 수 있도록 제공하시길 권장합니다.)*

---

## 📂 프로젝트 구조

```
├── src/
│   ├── main.py                 # 파이프라인 엔트리 포인트
│   ├── models.py               # 데이터 스키마 (dataclass)
│   ├── config.py               # 전역 설정/임계값
│   ├── vision/
│   │   ├── mediapipe_pose.py   # MediaPipe 관절 추출기 (PoseExtractor)
│   │   ├── throw_segmenter.py  # 투구 세분화 (손목 속도+각도 융합)
│   │   ├── phase_detector.py   # 투구 4-Phase 감지 (각도 기반)
│   │   ├── metrics_calculator.py # 생체역학 지표 계산
│   │   ├── dart_analyzer.py    # 분석 로직 통합 시스템
│   │   └── rule_engine.py      # 생체역학 분석 엔진 레거시
│   ├── hardware/
│   │   └── serial_receiver.py  # Arduino 통신 + 키보드 시뮬레이터
│   ├── llm/
│   │   └── feedback_generator.py # 템플릿 코칭 피드백
│   └── utils/
│       └── video_utils.py      # 비디오 코덱 변환 (H.264 자동 변환)
├── tests/                      # 단위 회귀 테스트
├── data/                       # 입력 비디오 (미제공 시 사용자 추가 필요)
├── output/                     # 분석 결과 (JSON + 디버그 그래프)
├── paper/                      # 참고 논문 세트
└── docs/                       # 기술/아키텍처 스펙 문서
```

## 🛠️ 설치 및 실행

### 의존성 설치
본 프로젝트는 속도와 재현성을 위해 파이썬 패키지 매니저로 `uv`를 사용합니다.
```bash
uv sync
```

### 비디오 분석 (자동)
인자를 입력하지 않으면 `data/` 폴더 내 가장 최신 mp4 파일을 자동으로 분석합니다.
```bash
uv run python src/main.py
```

### 특정 비디오 분석 및 디버그 그래프 출력
분석 중간 과정(속도 및 각도 피크 그래프)을 시각화하려면 `--debug-plot`을 추가하세요.
```bash
uv run python src/main.py --input data/sample_1.mp4 --debug-plot
```

### 튜닝 회귀 테스트 실행
기존 감지 정확도가 회귀하지 않는지 테스트하는 도구입니다.
```bash
uv run pytest -s tests/test_regression.py
```

## 🤝 협업 가이드

- 아키텍처스펙 문서: [darts_architecture_spec.md](darts_architecture_spec.md) 참고
- 하드웨어 및 임베디드 연동: [HARDWARE_INTERFACE.md](HARDWARE_INTERFACE.md) 참고
- 규칙 엔진 임계값 조정: `src/config.py` 수정
- LLM 설정 정보: `GEMINI.md` 참고
