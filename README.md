# AI Dart Coach 🎯

다트 투구 자세를 실시간으로 분석하고, 생체역학적 데이터를 기반으로 코칭 피드백을 제공하는 AI 프로젝트입니다.  
**본 프로젝트의 최종 목표는 'iOS 온디바이스(On-device) 애플리케이션' 개발이며, 현재 레포지토리는 핵심 분석 알고리즘 검증을 위한 파이썬 프로토타입입니다.**

## 🎯 주요 기능

- **관절 추출**: MediaPipe Pose로 어깨-팔꿈치-손목 3D 좌표를 프레임 단위 추출 (iOS Vision 프레임워크 전환 대비)
- **자동 투구 분리**: 연속 투구 영상에서 개별 투구를 자동으로 감지·분리
- **규칙 엔진**: 논문(Huang et al., 2024) 기반 생체역학 지표 분석
  - 팔꿈치 안정성, 테이크백 각도, 릴리즈 속도, 손목 스냅, 상체 흔들림
  - 투구 간 일관성 분석 (테이크백, 팔꿈치 속도)
- **AI 코칭**: 룰 엔진 기반 템플릿 코칭 피드백 (LLM 구조 Skeleton 유지)

## 📂 프로젝트 구조

```
├── src/
│   ├── main.py                 # 파이프라인 엔트리 포인트
│   ├── models.py               # 데이터 스키마 (dataclass)
│   ├── config.py               # 전역 설정/임계값
│   ├── vision/
│   │   ├── mediapipe_pose.py   # MediaPipe 관절 추출기 (PoseExtractor)
│   │   └── rule_engine.py      # 생체역학 분석 엔진 (PoseRuleEngine)
│   ├── hardware/
│   │   └── serial_receiver.py  # Arduino 통신 + 키보드 시뮬레이터
│   ├── llm/
│   │   └── feedback_generator.py # LLM/템플릿 코칭 피드백
│   └── utils/
│       └── video_utils.py      # 비디오 코덱 변환 유틸
├── tests/                      # 단위 테스트
├── data/                       # 입력 비디오
├── output/                     # 분석 결과 (JSON + 영상)
├── models/                     # 모델 가중치 파일
└── paper/                      # 참고 논문
```

## 🛠️ 설치 및 실행

### 의존성 설치
```bash
uv sync
```

### 비디오 분석 (기본)
```bash
uv run python src/main.py --mode video --input data/sample_1.mp4
```

### 비디오 분석 + 스켈레톤 오버레이 + JSON 리포트
```bash
uv run python src/main.py --mode video \
  --input data/sample_1.mp4 \
  --output-video output/result.mp4 \
  --output-json output/report.json
```

### 라이브 모드 (키보드 시뮬레이터)
```bash
uv run python src/main.py --mode live --simulate
```

### 라이브 모드 (실제 Arduino)
```bash
uv run python src/main.py --mode live --port /dev/ttyUSB0
```

### 테스트 실행
```bash
uv run pytest tests/ -v
```

## 📊 출력 예시

```
📊 다트 투구 분석 리포트 (3회 투구)
==================================================

🎯 투구 #1 (left손)
   · 팔꿈치 안정성: 0.0001
   · 테이크백 최소 각도: 110.1°
   · 팔꿈치 펴는 속도: 0.0°/s
   · 손목 스냅 속도: 0.0°/s

   💡 개선 포인트:
     → 테이크백이 너무 얕습니다...
     → 투구 시 상체가 좌우로 흔들리고 있습니다...
```

## 🤝 협업 가이드

- 하드웨어 연동: [HARDWARE_INTERFACE.md](HARDWARE_INTERFACE.md) 참고
- 규칙 엔진 임계값 조정: `src/config.py` 수정
- 새 분석 지표 추가: `src/vision/rule_engine.py`의 `_compute_metrics()` 확장
