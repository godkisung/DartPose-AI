# AI Dart Coach

다트 투구 자세를 실시간으로 분석하고, 생체역학적 데이터를 기반으로 코칭 피드백을 제공하는 AI 프로젝트입니다. 카메라를 통한 3D 관절 추적(MediaPipe)과 하드웨어 보드의 타격 센서 연동, 그리고 LLM을 활용한 개인화 피드백 시스템을 결합하여 구축됩니다.

## 🎯 주요 기능
- **비전 센싱 (Vision)**: MediaPipe를 활용하여 어깨 내회전, 팔꿈치 폄 속도, 손목 스냅(Palmar Flexion) 각속도 등 투구 시 핵심적인 3D 생체역학 지표를 실시간 추출합니다.
- **규칙 엔진 (Rule Engine)**: 학술 논문(Huang et al., 2024 등) 기반의 고급 선수(Advanced Player) 자세 임계값을 바탕으로 투구의 불안정성과 개선점을 수학적으로 찾아냅니다.
- **하드웨어 연동 (Hardware)**: Arduino 기반의 다트 타격 센서와 시리얼 통신을 통해 투구 종료(Hit)를 정확하게 감지하고 영상을 분할합니다.
- **AI 코칭 (LLM)**: 앞선 생체역학 데이터를 분석하여, 초보자도 이해하기 쉬운 자연어 코칭 피드백을 실시간으로 제공합니다.

## 📂 프로젝트 구조
```
├── src/                  # 메인 소스 코드 리포지토리
│   ├── hardware/         # Arduino 및 타격 센서 시리얼 통신 모듈
│   ├── vision/           # MediaPipe / YOLO 3D 관절 추적 및 룰 엔진
│   ├── llm/              # LLM 피드백 생성 모듈
│   └── main.py           # 전체 시스템 통합 엔트리 포인트
├── models/               # 포즈 에스티메이션 모델 (YOLOv8 pt 파일 등)
├── data/                 # 테스트 비디오 및 데이터 분절 폴더 (git ignore 대상)
├── output/               # 시각화 결과 및 영상 저장 (git ignore 대상)
├── tests/                # 단위 테스트
└── README.md             # 프로젝트 소개
```

## 🛠️ 설치 및 실행 방법

본 프로젝트는 의존성 관리 및 패키지 실행을 위해 **`uv`** 를 사용합니다.

### 1. 의존성 설치
```bash
# Python 최신 환경에서 uv를 통해 프로젝트 의존성을 설치합니다.
uv sync
```

### 2. 실행
```bash
# 동영상 기반 관절 추출 및 규칙 엔진 테스트
uv run src/vision/mediapipe_pose.py --input data/test_video.mp4 --output output/result.mp4

# [개발중] 전체 시스템 통합 실행 (카메라 + 하드웨어 센서 연동)
uv run src/main.py --mode live
```

## 🤝 협업 가이드
- 팀원 간 변경사항은 `feature/*` 브랜치를 생성하여 작업하고 PR을 통해 병합합니다.
- 코드를 추가할 때는 각 모듈(`src/hardware`, `vision`, `llm` 등)의 역할과 책임을 분리해 주세요.
- 물리적인 전자/하드웨어 연동을 담당하는 엔지니어는 `HARDWARE_INTERFACE.md` 문서를 먼저 참고해 주세요.
