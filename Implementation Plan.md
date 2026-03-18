# AI Dart Coach — 구현 계획서 (Python Prototype)

## 목표

현재 파이썬 기반의 AI Dart Coach 프로토타입을 고도화하여, **최종 목표인 iOS 온디바이스(On-device) 애플리케이션** 개발을 위한 핵심 생체역학 분석 알고리즘과 룰 엔진을 검증합니다.

**핵심 제약 조건**:
- 3주 마감, MediaPipe 전용, CLI 기반 (알고리즘 검증 목적)
- Multi-throw (3연속 투구) 자동 분리 지원
- 비전 기반 분석 및 템플릿 피드백 집중 (향후 Swift 이식을 위해 로직 모듈화)



## Proposed Changes

### Phase 0: 비디오 코덱 수정

#### [NEW] [video_utils.py](file:///home/kisung/workspace/2_applications/darts/src/utils/video_utils.py)
- `ffmpeg` 기반 AV1 → H.264 자동 변환 함수
- 입력 비디오의 코덱을 감지하여 필요 시 자동 변환 후 임시 파일 경로 반환
- MediaPipe 파이프라인 진입 전에 호출

---

### Phase 1: 아키텍처 & 파이프라인

#### [NEW] [models.py](file:///home/kisung/workspace/2_applications/darts/src/models.py)
**`dataclass` 기반 데이터 스키마** — 모듈 간 데이터 전달의 명시적 계약:
```python
@dataclass
class Keypoints:         # 단일 프레임의 관절 좌표
class FrameData:         # 프레임 인덱스 + 타임스탬프 + Keypoints
class ThrowPhases:       # address, takeback, release 등 인덱스
class ThrowMetrics:      # 팔꿈치 안정도, 테이크백 각도 등 수치
class ThrowAnalysis:     # ThrowPhases + ThrowMetrics + issues
class SessionResult:     # 전체 세션 (여러 투구) 분석 결과
```

#### [NEW] [config.py](file:///home/kisung/workspace/2_applications/darts/src/config.py)
**설정 파일** — 하드코딩된 임계값/상수를 중앙 관리:
```python
# Rule Engine 임계값
ELBOW_STABILITY_THRESHOLD = 0.005
TAKEBACK_MIN_ANGLE = 30
TAKEBACK_MAX_ANGLE = 110
ELBOW_EXTENSION_VEL_MIN = 150
BODY_SWAY_THRESHOLD = 0.05

# MediaPipe 설정
MEDIAPIPE_MODEL_COMPLEXITY = 2
MEDIAPIPE_MIN_DETECTION_CONF = 0.5
```

#### [MODIFY] [mediapipe_pose.py](file:///home/kisung/workspace/2_applications/darts/src/vision/mediapipe_pose.py)
**리팩토링**:
- 현재 하는 일: 비디오 I/O + MediaPipe 추론 + 렌더링 + 룰엔진 호출 + 출력 (God Function)
- **변경**: `PoseExtractor` 클래스로 재설계
  - `extract_from_video(path) → list[FrameData]`: 관절 추출만 담당
  - `extract_from_frame(frame) → Keypoints | None`: 단일 프레임 처리 (라이브 모드용)
  - 비디오 I/O, 렌더링, 룰엔진 호출은 `main.py`의 파이프라인으로 이동
- `video_utils.py`와 연동하여 AV1 자동 변환

#### [MODIFY] [rule_engine.py](file:///home/kisung/workspace/2_applications/darts/src/vision/rule_engine.py)
**Multi-throw 지원 리팩토링**:
- `PoseRuleEngine` → 입력: `list[FrameData]`, 출력: `SessionResult`
- `segment_throws(frames) → list[list[FrameData]]`: 연속 투구 자동 분리 (wrist velocity valley 기반)
- `analyze_single_throw(frames) → ThrowAnalysis`: 단일 투구 분석 (기존 `analyze_throw` 개선)
- `analyze_session(frames) → SessionResult`: 전체 세션 (여러 투구) 분석
- 임계값을 `config.py`에서 참조
- Phase Detection 개선: velocity-based peak detection

#### [MODIFY] [feedback_generator.py](file:///home/kisung/workspace/2_applications/darts/src/llm/feedback_generator.py)
**템플릿 기반 피드백 우선순위**:
- 룰 엔진의 이슈 목록(`issues`)을 한국어 템플릿으로 변환하여 즉각적 피드백 제공
- LLM 연동은 구조적 형태(Skeleton)만 유지하고 우선순위에서 제외

#### [MODIFY] [main.py](file:///home/kisung/workspace/2_applications/darts/src/main.py)
**비디오 분석 전용 파이프라인**:
- `video` 모드 집중: 비디오 → 코덱 변환 → 관절 추출 → 투구 분리 → 분석 → 피드백 → JSON 리포트
- 하드웨어 연동 및 라이브 모드 제외

---

### Phase 2: 분석 고도화 (비전 중심)

#### [MODIFY] [rule_engine.py](file:///home/kisung/workspace/2_applications/darts/src/vision/rule_engine.py)
- Phase Detection: wrist XY velocity의 peak/valley를 `scipy.signal.find_peaks`로 감지
- 지표 확장:
  - **전완 Pronation**: 손목-검지-새끼 벡터의 회전 분석
  - **어깨 안정성**: 어깨 Y 좌표 분산 (hip 대비 정규화)
  - **릴리즈 포인트 일관성**: 연속 투구 간 릴리즈 위치 분산
- 투구 간 비교 분석 (일관성 점수)

---

### Phase 3: 폴리싱

#### [MODIFY] [main.py](file:///home/kisung/workspace/2_applications/darts/src/main.py)
- CLI 출력 개선 (컬러, 진행 바, 섹션 분리)
- `--output-json` 옵션으로 결과 파일 경로 지정

#### [NEW] test files
- `tests/test_rule_engine.py`: 핵심 분석 로직 단위 테스트
- `tests/test_models.py`: 데이터 모델 검증

#### README 업데이트

---

## Verification Plan

### 자동 테스트

```bash
# Phase 0 검증: 비디오 변환 후 프레임 읽기 테스트
uv run python -c "
from src.utils.video_utils import ensure_compatible_video
import cv2
path = ensure_compatible_video('data/sample_1.mp4')
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
assert ret, 'Frame read failed!'
print(f'Success: frame shape = {frame.shape}')
cap.release()
"

# Phase 1 검증: 전체 파이프라인 실행
uv run src/main.py --input data/sample_1.mp4

# 단위 테스트
uv run pytest tests/ -v
```

### 수동 검증

1. **Phase 0**: `uv run src/main.py --input data/sample_1.mp4` 실행 후 `"No data fed to engine"` 에러가 사라지고, 실제 분석 결과(metrics, issues)가 출력되는지 확인
2. **Phase 1**: 출력된 JSON에 `throws` 배열이 3개의 투구 분석 결과를 포함하는지 확인
3. **Phase 2**: 각 투구의 `phases_frames`가 합리적인 프레임 범위를 가지는지 확인
4. **LLM**: 향후 활성화 시 Ollama 연동 확인 (현재는 템플릿 기반 피드백 확인)
