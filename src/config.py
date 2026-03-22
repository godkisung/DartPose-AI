"""프로젝트 전역 설정.

하드코딩된 임계값과 상수를 중앙에서 관리합니다.
모든 임계값은 정규화된 좌표 기준입니다 (MediaPipe: 0~1 범위).
"""

# ─── MediaPipe ────────────────────────────────────────────────────────────────

MEDIAPIPE_MODEL_COMPLEXITY = 2          # 모델 정확도 (0=가벼움, 1=보통, 2=정확)
MEDIAPIPE_MIN_DETECTION_CONF = 0.5      # 포즈 초기 감지 최소 신뢰도
MEDIAPIPE_MIN_TRACKING_CONF = 0.5       # 포즈 추적 최소 신뢰도

# ─── PoseNormalizer ──────────────────────────────────────────────────────────

# 시간축 이동평균 윈도우 크기 (클수록 떨림 제거, 시간 해상도 감소)
NORMALIZER_SMOOTHING_WINDOW = 5

# ─── ThrowSegmenter ──────────────────────────────────────────────────────────

# 가우시안 스무딩 강도 (초 단위 표준편차)
# 클수록 노이즈 제거 강하지만 속도 피크 시간 해상도 감소
# 30fps 기준: 0.06초 = 약 2프레임 스무딩
SEGMENTER_SMOOTHING_SIGMA = 0.06

# 피크 돌출도 최소값 (정규화 좌표 단위의 속도)
# 손목이 이 값 이상의 속도 변화를 보여야 유효 투구 피크로 간주
# 실험값: 0.008 ~ 0.020 사이 (영상 해상도/거리에 따라 조정)
SEGMENTER_MIN_PEAK_PROMINENCE = 0.010

# 연속 피크 간 최소 시간 간격 (초)
# 한 투구의 최소 지속 시간. 1.5초 미만의 간격에서 두 번 투구는 불가능.
SEGMENTER_MIN_PEAK_DISTANCE = 1.5

# 피크 전후로 세그먼트를 확장하는 시간 (초)
# 투구는 테이크백부터 팔로스루까지 보통 1~2초이므로 ±1.2초 확장
SEGMENTER_SEGMENT_EXPAND_FRAMES = 1.2

# 유효 세그먼트의 최소 프레임 수 (너무 짧은 세그먼트 제거)
SEGMENTER_MIN_SEGMENT_FRAMES = 15

# 간격이 이 프레임 수 이하인 세그먼트는 하나로 병합
SEGMENTER_MERGE_GAP_FRAMES = 15

# ─── PhaseDetector ────────────────────────────────────────────────────────────

# (현재 PhaseDetector는 fps만 파라미터로 받고, 내부 로직에서 비율 기반 처리)

# ─── MetricsCalculator ───────────────────────────────────────────────────────

# 릴리즈 각도 계산 시 전완 벡터의 최소 유효 크기 (정규화 단위)
METRICS_MIN_FOREARM_VEC_NORM = 1e-5

# ─── Validation Thresholds ────────────────────────────────────────────────────

# 투구로 인정되는 손목 최소 변위 (정규화 단위)
VALIDATION_MIN_WRIST_DISPLACEMENT = 0.10

# 투구로 인정되는 최소 팔꿈치 굽힘 각도 (도)
VALIDATION_MIN_TAKEBACK_ANGLE = 10.0

# ─── Rule Engine 피드백 임계값 ────────────────────────────────────────────────
# (메트릭 계산 후 이슈 판별에 사용)

ELBOW_STABILITY_THRESHOLD = 0.005      # 팔꿈치 드리프트 상한 (정규화 단위)
TAKEBACK_MIN_ANGLE = 30                # 테이크백 각도 하한 (도)
TAKEBACK_MAX_ANGLE = 110               # 테이크백 각도 상한 (도)
ELBOW_EXTENSION_VEL_MIN = 150          # 릴리즈 시 팔꿈치 확장 각속도 하한 (도/초)
BODY_SWAY_THRESHOLD = 0.05             # 몸통 흔들림 X변위 상한 (정규화 단위)
SHOULDER_STABILITY_THRESHOLD = 0.003   # 어깨 분산 상한 (정규화 단위)

# ─── LLM (Skeleton) ───────────────────────────────────────────────────────────

THROW_MIN_FRAMES = 15                  # 하나의 투구로 인정되는 최소 프레임 수
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
LLM_TIMEOUT = 30                       # LLM 응답 대기 타임아웃 (초)

# ─── Legacy (하위 호환용 — 삭제 예정) ─────────────────────────────────────────

VELOCITY_SMOOTHING_WINDOW = 7          # 구 rule_engine.py 호환용
THROW_IDLE_FRAMES = 10                 # 구 rule_engine.py 호환용
