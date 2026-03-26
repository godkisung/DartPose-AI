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

# ─── ThrowSegmenter (FSM 기반) ────────────────────────────────────────────────

# 가우시안 스무딩 강도 (초 단위 표준편차)
# 각도 시계열의 노이즈를 제거합니다.
# 30fps 기준: 0.10초 = 약 3프레임 스무딩 (각도는 속도보다 안정적이므로 더 넓게)
SEGMENTER_SMOOTHING_SIGMA = 0.06

# ── FSM 상태 전환 임계값 ──

# IDLE → COCKING 전환: 팔꿈치 각도가 최근 안정 각도에서 이 값 이상 감소하면
# "팔을 접기 시작"으로 판정 (도 단위)
SEGMENTER_ANGLE_DROP_THRESHOLD = 10.0

# COCKING → RELEASING 전환 후, RELEASING → FOLLOW_THROUGH 전환:
# 테이크백 최저 각도에서 이 값 이상 각도가 증가하면 "팔 펴기 완료"로 판정 (도 단위)
SEGMENTER_RELEASE_ANGLE_RISE = 15.0

# FOLLOW_THROUGH → IDLE 전환: 각도 변화가 이 프레임 수 동안 안정되면
# 한 투구 사이클 완료로 판정
SEGMENTER_IDLE_STABILITY_FRAMES = 8

# 유효 투구로 인정되는 최소 팔꿈치 각도 변화 (도)
# 단순 손 들기/내리기를 필터링하는 용도
SEGMENTER_MIN_COCKING_ANGLE = 12.0

# 한 투구 사이클의 최소 프레임 수 (너무 짧으면 노이즈)
SEGMENTER_MIN_SEGMENT_FRAMES = 10

# 한 투구 사이클의 최소 시간 (초) — 연속 투구 간 쿨다운
SEGMENTER_MIN_THROW_INTERVAL_S = 1.0

# 유효 세그먼트의 최대 지속 시간 (초) — 이보다 긴 세그먼트는 분할 시도
SEGMENTER_MAX_SEGMENT_DURATION_S = 4.0

# 세그먼트 확장 시간 (초) — 피크 전후 여유 프레임 확보
SEGMENTER_SEGMENT_PAD_S = 0.5

# 간격이 이 프레임 수 이하인 세그먼트는 하나로 병합
SEGMENTER_MERGE_GAP_FRAMES = -1

# ── Legacy Peak 감지 파라미터 (A/B 비교용) ──
SEGMENTER_LEGACY_MIN_PEAK_PROMINENCE = 0.010
SEGMENTER_LEGACY_MIN_PEAK_DISTANCE = 1.5
SEGMENTER_LEGACY_SEGMENT_EXPAND_FRAMES = 1.2

# ─── PhaseDetector ────────────────────────────────────────────────────────────

# (현재 PhaseDetector는 fps만 파라미터로 받고, 내부 로직에서 비율 기반 처리)

# ─── MetricsCalculator ───────────────────────────────────────────────────────

# 릴리즈 각도 계산 시 전완 벡터의 최소 유효 크기 (정규화 단위)
METRICS_MIN_FOREARM_VEC_NORM = 1e-5

# ─── Validation Thresholds ────────────────────────────────────────────────────

# 투구로 인정되는 손목 최소 변위 (정규화 단위)
VALIDATION_MIN_WRIST_DISPLACEMENT = 0.07

# 투구로 인정되는 최소 팔꿈치 굽힘 각도 (도)
VALIDATION_MIN_TAKEBACK_ANGLE = 5.0

# 노이즈 사이클 필터링: 이 값을 초과하는 팔꿈치 각속도는 추적 오류로 판단 (도/초)
VALIDATION_MAX_ELBOW_VELOCITY = 2000.0

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
