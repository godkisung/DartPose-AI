"""프로젝트 전역 설정.

하드코딩된 임계값과 상수를 중앙에서 관리합니다.
"""

# ─── Rule Engine 임계값 ────────────────────────────────────────────────
# Huang et al., 2024 논문 기반 + 정규화 좌표 보정

ELBOW_STABILITY_THRESHOLD = 0.005      # 팔꿈치 Y좌표 분산 상한 (정규화)
TAKEBACK_MIN_ANGLE = 30                # 테이크백 각도 하한 (도)
TAKEBACK_MAX_ANGLE = 110               # 테이크백 각도 상한 (도)
ELBOW_EXTENSION_VEL_MIN = 150          # 릴리즈 시 팔꿈치 펴는 각속도 하한 (도/초)
BODY_SWAY_THRESHOLD = 0.05            # 몸통 흔들림 X변위 상한 (정규화)
SHOULDER_STABILITY_THRESHOLD = 0.003   # 어깨 Y좌표 분산 상한 (정규화)

# ─── Multi-Throw Segmentation ─────────────────────────────────────────

THROW_MIN_FRAMES = 15                  # 하나의 투구로 인정되는 최소 프레임 수
THROW_IDLE_FRAMES = 10                 # 투구 사이 유휴 상태 최소 프레임 수
VELOCITY_SMOOTHING_WINDOW = 7          # 속도 평활화 윈도우 크기

# ─── MediaPipe ─────────────────────────────────────────────────────────

MEDIAPIPE_MODEL_COMPLEXITY = 2         # 모델 정확도 (0=가벼움, 1=보통, 2=정확)
MEDIAPIPE_MIN_DETECTION_CONF = 0.5
MEDIAPIPE_MIN_TRACKING_CONF = 0.5

# ─── Hardware (Serial) ─────────────────────────────────────────────────

SERIAL_BAUDRATE = 115200
SERIAL_DEFAULT_PORT = "/dev/ttyUSB0"
SERIAL_TIMEOUT = 1                     # 시리얼 읽기 타임아웃 (초)
SERIAL_RECONNECT_DELAY = 2            # 재연결 시도 간격 (초)

# ─── LLM ───────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
LLM_TIMEOUT = 30                       # LLM 응답 대기 타임아웃 (초)
