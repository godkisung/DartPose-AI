"""투구 페이즈 감지 모듈.

단일 투구 세그먼트에서 4단계(Address, Takeback, Release, Follow-through)를
정밀하게 감지합니다.

핵심 개선 사항:
- 기존: thumb_tip 첫 출현 = 릴리즈 → 손가락 감지 불안정으로 항상 1프레임 오차
- 개선 v2: 하이브리드 릴리즈 스코어링
  Score_release = w1·Peak(θ̇_elbow) + w2·ZeroCrossing(ẍ_wrist)
  가속도 단독이 아닌 복합 가중치 스코어링 방식으로 노이즈 증폭 리스크 억제.

알고리즘 근거 (Huang et al., 2024):
- 릴리즈 순간 = 팔꿈치가 가장 빠르게 펴지는 순간 (각속도 최대)
- Takeback 정점 = 팔꿈치 각도 최솟값 (팔이 가장 많이 접힌 순간)

설계 원칙:
- FSM Loose 모드: Address가 감지되지 않아도 Takeback 피크가 확실하면 투구 인정
- 페이즈 타임아웃: 특정 페이즈에 2초 이상 머물면 강제 전환
"""

import numpy as np
from src.models import FrameData, ThrowPhases
from src.vision.pose_normalizer import PoseNormalizer


# ─── 하이브리드 릴리즈 스코어링 가중치 ──────────────────────────────────────
# Score = w1·(각속도 정규화) + w2·(손목-팔꿈치 거리 변화율 정규화)
# 가속도(2차 미분)는 가중치를 낮게 설정하여 노이즈 증폭 리스크 억제
_W_ELBOW_ANGULAR_VEL = 0.65   # 팔꿈치 각속도 피크 가중치 (Primary)
_W_FOREARM_EXTENSION = 0.35   # 손목-팔꿈치 거리 변화율 가중치 (Secondary)

# 페이즈 타임아웃 (초): 특정 페이즈에 이 시간 이상 머물면 강제 전환
_PHASE_TIMEOUT_S = 2.0


class PhaseDetector:
    """투구 단계(Phase)를 감지하는 클래스.

    FSM Loose 모드:
    - Address가 없어도 Takeback 피크가 확실하면 투구로 인정
    - 각 Phase에 타임아웃 적용 (2초 초과 시 강제 전환)

    사용 예시:
        detector = PhaseDetector(fps=30)
        phases = detector.detect(frames, normalized_data, side="right")
    """

    def __init__(self, fps: float = 30.0):
        """초기화.

        Args:
            fps: 영상 프레임레이트.
        """
        self.fps = fps
        self.dt = 1.0 / fps  # 프레임 간격 (초)
        self._normalizer = PoseNormalizer()
        # FPS 기반 타임아웃 프레임 수 계산
        self._phase_timeout_frames = int(_PHASE_TIMEOUT_S * fps)

    # ─── Public API ─────────────────────────────────────────────────────────

    def detect(
        self,
        frames: list[FrameData],
        normalized_data: dict[str, np.ndarray],
        side: str,
    ) -> ThrowPhases | None:
        """투구 세그먼트에서 4-Phase 경계를 감지합니다.

        FSM Loose 모드: Address가 없어도 Takeback 피크만 확실하면
        투구로 인정하여, 바로 던지는 투구자도 누락하지 않습니다.

        ※ 핵심 설계: 각도 계산은 **원시 keypoints**에서 직접 수행합니다.
        정규화 좌표(어깨 중점 차감 → 스케일링)는 각도를 왜곡하므로,
        각도는 원시 좌표에서, 위치 기반 분석은 정규화 좌표에서 수행합니다.

        Args:
            frames: 단일 투구 세그먼트의 FrameData 리스트.
            normalized_data: PoseNormalizer.normalize()의 출력 (Phase에서는 미사용).
            side: 투구 팔 방향 ('left' 또는 'right').

        Returns:
            ThrowPhases 객체. 감지 실패 시 None 반환.
        """
        n = len(frames)
        if n < 10:
            return None

        # ※ 원시 keypoints에서 직접 좌표 추출 (정규화 좌표는 각도 왜곡 야기)
        shoulder = self._extractRawJointCoords(frames, f"{side}_shoulder")
        elbow    = self._extractRawJointCoords(frames, f"{side}_elbow")
        wrist    = self._extractRawJointCoords(frames, f"{side}_wrist")

        # ─ Step 1: 팔꿈치 각도 시계열 계산 (원시 좌표 기반) ────────────────
        elbow_angles = np.array([
            self._normalizer._angle3d(shoulder[i], elbow[i], wrist[i])
            for i in range(n)
        ])

        # 각도가 전부 0이면 데이터 불량
        if np.all(elbow_angles < 1.0):
            return None

        # ─ Step 2: 팔꿈치 각속도 계산 (1차 미분만 사용, 원시 좌표 기반) ─────
        # ※ 2차 미분(가속도)은 노이즈 증폭이 심하므로 직접 사용하지 않음
        angle_velocity = np.gradient(elbow_angles, self.dt)  # shape: (N,)

        # ─ Step 3: 손목-팔꿈치 거리 변화율 계산 (릴리즈 보조 신호, 원시 좌표) ─
        # 가속도(2차 미분) 대신 더 안정적인 1차 미분 기반 신호
        forearm_lengths = np.linalg.norm(wrist - elbow, axis=1)  # shape: (N,)
        forearm_rate = np.gradient(forearm_lengths, self.dt)      # shape: (N,)

        # ─ Step 4: 스무딩 (FPS 적응형 윈도우) ────────────────────────────────
        # 윈도우 크기를 FPS에 비례하게 설정 (30fps→5, 60fps→9)
        smooth_window = max(3, int(self.fps / 6.0)) | 1  # 홀수 보장
        smooth_angles   = self._movingAverage(elbow_angles, window=smooth_window)
        smooth_velocity = self._movingAverage(angle_velocity, window=smooth_window)
        smooth_forearm  = self._movingAverage(forearm_rate, window=smooth_window)

        # ─ Step 5: 테이크백 정점 감지 ──────────────────────────────────────
        # Takeback max = 팔꿈치 각도 최솟값 (팔이 가장 많이 접힌 순간)
        # 앞쪽 75% 구간에서만 탐색 (릴리즈 이후 구간 배제)
        search_end = max(5, int(n * 0.75))
        takeback_max_local = int(np.argmin(smooth_angles[:search_end]))

        # ─ Step 6: 하이브리드 릴리즈 스코어링 ──────────────────────────────
        release_local = self._detectReleaseHybrid(
            smooth_velocity, smooth_forearm, takeback_max_local, n
        )

        # ─ Step 7: 타임아웃 보정 ──────────────────────────────────────────
        # 릴리즈가 테이크백 정점으로부터 너무 먼 경우 강제 제한
        max_release_span = self._phase_timeout_frames
        if release_local - takeback_max_local > max_release_span:
            release_local = min(n - 1, takeback_max_local + max_release_span)

        # ─ Step 8: 나머지 Phase 경계 결정 (Loose FSM) ────────────────────────
        # Address: 세그먼트 시작 (Loose: Address 없어도 투구 인정)
        address_local = 0

        # Takeback start: 각도가 감소하기 시작하는 점 (address ~ takeback_max 사이)
        takeback_start_local = self._findTakebackStart(
            smooth_angles, address_local, takeback_max_local
        )

        # Loose FSM: takeback_start가 takeback_max와 같으면
        # Address가 없는 것 → takeback_start = address로 설정
        if takeback_start_local >= takeback_max_local:
            takeback_start_local = address_local

        # Follow-through: 릴리즈 이후 각도가 다시 안정되는 시점
        follow_through_local = self._findFollowThrough(
            smooth_angles, release_local, n
        )

        # 팔로스루 타임아웃 보정
        if follow_through_local - release_local > max_release_span:
            follow_through_local = min(n - 1, release_local + max_release_span)

        # ─ Step 9: 절대 프레임 인덱스로 변환 ───────────────────────────────
        def to_abs(local_i: int) -> int:
            return frames[min(local_i, n - 1)].frame_index

        return ThrowPhases(
            address=to_abs(address_local),
            takeback_start=to_abs(takeback_start_local),
            takeback_max=to_abs(takeback_max_local),
            release=to_abs(release_local),
            follow_through=to_abs(follow_through_local),
        )

    def extractAllMetricSignals(
        self,
        frames: list[FrameData],
        normalized_data: dict[str, np.ndarray],
        side: str,
    ) -> dict[str, np.ndarray]:
        """메트릭 계산에 필요한 모든 신호를 한 번에 추출합니다.

        MetricsCalculator에서 사용하기 위한 헬퍼 메서드.
        ※ 각도 계산은 원시 keypoints에서 직접 수행합니다.

        Returns:
            딕셔너리 키:
            - 'elbow_angle': 팔꿈치 각도 시계열 (도)
            - 'elbow_velocity': 팔꿈치 각속도 시계열 (도/초)
            - 'wrist_speed': 손목 XY 속도 크기 시계열
        """
        n = len(frames)
        # ※ 원시 keypoints에서 직접 추출 (정규화 좌표는 각도를 왜곡)
        shoulder = self._extractRawJointCoords(frames, f"{side}_shoulder")
        elbow    = self._extractRawJointCoords(frames, f"{side}_elbow")
        wrist    = self._extractRawJointCoords(frames, f"{side}_wrist")

        # 팔꿈치 각도 (원시 좌표 기반)
        elbow_angles = np.array([
            self._normalizer._angle3d(shoulder[i], elbow[i], wrist[i])
            for i in range(n)
        ])

        # 팔꿈치 각속도 (1차 미분만)
        elbow_velocity = np.gradient(elbow_angles, self.dt)

        # 손목 XY 속도 (원시 좌표 기반)
        wrist_displacements = np.diff(wrist[:, :2], axis=0)
        wrist_speed = np.concatenate([[0.0], np.linalg.norm(wrist_displacements, axis=1)])

        return {
            "elbow_angle": elbow_angles,
            "elbow_velocity": elbow_velocity,
            "wrist_speed": wrist_speed,
        }

    # ─── Phase Detection Helpers ────────────────────────────────────────────

    def _detectReleaseHybrid(
        self,
        angle_velocity: np.ndarray,
        forearm_rate: np.ndarray,
        takeback_max: int,
        n: int,
    ) -> int:
        """하이브리드 스코어링으로 릴리즈 순간을 감지합니다.

        기존: 팔꿈치 각속도 피크 단독 사용
        개선: 팔꿈치 각속도 피크 + 손목-팔꿈치 거리 변화율의 가중합
              (가속도(2차 미분) 대신 더 안정적인 1차 미분 기반 보조 신호 사용)

        Score(t) = w1·norm(θ̇_elbow(t)) + w2·norm(d/dt ||wrist-elbow||(t))

        Args:
            angle_velocity: 팔꿈치 각속도 시계열 (deg/s, 스무딩 후).
            forearm_rate: 손목-팔꿈치 거리 변화율 (스무딩 후).
            takeback_max: 테이크백 정점 로컬 인덱스.
            n: 전체 프레임 수.

        Returns:
            릴리즈 로컬 인덱스.
        """
        # 테이크백 정점 이후 구간 (릴리즈는 반드시 이후에 발생)
        search_start = takeback_max + 1
        if search_start >= n - 1:
            return n - 1

        region_vel = angle_velocity[search_start:]
        region_arm = forearm_rate[search_start:]

        if len(region_vel) == 0:
            return n - 1

        # 각 신호를 0~1 범위로 정규화 (max-min scaling)
        def normalizeSignal(sig: np.ndarray) -> np.ndarray:
            """신호를 0~1 범위로 정규화합니다. 범위가 0이면 0 배열 반환."""
            vmin, vmax = sig.min(), sig.max()
            if vmax - vmin < 1e-10:
                return np.zeros_like(sig)
            return (sig - vmin) / (vmax - vmin)

        norm_vel = normalizeSignal(region_vel)
        norm_arm = normalizeSignal(region_arm)

        # 하이브리드 스코어 계산
        hybrid_score = (
            _W_ELBOW_ANGULAR_VEL * norm_vel +
            _W_FOREARM_EXTENSION * norm_arm
        )

        # 최대 스코어 시점 = 릴리즈
        local_release = int(np.argmax(hybrid_score))
        release_idx = search_start + local_release

        # 검증: 팔이 실제로 펴지는 동작이 있는지 확인
        if angle_velocity[release_idx] > 0:
            return release_idx

        # Fallback: 각속도 최대값만 사용
        fallback_local = int(np.argmax(region_vel))
        fallback_idx = search_start + fallback_local
        if angle_velocity[fallback_idx] > 0:
            return fallback_idx

        # 최종 Fallback: 테이크백 정점에서 40% 진행 시점
        return min(n - 1, takeback_max + max(1, int((n - takeback_max) * 0.4)))

    def _findTakebackStart(
        self,
        smooth_angles: np.ndarray,
        address_local: int,
        takeback_max_local: int,
    ) -> int:
        """팔꿈치 각도가 감소하기 시작하는 시점(테이크백 시작)을 찾습니다.

        address ~ takeback_max 구간을 역방향 탐색하여 각도가
        최대인 시점을 찾습니다 (팔이 펴진 상태에서 접기 시작하는 순간).

        Loose FSM 대응: 이 구간이 매우 짧으면 (Address 없이 바로 Takeback)
        address_local을 그대로 반환합니다.

        Args:
            smooth_angles: 스무딩된 팔꿈치 각도 시계열.
            address_local: 어드레스 로컬 인덱스.
            takeback_max_local: 테이크백 정점 로컬 인덱스.

        Returns:
            테이크백 시작 로컬 인덱스.
        """
        search_region = smooth_angles[address_local:takeback_max_local + 1]
        if len(search_region) < 3:
            # 구간이 너무 짧음 → Address 없이 바로 Takeback (Loose FSM)
            return address_local

        # address ~ takeback_max 구간에서 각도 최댓값 = 팔이 가장 펴져있는 순간
        local_max = int(np.argmax(search_region))
        return address_local + local_max

    def _findFollowThrough(
        self,
        smooth_angles: np.ndarray,
        release_local: int,
        n: int,
    ) -> int:
        """팔로스루 완료 시점을 감지합니다.

        릴리즈 이후 팔꿈치 각도가 다시 안정되는 시점:
        각도 변화율이 작아지는 시점 (팔이 완전히 펴진 상태로 수렴).

        Args:
            smooth_angles: 스무딩된 팔꿈치 각도 시계열.
            release_local: 릴리즈 로컬 인덱스.
            n: 전체 프레임 수.

        Returns:
            팔로스루 로컬 인덱스.
        """
        search_start = release_local + 1
        if search_start >= n:
            return n - 1

        search_region = smooth_angles[search_start:]
        if len(search_region) < 2:
            return n - 1

        # 릴리즈 이후 각도 최댓값 시점 (팔이 가장 많이 펴진 순간)
        local_max = int(np.argmax(search_region))

        # 팔로스루 = 릴리즈 이후 최대 신전 시점
        return min(n - 1, search_start + local_max)

    # ─── Utility ────────────────────────────────────────────────────────────

    @staticmethod
    def _extractRawJointCoords(
        frames: list[FrameData],
        joint_name: str,
    ) -> np.ndarray:
        """FrameData 리스트에서 특정 관절의 원시 좌표 배열을 추출합니다.

        정규화 없이 MediaPipe 0~1 좌표를 그대로 사용합니다.
        누락된 값(None)은 직전 유효값으로 채웁니다(forward fill).

        Args:
            frames: FrameData 리스트.
            joint_name: 관절 이름 (예: 'right_shoulder').

        Returns:
            shape (N, 3) numpy 배열.
        """
        n = len(frames)
        coords = np.zeros((n, 3), dtype=np.float64)
        last_valid = np.zeros(3, dtype=np.float64)

        for i, frame in enumerate(frames):
            if frame.keypoints:
                val = frame.keypoints.get(joint_name)
                if val is not None:
                    last_valid = np.array(val, dtype=np.float64)
            coords[i] = last_valid

        return coords

    @staticmethod
    def _movingAverage(signal: np.ndarray, window: int) -> np.ndarray:
        """이동평균 스무딩.

        커널 크기는 반드시 홀수로 유지합니다.
        (vDSP.convolve 포팅 시 1:1 매칭을 위함)

        Args:
            signal: 1D numpy 배열.
            window: 윈도우 크기 (홀수 권장, 짝수면 자동 보정).

        Returns:
            스무딩된 1D numpy 배열 (길이 동일).
        """
        if window <= 1:
            return signal
        # vDSP 포팅을 위해 커널 크기를 홀수로 강제
        window = window | 1
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode="same")
