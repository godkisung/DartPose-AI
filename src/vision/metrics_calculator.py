"""메트릭 계산 모듈.

ThrowPhases와 정규화된 관절 좌표를 받아
ThrowMetrics의 모든 필드를 계산합니다.

Huang et al. (2024) 논문 기반의 생체역학 지표를 계산합니다:
- 팔꿈치 드리프트 (Elbow Drift)
- 어깨 안정성 (Shoulder Stability)
- 몸통 흔들림 (Body Sway)
- 테이크백 각도 (Takeback Angle)
- 릴리즈 각도 (Release Angle)
- 팔로스루 각도 (Follow-through Angle)
- 최대 팔꿈치 확장 각속도 (Max Elbow Extension Velocity)
- 릴리즈 타이밍 (Release Timing)
- 손가락 스피드 (Finger Release Speed)
- 일관성 점수 (Consistency Score)
"""

import numpy as np
from src.models import ThrowAnalysis, ThrowPhases, ThrowMetrics, FrameData
from src.vision.pose_normalizer import PoseNormalizer


class MetricsCalculator:
    """단일 투구의 생체역학 메트릭을 계산하는 클래스.

    사용 예시:
        calc = MetricsCalculator(fps=30)
        metrics = calc.compute(frames, normalized_data, phases, side="right")
    """

    def __init__(self, fps: float = 30.0):
        """초기화.

        Args:
            fps: 영상 프레임레이트.
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self._normalizer = PoseNormalizer()

    # ─── Public API ─────────────────────────────────────────────────────────

    def compute(
        self,
        frames: list[FrameData],
        normalized_data: dict[str, np.ndarray],
        phases: ThrowPhases,
        side: str,
    ) -> ThrowMetrics:
        """ThrowMetrics의 모든 필드를 계산합니다.

        ※ 핵심 설계: 메트릭 유형에 따라 좌표 소스를 구분합니다.
        - 각도 메트릭 (takeback_angle, release_angle 등): **원시 keypoints** 사용
        - 위치 메트릭 (elbow_drift, body_sway 등): **정규화 좌표** 사용

        각도는 좌표계 변환에 불변하므로 정규화가 불필요하고,
        정규화 시 어깨 중점이 영점이 되어 오히려 각도를 파괴합니다.

        Args:
            frames: 단일 투구 세그먼트의 FrameData 리스트.
            normalized_data: PoseNormalizer.normalize()의 출력 (위치 메트릭용).
            phases: PhaseDetector.detect()의 출력.
            side: 투구 팔 방향 ('left' 또는 'right').

        Returns:
            ThrowMetrics — 모든 필드가 계산된 상태.
        """
        n = len(frames)

        # 프레임 절대 인덱스 → 로컬 인덱스 매핑
        frame_idx_map = {f.frame_index: i for i, f in enumerate(frames)}

        def localIdx(abs_frame: int) -> int:
            """절대 프레임 인덱스를 세그먼트 내 로컬 인덱스로 변환."""
            if abs_frame in frame_idx_map:
                return frame_idx_map[abs_frame]
            # 바로 근처 값으로 fallback
            closest = min(frame_idx_map.keys(), key=lambda k: abs(k - abs_frame))
            return frame_idx_map[closest]

        # 각 Phase의 로컬 인덱스
        addr_i    = localIdx(phases.address)
        tb_start_i = localIdx(phases.takeback_start)
        tb_max_i  = localIdx(phases.takeback_max)
        rel_i     = localIdx(phases.release)
        ft_i      = localIdx(phases.follow_through)

        # ── 원시 좌표 (각도 메트릭용) ──────────────────────────────────────
        raw_shoulder = self._extractRawJointCoords(frames, f"{side}_shoulder")
        raw_elbow    = self._extractRawJointCoords(frames, f"{side}_elbow")
        raw_wrist    = self._extractRawJointCoords(frames, f"{side}_wrist")
        raw_index    = self._extractRawJointCoords(frames, f"{side}_index_tip")

        # ── 정규화 좌표 (위치 메트릭용) ──────────────────────────────────────
        norm_shoulder = normalized_data.get(f"{side}_shoulder", np.zeros((n, 3)))
        norm_elbow    = normalized_data.get(f"{side}_elbow",    np.zeros((n, 3)))
        norm_hip      = normalized_data.get(f"{side}_hip",      np.zeros((n, 3)))

        # 팔꿈치 각도 시계열 (원시 좌표 기반 — 정확한 각도 산출)
        elbow_angles = np.array([
            self._normalizer._angle3d(raw_shoulder[i], raw_elbow[i], raw_wrist[i])
            for i in range(n)
        ])

        # ─ 1. 팔꿈치 드리프트 (위치 메트릭 → 정규화 좌표 사용) ──────────────
        # 투구 구간 동안 팔꿈치 이동의 비직선성을 상완 길이로 정규화
        elbow_drift = self._computeElbowDrift(norm_elbow, norm_shoulder, addr_i, ft_i)

        # ─ 2. 어깨 안정성 (위치 메트릭 → 정규화 좌표 사용) ──────────────────
        # 투구 구간 어깨 XY 좌표의 분산 합산
        shoulder_stability = self._computeJointVariance(norm_shoulder, addr_i, ft_i)

        # ─ 3. 몸통 흔들림 (위치 메트릭 → 정규화 좌표 사용) ──────────────────
        # 어드레스~릴리즈 구간의 엉덩이 X축 최대 변위
        body_sway = self._computeBodySway(norm_hip, addr_i, rel_i)

        # ─ 4. 테이크백 각도 (각도 메트릭 → 원시 좌표 기반) ──────────────────
        takeback_angle = float(elbow_angles[tb_max_i]) if elbow_angles[tb_max_i] > 0 else 0.0

        # ─ 5. 릴리즈 각도 (각도 메트릭 → 원시 좌표 기반) ────────────────────
        # 전완 벡터가 수평선과 이루는 각도 (MediaPipe Y축 반전)
        release_angle = self._computeReleaseAngle(raw_elbow[rel_i], raw_wrist[rel_i])

        # ─ 6. 팔로스루 각도 (각도 메트릭 → 원시 좌표 기반) ───────────────────
        follow_through_angle = float(elbow_angles[ft_i]) if elbow_angles[ft_i] > 0 else 0.0

        # ─ 7. 최대 팔꿈치 확장 각속도 (각도 메트릭 → 원시 좌표 기반) ──────────
        # 테이크백 정점 → 릴리즈 구간에서 팔꿈치 각속도의 최대값 (도/초)
        max_elbow_velocity = self._computeMaxElbowVelocity(
            elbow_angles, tb_max_i, rel_i
        )

        # ─ 8. 릴리즈 타이밍 (Release Timing ms) ──────────────────────────────
        # 테이크백 정점 → 릴리즈 까지 걸린 시간 (ms)
        timing_ms = (rel_i - tb_max_i) * self.dt * 1000.0
        # 방어: 음수나 0이면 최소 1프레임 값으로
        timing_ms = max(timing_ms, self.dt * 1000.0)

        # ─ 9. 손가락 릴리즈 스피드 (원시 좌표 기반) ─────────────────────────
        # raw_index에서 모두 0이면 감지 실패
        has_finger = not np.all(raw_index == 0)
        finger_speed = self._computeFingerSpeed(raw_index, rel_i) if has_finger else 0.0

        return ThrowMetrics(
            elbow_drift_norm=float(elbow_drift),
            shoulder_stability=float(shoulder_stability),
            body_sway=float(body_sway),
            takeback_angle_deg=float(takeback_angle),
            release_angle_deg=float(release_angle),
            follow_through_angle_deg=float(follow_through_angle),
            max_elbow_velocity_deg_s=float(max_elbow_velocity),
            release_timing_ms=float(timing_ms),
            finger_release_speed=float(finger_speed),
        )

    def computeConsistencyScore(self, analyses: list[ThrowAnalysis]) -> float:
        """세션 내 여러 투구 간 일관성 점수를 계산합니다 (0~100점).

        각 메트릭의 표준편차를 정규화하여 종합 점수를 산출합니다.
        점수가 높을수록 일관된 투구 폼을 의미합니다.

        Args:
            analyses: 세션 내 모든 ThrowAnalysis 리스트.

        Returns:
            일관성 점수 (0~100). 투구가 1개면 100.0 반환.
        """
        if len(analyses) < 2:
            return 100.0

        def normStd(values: list[float], scale: float) -> float:
            """표준편차를 주어진 스케일로 정규화 (0~1)."""
            if len(values) < 2:
                return 0.0
            return min(1.0, np.std(values) / scale)

        # 주요 메트릭의 표준편차 계산 (각 스케일로 정규화)
        takeback_std = normStd(
            [a.metrics.takeback_angle_deg for a in analyses if a.metrics.takeback_angle_deg > 0],
            scale=20.0,  # 20도 이상 차이나면 완전히 불일치
        )
        velocity_std = normStd(
            [a.metrics.max_elbow_velocity_deg_s for a in analyses if a.metrics.max_elbow_velocity_deg_s > 0],
            scale=100.0,  # 100 deg/s 이상 차이나면 완전히 불일치
        )
        timing_std = normStd(
            [a.metrics.release_timing_ms for a in analyses if a.metrics.release_timing_ms > 0],
            scale=100.0,  # 100ms 이상 차이나면 완전히 불일치
        )
        release_angle_std = normStd(
            [a.metrics.release_angle_deg for a in analyses],
            scale=15.0,  # 15도 이상 차이나면 완전히 불일치
        )

        # 가중 평균으로 불일치 점수 계산 (낮을수록 일관적)
        inconsistency = (
            takeback_std      * 0.30 +
            velocity_std      * 0.30 +
            timing_std        * 0.20 +
            release_angle_std * 0.20
        )

        # 일관성 점수 = 1 - 불일치 점수 (0~100)
        return float((1.0 - inconsistency) * 100.0)

    # ─── Private Metric Calculators ─────────────────────────────────────────

    def _computeElbowDrift(
        self,
        elbow: np.ndarray,
        shoulder: np.ndarray,
        start_i: int,
        end_i: int,
    ) -> float:
        """투구 중 팔꿈치 흔들림(드리프트)을 상완 길이로 정규화하여 계산합니다.

        팔꿈치가 이상적으로는 같은 궤도를 따라야 합니다.
        XY 평면에서 팔꿈치의 이동 비직선성을 측정하고,
        상완 길이(어깨-팔꿈치 거리 중앙값)로 나누어 체격 차이를 정규화합니다.

        (검토 의견: P1(테이크백)과 P2(릴리즈) 사이 거리를 상완 길이로
         정규화해야 체격이 다른 사용자 간 비교가 가능함)

        Args:
            elbow: 팔꿈치 정규화 좌표 배열 (N, 3).
            shoulder: 어깨 정규화 좌표 배열 (N, 3).
            start_i, end_i: 측정 구간 로컬 인덱스.

        Returns:
            팔꿈치 드리프트 (상완 길이 정규화 단위, 무차원).
        """
        segment = elbow[start_i:end_i + 1, :2]  # XY만 사용
        if len(segment) < 2:
            return 0.0

        # 경로 총 길이 (연속된 점 간 거리 합)
        diffs = np.diff(segment, axis=0)
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

        # 시작~끝 직선 거리
        direct_dist = float(np.linalg.norm(segment[-1] - segment[0]))

        # 드리프트 = 경로 길이 - 직선 거리 (클수록 팔꿈치가 많이 흔들린 것)
        raw_drift = max(0.0, path_length - direct_dist)

        # 상완 길이(Humerus)로 정규화: 어깨-팔꿈치 거리의 중앙값
        humerus_lengths = np.linalg.norm(
            shoulder[start_i:end_i + 1, :2] - elbow[start_i:end_i + 1, :2],
            axis=1,
        )
        median_humerus = float(np.median(humerus_lengths))

        # 유효하지 않은 스케일 방지
        if median_humerus < 1e-5:
            return raw_drift

        return raw_drift / median_humerus

    def _computeJointVariance(
        self,
        joint_coords: np.ndarray,
        start_i: int,
        end_i: int,
    ) -> float:
        """관절의 XY 좌표 분산 합산을 계산합니다.

        전체 구간에서 관절이 얼마나 고정되어 있는지를 나타냅니다.
        클수록 관절이 불안정합니다.

        Args:
            joint_coords: 관절 정규화 좌표 배열 (N, 3).
            start_i, end_i: 측정 구간 로컬 인덱스.

        Returns:
            XY 좌표 분산 합산.
        """
        segment = joint_coords[start_i:end_i + 1, :2]
        if len(segment) < 2:
            return 0.0
        return float(np.var(segment[:, 0]) + np.var(segment[:, 1]))

    def _computeBodySway(
        self,
        hip: np.ndarray,
        start_i: int,
        end_i: int,
    ) -> float:
        """몸통 좌우 흔들림(body sway)을 계산합니다.

        어드레스~릴리즈 구간에서 엉덩이 X좌표의 최대 변위를 측정합니다.
        클수록 투구 중 몸이 많이 흔들린 것입니다.

        Args:
            hip: 엉덩이 정규화 좌표 배열 (N, 3).
            start_i, end_i: 측정 구간 로컬 인덱스.

        Returns:
            X축 최대 변위 (정규화 단위).
        """
        segment = hip[start_i:end_i + 1, 0]  # X축만
        if len(segment) < 2:
            return 0.0
        return float(np.max(segment) - np.min(segment))

    def _computeReleaseAngle(
        self,
        elbow_at_release: np.ndarray,
        wrist_at_release: np.ndarray,
    ) -> float:
        """릴리즈 시점의 전완(forearm) 각도를 계산합니다.

        전완 벡터(팔꿈치→손목)가 수평선(X축)과 이루는 각도를 측정합니다.
        MediaPipe 좌표계에서 Y축은 아래가 +이므로 arctan2 계산 시 Y를 반전합니다.

        Args:
            elbow_at_release: 릴리즈 시점의 팔꿈치 좌표.
            wrist_at_release: 릴리즈 시점의 손목 좌표.

        Returns:
            수평 기준 전완 각도 (도). 양수 = 위로 향함, 음수 = 아래로 향함.
        """
        forearm_vec = wrist_at_release - elbow_at_release
        if np.linalg.norm(forearm_vec) < 1e-8:
            return 0.0

        # -Y: MediaPipe Y축 반전 (위가 +이 되도록)
        # abs(X): 투구 방향(좌/우)에 무관하게 0~90 범위
        angle = np.degrees(np.arctan2(-forearm_vec[1], abs(forearm_vec[0])))
        return float(angle)

    def _computeMaxElbowVelocity(
        self,
        elbow_angles: np.ndarray,
        tb_max_i: int,
        rel_i: int,
    ) -> float:
        """테이크백 정점 ~ 릴리즈 구간의 최대 팔꿈치 각속도를 계산합니다.

        이 값이 클수록 팔을 빠르게 펴는 것을 의미하며,
        다트 속도와 직결되는 핵심 지표입니다.

        Args:
            elbow_angles: 팔꿈치 각도 시계열 (도).
            tb_max_i: 테이크백 정점 로컬 인덱스.
            rel_i: 릴리즈 로컬 인덱스.

        Returns:
            최대 팔꿈치 각속도 (도/초).
        """
        end = min(rel_i + 1, len(elbow_angles))
        segment = elbow_angles[tb_max_i:end]

        if len(segment) < 2:
            return 0.0

        # 각속도 = 각도의 1차 미분
        velocities = np.gradient(segment, self.dt)

        # 최대 양의 각속도만 (팔이 펴지는 방향)
        positive_velocities = velocities[velocities > 0]
        if len(positive_velocities) == 0:
            return 0.0

        return float(np.max(positive_velocities))

    def _computeFingerSpeed(
        self,
        index_tip: np.ndarray,
        rel_i: int,
        window: int = 3,
    ) -> float:
        """릴리즈 전후 손가락(검지) 이동 속도를 계산합니다.

        손가락이 감지된 경우에만 유효한 값을 반환합니다.
        릴리즈 직전~직후 ±window 프레임에서 손가락 속도의 최대값을 사용합니다.

        Args:
            index_tip: 검지 끝 원시 좌표 배열 (N, 3).
            rel_i: 릴리즈 로컬 인덱스.
            window: 릴리즈 전후 탐색 범위 (프레임).

        Returns:
            손가락 속도 (원시 좌표 단위/초). 감지 불가 시 0.0.
        """
        n = len(index_tip)
        start = max(0, rel_i - window)
        end = min(n - 1, rel_i + window)

        segment = index_tip[start:end + 1]

        # 손가락 좌표가 모두 0이면 감지 실패
        if np.all(segment == 0):
            return 0.0

        # 손가락 XY 속도
        displacements = np.diff(segment[:, :2], axis=0)
        if len(displacements) == 0:
            return 0.0

        speeds = np.linalg.norm(displacements, axis=1) / self.dt
        return float(np.max(speeds))

    # ─── Raw Coordinate Extraction ───────────────────────────────────────────

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
