"""다트 투구 분석 규칙 엔진.

생체역학 논문(Huang et al., 2024) 기반 관절 각도 및 속도 분석과
연속 투구 자동 분리(multi-throw segmentation) 기능을 제공합니다.
"""

import numpy as np
from src.models import (
    FrameData, Keypoints, ThrowPhases, ThrowMetrics, ThrowAnalysis, SessionResult,
)
from src.config import (
    ELBOW_STABILITY_THRESHOLD,
    TAKEBACK_MIN_ANGLE,
    TAKEBACK_MAX_ANGLE,
    ELBOW_EXTENSION_VEL_MIN,
    BODY_SWAY_THRESHOLD,
    SHOULDER_STABILITY_THRESHOLD,
    THROW_MIN_FRAMES,
    THROW_IDLE_FRAMES,
    VELOCITY_SMOOTHING_WINDOW,
)


class PoseRuleEngine:
    """다트 투구 생체역학 분석 엔진."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    # ─── Public API ────────────────────────────────────────────────

    def analyze_session(self, frames: list[FrameData]) -> SessionResult:
        """전체 세션(여러 투구)을 분석합니다.

        Args:
            frames: PoseExtractor에서 추출된 FrameData 리스트.

        Returns:
            SessionResult: 세션 전체 분석 결과.
        """
        # keypoints가 있는 프레임만 추출
        valid_frames = [f for f in frames if f.keypoints is not None]

        if len(valid_frames) < THROW_MIN_FRAMES:
            return SessionResult(
                total_frames=len(frames),
                fps=self.fps,
                total_throws_detected=0,
                throws=[],
            )

        throwing_arm = self._detect_throwing_arm(valid_frames)
        throw_segments = self._segment_throws(valid_frames, throwing_arm)

        analyses: list[ThrowAnalysis] = []
        for idx, segment in enumerate(throw_segments):
            analysis = self._analyze_single_throw(segment, throwing_arm, throw_index=idx + 1)
            if analysis is not None:
                analyses.append(analysis)

        # 여러 투구 간 일관성 분석
        if len(analyses) >= 2:
            self._add_consistency_issues(analyses)

        return SessionResult(
            total_frames=len(frames),
            fps=self.fps,
            total_throws_detected=len(analyses),
            throws=analyses,
        )

    # ─── Throw Segmentation ───────────────────────────────────────

    def _segment_throws(
        self, frames: list[FrameData], throwing_side: str,
    ) -> list[list[FrameData]]:
        """연속 투구를 자동으로 분리합니다.

        손목의 속도(velocity magnitude)를 기준으로 투구 동작을 감지합니다.
        속도가 높은 구간 = 투구 동작, 속도가 낮은 구간 = 유휴 상태.
        """
        wrist_key = f"{throwing_side}_wrist"

        # 손목 XY 좌표 시계열 추출
        positions = []
        for f in frames:
            wrist = f.keypoints.get(wrist_key)
            if wrist:
                positions.append((wrist[0], wrist[1]))
            elif positions:
                positions.append(positions[-1])
            else:
                positions.append((0.0, 0.0))

        positions = np.array(positions)

        # 프레임 간 속도(변위 크기) 계산
        displacements = np.diff(positions, axis=0)
        velocities = np.linalg.norm(displacements, axis=1)

        # 평활화
        kernel = np.ones(VELOCITY_SMOOTHING_WINDOW) / VELOCITY_SMOOTHING_WINDOW
        smoothed = np.convolve(velocities, kernel, mode='same')

        # 속도 임계값: 중앙값의 2배를 활성 구간으로 판단
        threshold = max(np.median(smoothed) * 2.0, 0.002)

        # 활성 구간 (투구 중) 검출
        is_active = smoothed > threshold
        segments: list[list[FrameData]] = []
        current_segment: list[FrameData] = []
        idle_count = 0

        for i, active in enumerate(is_active):
            if active:
                idle_count = 0
                current_segment.append(frames[i])
            else:
                idle_count += 1
                if current_segment:
                    if idle_count <= 3:
                        # 짧은 유휴는 같은 투구에 포함
                        current_segment.append(frames[i])
                    else:
                        # 투구 종료
                        if len(current_segment) >= THROW_MIN_FRAMES:
                            segments.append(current_segment)
                        current_segment = []

        # 마지막 세그먼트 처리
        if current_segment and len(current_segment) >= THROW_MIN_FRAMES:
            segments.append(current_segment)

        # fallback: 세그먼트가 없으면 전체를 하나의 투구로
        if not segments:
            segments = [frames]

        return segments

    # ─── Single Throw Analysis ────────────────────────────────────

    def _analyze_single_throw(
        self,
        frames: list[FrameData],
        throwing_side: str,
        throw_index: int,
    ) -> ThrowAnalysis | None:
        """단일 투구 구간을 분석합니다."""
        if len(frames) < THROW_MIN_FRAMES:
            return None

        phases = self._detect_phases(frames, throwing_side)
        if phases is None:
            return None

        metrics = self._compute_metrics(frames, phases, throwing_side)
        issues = self._evaluate_issues(metrics)

        return ThrowAnalysis(
            throw_index=throw_index,
            throwing_arm=throwing_side,
            frame_range=(frames[0].frame_index, frames[-1].frame_index),
            phases=phases,
            metrics=metrics,
            issues=issues,
        )

    # ─── Phase Detection ──────────────────────────────────────────

    def _detect_phases(self, frames: list[FrameData], side: str) -> ThrowPhases | None:
        """투구 단계(address, takeback, release, follow-through)를 감지합니다.

        개선: 손목 X축 위치 대신 속도 peak/valley 기반으로 Phase를 감지합니다.
        """
        if len(frames) < 10:
            return None

        wrist_key = f"{side}_wrist"
        x_positions = []

        for f in frames:
            wrist = f.keypoints.get(wrist_key)
            if wrist:
                x_positions.append(wrist[0])
            elif x_positions:
                x_positions.append(x_positions[-1])
            else:
                x_positions.append(0.0)

        x_arr = np.array(x_positions)

        # 평활화
        kernel = np.ones(5) / 5
        smoothed = np.convolve(x_arr, kernel, mode='same')

        n = len(frames)

        # 투구 방향 결정 (X 변위 방향)
        diff = smoothed[-1] - smoothed[0]
        if diff > 0:
            # 오른쪽으로 던짐: takeback = min X, release = max X (뒤에서)
            takeback_idx = int(np.argmin(smoothed))
            release_idx = int(np.argmax(smoothed[takeback_idx:])) + takeback_idx
        else:
            # 왼쪽으로 던짐: takeback = max X, release = min X (뒤에서)
            takeback_idx = int(np.argmax(smoothed))
            release_idx = int(np.argmin(smoothed[takeback_idx:])) + takeback_idx

        # fallback
        if takeback_idx == release_idx or takeback_idx == 0:
            takeback_idx = int(n * 0.4)
            release_idx = int(n * 0.6)

        takeback_start = max(0, takeback_idx - int(n * 0.15))
        follow_through = min(n - 1, release_idx + int(n * 0.15))

        # 절대 프레임 인덱스로 변환
        return ThrowPhases(
            address=frames[0].frame_index,
            takeback_start=frames[takeback_start].frame_index,
            takeback_max=frames[takeback_idx].frame_index,
            release=frames[release_idx].frame_index,
            follow_through=frames[follow_through].frame_index,
        )

    # ─── Metrics Computation ──────────────────────────────────────

    def _compute_metrics(
        self,
        frames: list[FrameData],
        phases: ThrowPhases,
        side: str,
    ) -> ThrowMetrics:
        """생체역학 수치 지표를 계산합니다."""
        wrist = f"{side}_wrist"
        elbow = f"{side}_elbow"
        shoulder = f"{side}_shoulder"
        hip = f"{side}_hip"
        index = f"{side}_index"

        # 프레임 인덱스 → 로컬 인덱스 매핑 (빠른 검색용)
        frame_map = {f.frame_index: i for i, f in enumerate(frames)}

        def local_idx(frame_idx: int) -> int | None:
            return frame_map.get(frame_idx)

        def get_joint(local_i: int, joint: str) -> np.ndarray | None:
            if 0 <= local_i < len(frames) and frames[local_i].keypoints:
                coords = frames[local_i].keypoints.get(joint)
                return np.array(coords) if coords else None
            return None

        # 1. 팔꿈치 Y 안정성 (address ~ follow_through 구간)
        elbow_y = []
        for f in frames:
            if f.keypoints:
                kp = f.keypoints.get(elbow)
                if kp:
                    elbow_y.append(kp[1])
        elbow_stability = float(np.var(elbow_y)) if elbow_y else 0.0

        # 2. 테이크백 최소 각도 (takeback_start ~ release 구간)
        tb_start_local = local_idx(phases.takeback_start) or 0
        rel_local = local_idx(phases.release) or (len(frames) - 1)

        takeback_angles = []
        for i in range(tb_start_local, min(rel_local + 1, len(frames))):
            angle = self._angle_3d(
                get_joint(i, shoulder), get_joint(i, elbow), get_joint(i, wrist),
            )
            if angle > 0:
                takeback_angles.append(angle)
        min_takeback = min(takeback_angles) if takeback_angles else 0.0

        # 3. 릴리즈 시 팔꿈치 펴는 각속도 (절대값 — 속도의 크기만 측정)
        vel_elbow = 0.0
        vel_wrist = 0.0

        # 가용 윈도우 크기 결정 (±3 선호, ±2 fallback, ±1 최소)
        window = 0
        for w in [3, 2, 1]:
            if rel_local is not None and w < rel_local < len(frames) - w:
                window = w
                break

        if window > 0 and rel_local is not None:
            dt = (frames[rel_local + window].timestamp_ms - frames[rel_local - window].timestamp_ms) / 1000.0
            if dt > 0:
                ang_before = self._angle_3d(
                    get_joint(rel_local - window, shoulder),
                    get_joint(rel_local - window, elbow),
                    get_joint(rel_local - window, wrist),
                )
                ang_after = self._angle_3d(
                    get_joint(rel_local + window, shoulder),
                    get_joint(rel_local + window, elbow),
                    get_joint(rel_local + window, wrist),
                )
                vel_elbow = abs(ang_after - ang_before) / dt

                # 4. 손목 스냅 속도
                w_before = self._angle_3d(
                    get_joint(rel_local - window, elbow),
                    get_joint(rel_local - window, wrist),
                    get_joint(rel_local - window, index),
                )
                w_after = self._angle_3d(
                    get_joint(rel_local + window, elbow),
                    get_joint(rel_local + window, wrist),
                    get_joint(rel_local + window, index),
                )
                vel_wrist = abs(w_after - w_before) / dt

        # 5. 몸통 흔들림
        addr_local = local_idx(phases.address) or 0
        sway = 0.0
        s_start = get_joint(addr_local, shoulder)
        s_end = get_joint(rel_local or 0, shoulder)
        if s_start is not None and s_end is not None:
            sway = float(abs(s_end[0] - s_start[0]))

        # 6. 어깨 안정성
        shoulder_y = []
        for f in frames:
            if f.keypoints:
                kp = f.keypoints.get(shoulder)
                if kp:
                    shoulder_y.append(kp[1])
        shoulder_stability = float(np.var(shoulder_y)) if shoulder_y else 0.0

        return ThrowMetrics(
            elbow_stability_variance=elbow_stability,
            takeback_min_angle_deg=min_takeback,
            elbow_extension_velocity_deg_s=vel_elbow,
            wrist_snap_velocity_deg_s=vel_wrist,
            body_sway_x_norm=sway,
            shoulder_stability_variance=shoulder_stability,
        )

    def _evaluate_issues(self, m: ThrowMetrics) -> list[str]:
        """임계값 기반으로 이슈를 판별합니다."""
        issues = []
        if m.elbow_stability_variance > ELBOW_STABILITY_THRESHOLD:
            issues.append("elbow_unstable_y")
        if 0 < m.takeback_min_angle_deg < TAKEBACK_MIN_ANGLE:
            issues.append("takeback_too_deep")
        if m.takeback_min_angle_deg > TAKEBACK_MAX_ANGLE:
            issues.append("takeback_too_shallow")
        if 0 < m.elbow_extension_velocity_deg_s < ELBOW_EXTENSION_VEL_MIN:
            issues.append("slow_elbow_extension")
        if m.body_sway_x_norm > BODY_SWAY_THRESHOLD:
            issues.append("body_sway_detected")
        if m.shoulder_stability_variance > SHOULDER_STABILITY_THRESHOLD:
            issues.append("shoulder_unstable")
        return issues

    # ─── Cross-Throw Consistency ────────────────────────────────

    def _add_consistency_issues(self, analyses: list[ThrowAnalysis]):
        """여러 투구 간 일관성을 분석하고 이슈를 추가합니다."""
        if len(analyses) < 2:
            return

        # 테이크백 각도 분산
        tb_angles = [a.metrics.takeback_min_angle_deg for a in analyses if a.metrics.takeback_min_angle_deg > 0]
        if len(tb_angles) >= 2 and np.std(tb_angles) > 15:
            for a in analyses:
                if "inconsistent_takeback" not in a.issues:
                    a.issues.append("inconsistent_takeback")

        # 팔꿈치 펴는 속도 분산
        elbow_vels = [a.metrics.elbow_extension_velocity_deg_s for a in analyses if a.metrics.elbow_extension_velocity_deg_s > 0]
        if len(elbow_vels) >= 2 and np.std(elbow_vels) > 80:
            for a in analyses:
                if "inconsistent_elbow_speed" not in a.issues:
                    a.issues.append("inconsistent_elbow_speed")

    # ─── Helpers ──────────────────────────────────────────────────

    def _detect_throwing_arm(self, frames: list[FrameData]) -> str:
        """투구 팔(좌/우)을 자동 감지합니다."""
        right_positions = []
        left_positions = []

        for f in frames:
            if f.keypoints:
                rw = f.keypoints.get("right_wrist")
                lw = f.keypoints.get("left_wrist")
                if rw:
                    right_positions.append(rw[:2])
                if lw:
                    left_positions.append(lw[:2])

        if not right_positions or not left_positions:
            return "right"

        r_var = np.var([p[0] for p in right_positions]) + np.var([p[1] for p in right_positions])
        l_var = np.var([p[0] for p in left_positions]) + np.var([p[1] for p in left_positions])

        return "right" if r_var > l_var else "left"

    @staticmethod
    def _angle_3d(p1, p2, p3) -> float:
        """3점에서 p2를 꼭짓점으로 하는 3D 각도(도)를 계산합니다."""
        if p1 is None or p2 is None or p3 is None:
            return 0.0
        v1 = p1 - p2
        v2 = p3 - p2
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    # ─── Legacy compatibility (for standalone script) ─────────────

    def feed_frame(self, frame_index, timestamp_ms, keypoints_dict):
        """Legacy: 기존 호출 호환용."""
        pass  # 새 API에서는 analyze_session(frames) 사용

    def analyze_throw(self):
        """Legacy: 기존 호출 호환용."""
        return {"error": "Use analyze_session() with FrameData list instead"}
