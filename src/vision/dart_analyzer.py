"""다트 투구 분석 통합 엔진.

ThrowSegmenter, PhaseDetector, PoseNormalizer, MetricsCalculator를
조율하여 전체 분석 파이프라인을 실행합니다.

기존 AdvancedPoseEngine, PoseRuleEngine을 완전히 대체합니다.

파이프라인:
  FrameData 리스트
    → PoseNormalizer (좌표 정규화 + 흔들림 보정)
    → ThrowSegmenter (투구 구간 분리)
    → PhaseDetector (4-Phase 감지)
    → MetricsCalculator (생체역학 메트릭 계산)
    → SessionResult

--debug-plot 옵션 사용 시 각 단계의 중간 신호를 시각화합니다.
"""

import numpy as np
from typing import Optional

from src.models import (
    FrameData, ThrowAnalysis, ThrowPhases, SessionResult,
)
from src.config import (
    THROW_MIN_FRAMES,
    VALIDATION_MIN_WRIST_DISPLACEMENT,
    VALIDATION_MIN_TAKEBACK_ANGLE,
    VALIDATION_MAX_ELBOW_VELOCITY,
)
from src.vision.pose_normalizer import PoseNormalizer
from src.vision.throw_segmenter import ThrowSegmenter
from src.vision.phase_detector import PhaseDetector
from src.vision.metrics_calculator import MetricsCalculator


class DartAnalyzer:
    """다트 투구 분석 통합 엔진.

    사용 예시:
        analyzer = DartAnalyzer(fps=30)
        session_result = analyzer.analyze_session(frames)

        # 디버그 시각화 포함:
        analyzer = DartAnalyzer(fps=30, debug_plot=True)
        session_result = analyzer.analyze_session(frames, sample_name="sample_10")
    """

    def __init__(self, fps: float = 30.0, debug_plot: bool = False):
        """초기화.

        Args:
            fps: 입력 영상의 프레임레이트.
            debug_plot: True면 분석 후 디버그 시각화 그래프를 자동 생성.
        """
        self.fps = fps
        self.debug_plot = debug_plot
        self._normalizer = PoseNormalizer()
        self._segmenter = ThrowSegmenter(fps=fps)
        self._phase_detector = PhaseDetector(fps=fps)
        self._metrics_calc = MetricsCalculator(fps=fps)

        # 디버그 시각화 도구 (필요할 때만 import)
        self._plotter = None
        if debug_plot:
            from src.vision.debug_plotter import DebugPlotter
            self._plotter = DebugPlotter()

    # ─── Public API ─────────────────────────────────────────────────────────

    def analyze_session(
        self,
        frames: list[FrameData],
        sample_name: str = "sample",
    ) -> SessionResult:
        """전체 세션(연속 영상)을 분석하여 SessionResult를 반환합니다.

        Args:
            frames: PoseExtractor에서 추출된 FrameData 리스트.

        Returns:
            SessionResult: 세션 전체 분석 결과.
        """
        # keypoints가 있는 유효 프레임만 사용
        valid_frames = [f for f in frames if f.keypoints is not None]

        if len(valid_frames) < THROW_MIN_FRAMES:
            print(f"  ⚠ 유효 프레임 부족 ({len(valid_frames)}개) — 분석 중단")
            return SessionResult(
                total_frames=len(frames),
                fps=self.fps,
                total_throws_detected=0,
                throws=[],
            )

        # ─ Step 1: 투구 팔 자동 감지 ─────────────────────────────────────
        throwing_side = self._detectThrowingSide(valid_frames)
        print(f"  ℹ 투구 팔: {throwing_side}")

        # ─ Step 2: 좌표 정규화 (전체 세션) ──────────────────────────────
        normalized_data = self._normalizer.normalize(valid_frames, throwing_side)

        # ─ Step 3: 투구 세그먼트 분리 ────────────────────────────────────
        segments = self._segmenter.segment(valid_frames, normalized_data, throwing_side)
        print(f"  ℹ 세그먼트 후보: {len(segments)}개")

        # ─ Step 4: 각 세그먼트 분석 ──────────────────────────────────────
        analyses: list[ThrowAnalysis] = []
        for i, segment in enumerate(segments):
            print(f"  → 세그먼트 {i+1} 분석 중... (프레임 {segment[0].frame_index}~{segment[-1].frame_index})")

            analysis = self._analyzeSingleThrow(
                segment, throwing_side, throw_index=len(analyses) + 1
            )
            if analysis is None:
                print(f"    ✗ 분석 실패 — 건너뜀")
                continue

            # 유효성 검증
            is_valid, reason = self._validateThrow(analysis, segment, throwing_side)
            if is_valid:
                analyses.append(analysis)
                print(f"    ✓ 투구 {len(analyses)} 확정")
            else:
                print(f"    ✗ 기각 ({reason})")

        # ─ Step 5: 다중 투구 일관성 점수 계산 ────────────────────────────
        if len(analyses) >= 2:
            consistency = self._metrics_calc.computeConsistencyScore(analyses)
            for a in analyses:
                a.metrics.consistency_score = round(consistency, 1)
            print(f"  ℹ 일관성 점수: {consistency:.1f}/100")

        print(f"  ✅ 총 {len(analyses)}번 투구 감지 완료")

        session = SessionResult(
            total_frames=len(frames),
            fps=self.fps,
            total_throws_detected=len(analyses),
            throws=analyses,
        )

        # ─ Step 6: 디버그 시각화 (옵션) ────────────────────────────────────
        if self.debug_plot and self._plotter is not None:
            try:
                wrist_vel = getattr(self._segmenter, "_last_velocity", np.array([]))
                peak_idx = getattr(self._segmenter, "_last_peaks", [])
                self._plotter.plotSessionOverview(
                    frames=valid_frames,
                    wrist_velocity=wrist_vel,
                    peak_indices=peak_idx,
                    segments=segments,
                    session=session,
                    side=throwing_side,
                    sample_name=sample_name,
                )
            except Exception as e:
                print(f"  ⚠ 디버그 시각화 실패: {e}")

        return session

    # ─── Single Throw Analysis ────────────────────────────────────────────

    def _analyzeSingleThrow(
        self,
        frames: list[FrameData],
        side: str,
        throw_index: int,
    ) -> Optional[ThrowAnalysis]:
        """단일 투구 세그먼트를 분석합니다.

        Args:
            frames: 투구 세그먼트의 FrameData 리스트.
            side: 투구 팔 방향.
            throw_index: 투구 순번.

        Returns:
            ThrowAnalysis. 실패 시 None.
        """
        if len(frames) < THROW_MIN_FRAMES:
            return None

        # 세그먼트 단위로 정규화
        normalized = self._normalizer.normalize(frames, side)

        # 4-Phase 감지
        phases = self._phase_detector.detect(frames, normalized, side)
        if phases is None:
            return None

        # 메트릭 계산
        metrics = self._metrics_calc.compute(frames, normalized, phases, side)

        return ThrowAnalysis(
            throw_index=throw_index,
            throwing_arm=side,
            frame_range=(frames[0].frame_index, frames[-1].frame_index),
            phases=phases,
            metrics=metrics,
            issues=[],
        )

    # ─── Validation ──────────────────────────────────────────────────────

    def _validateThrow(
        self,
        analysis: ThrowAnalysis,
        frames: list[FrameData],
        side: str,
    ) -> tuple[bool, str]:
        """분석 결과가 유효한 투구인지 검증합니다.

        투구가 아닌 단순 움직임(손 들기, 자세 조정 등)을 필터링합니다.

        Args:
            analysis: 분석 결과.
            frames: 투구 세그먼트 프레임.
            side: 투구 팔 방향.

        Returns:
            (is_valid: bool, reason: str) 튜플.
        """
        wrist_key = f"{side}_wrist"

        # 손목 좌표 수집
        wrist_coords = []
        for f in frames:
            if f.keypoints:
                w = f.keypoints.get(wrist_key)
                if w:
                    wrist_coords.append(w[:2])

        if len(wrist_coords) < 5:
            return False, "손목 좌표 부족"

        coords = np.array(wrist_coords)

        # 검증 1: 손목 최대 변위
        max_disp = float(np.max(np.linalg.norm(coords - coords[0], axis=1)))
        if max_disp < VALIDATION_MIN_WRIST_DISPLACEMENT:
            return False, f"변위 부족 ({max_disp:.3f})"

        # 검증 2: 릴리즈 타이밍 (테이크백 이후 최소 1프레임 이상)
        if analysis.metrics.release_timing_ms < (1000 / self.fps):
            return False, f"릴리즈 타이밍 부족 ({analysis.metrics.release_timing_ms:.1f}ms)"

        # 검증 3: 팔꿈치 각도 변화 (팔을 접었다 펴는 동작이 있어야 함)
        if analysis.metrics.takeback_angle_deg < VALIDATION_MIN_TAKEBACK_ANGLE:
            return False, f"팔꿈치 굽힘 부족 ({analysis.metrics.takeback_angle_deg:.1f}°)"

        # 검증 4: 비현실적인 팔꿈치 각속도 (추적 오류/노이즈 사이클 필터링)
        if analysis.metrics.max_elbow_velocity_deg_s > VALIDATION_MAX_ELBOW_VELOCITY:
            return False, f"비현실적인 각속도 ({analysis.metrics.max_elbow_velocity_deg_s:.0f}°/s)"

        return True, ""

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _detectThrowingSide(self, frames: list[FrameData]) -> str:
        """투구 팔(좌/우)을 자동 감지합니다 (90도 정측면 촬영 기준).

        3개 기준의 다수결 투표로 결정합니다:
        1. Visibility : 투구 팔 관절의 평균 가시도가 더 높음 (정측면에서 투구 팔이 선명)
        2. Depth (Z)  : 투구 팔이 카메라에 더 가까워 Z값이 더 작음 (음수 방향)
        3. Variance   : 투구 손목의 XY 분산이 비투구 팔보다 현저히 큼

        각 기준에서 right가 우세하면 +1, left가 우세하면 -1.
        합계 >= 0이면 'right', < 0이면 'left' 반환.

        Args:
            frames: 유효 FrameData 리스트.

        Returns:
            투구 팔 방향 ('left' 또는 'right').
        """
        _ARM_JOINTS = ("shoulder", "elbow", "wrist")

        r_vis: list[float] = []
        l_vis: list[float] = []
        r_z:   list[float] = []
        l_z:   list[float] = []
        r_wrist_xy: list[list[float]] = []
        l_wrist_xy: list[list[float]] = []

        for f in frames:
            kp = f.keypoints
            if kp is None:
                continue
            for joint in _ARM_JOINTS:
                rc = kp.get(f"right_{joint}")
                lc = kp.get(f"left_{joint}")
                if rc:
                    r_z.append(rc[2])
                    if len(rc) >= 4:
                        r_vis.append(rc[3])
                if lc:
                    l_z.append(lc[2])
                    if len(lc) >= 4:
                        l_vis.append(lc[3])
            rw = kp.get("right_wrist")
            lw = kp.get("left_wrist")
            if rw:
                r_wrist_xy.append(rw[:2])
            if lw:
                l_wrist_xy.append(lw[:2])

        votes = 0
        log: list[str] = []

        # 기준 1: Visibility
        if r_vis and l_vis:
            rv, lv = float(np.mean(r_vis)), float(np.mean(l_vis))
            if rv > lv:
                votes += 1
                log.append(f"vis→R({rv:.2f}>{lv:.2f})")
            else:
                votes -= 1
                log.append(f"vis→L({lv:.2f}>{rv:.2f})")

        # 기준 2: Z축 깊이 (값이 작을수록 카메라에 가까움 = 투구 팔)
        if r_z and l_z:
            rz, lz = float(np.mean(r_z)), float(np.mean(l_z))
            if rz < lz:
                votes += 1
                log.append(f"z→R({rz:.3f}<{lz:.3f})")
            else:
                votes -= 1
                log.append(f"z→L({lz:.3f}<{rz:.3f})")

        # 기준 3: 손목 XY 분산
        if len(r_wrist_xy) >= 3 and len(l_wrist_xy) >= 3:
            rv2 = float(np.var(np.array(r_wrist_xy), axis=0).sum())
            lv2 = float(np.var(np.array(l_wrist_xy), axis=0).sum())
            if rv2 > lv2:
                votes += 1
                log.append(f"var→R({rv2:.4f}>{lv2:.4f})")
            else:
                votes -= 1
                log.append(f"var→L({lv2:.4f}>{rv2:.4f})")

        side = "right" if votes >= 0 else "left"
        print(f"  ℹ 투구 팔 감지: {side} (vote={votes:+d} | {', '.join(log)})")
        return side
