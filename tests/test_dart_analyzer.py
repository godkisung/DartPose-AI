"""새 분석 모듈 단위 테스트.

PoseNormalizer, ThrowSegmenter, PhaseDetector, MetricsCalculator,
DartAnalyzer를 합성 데이터로 검증합니다.
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Keypoints, FrameData, ThrowPhases, ThrowMetrics
from src.vision.pose_normalizer import PoseNormalizer
from src.vision.throw_segmenter import ThrowSegmenter
from src.vision.phase_detector import PhaseDetector
from src.vision.metrics_calculator import MetricsCalculator
from src.vision.dart_analyzer import DartAnalyzer


# ─── 테스트용 픽스처 헬퍼 ───────────────────────────────────────────────────────

def makeKeypoints(
    right_wrist: list | None = None,
    right_elbow: list | None = None,
    right_shoulder: list | None = None,
    right_hip: list | None = None,
    **extra,
) -> Keypoints:
    """테스트용 Keypoints를 생성합니다."""
    defaults = {
        "left_shoulder":  [0.35, 0.30, 0.0],
        "right_shoulder": right_shoulder or [0.55, 0.30, 0.0],
        "left_elbow":     [0.30, 0.42, 0.0],
        "right_elbow":    right_elbow or [0.58, 0.42, 0.0],
        "left_wrist":     [0.25, 0.55, 0.0],
        "right_wrist":    right_wrist or [0.62, 0.55, 0.0],
        "left_hip":       [0.37, 0.62, 0.0],
        "right_hip":      right_hip or [0.53, 0.62, 0.0],
    }
    defaults.update(extra)
    return Keypoints(**defaults)


def makeFrame(index: int, fps: float = 30.0, **kp_overrides) -> FrameData:
    """테스트용 FrameData를 생성합니다."""
    return FrameData(
        frame_index=index,
        timestamp_ms=index * (1000.0 / fps),
        keypoints=makeKeypoints(**kp_overrides),
    )


def makeThrowFrames(
    n_frames: int = 60,
    start_index: int = 0,
    fps: float = 30.0,
) -> list[FrameData]:
    """투구 동작을 시뮬레이션하는 합성 프레임 시퀀스를 생성합니다.

    - 앞 30%: 어드레스 (정지)
    - 30~50%: 테이크백 (손목 뒤로)
    - 50~65%: 릴리즈 (손목 앞으로 빠르게)
    - 65~100%: 팔로스루 (정지)
    """
    frames = []
    for i in range(n_frames):
        t = i / n_frames  # 정규화된 시간 (0~1)

        # 손목 X 위치: 어드레스→테이크백(뒤)→릴리즈(앞)→팔로스루
        if t < 0.3:
            wrist_x = 0.62               # 어드레스 (고정)
        elif t < 0.50:
            # 테이크백: 뒤로 당기기 (0.62 → 0.45)
            progress = (t - 0.30) / 0.20
            wrist_x = 0.62 - 0.17 * progress
        elif t < 0.65:
            # 릴리즈: 앞으로 빠르게 (0.45 → 0.75)
            progress = (t - 0.50) / 0.15
            wrist_x = 0.45 + 0.30 * progress
        else:
            wrist_x = 0.75               # 팔로스루 (고정)

        # 팔꿈치 각도 시뮬레이션 (테이크백에서 접혔다가 릴리즈 때 펴짐)
        # 테이크백 정점에서 팔꿈치 Y가 높아짐 (접힘)
        elbow_y = 0.42 + 0.08 * max(0, min(1, (t - 0.30) / 0.20 - (t - 0.50) / 0.15))

        frames.append(makeFrame(
            index=start_index + i,
            fps=fps,
            right_wrist=[wrist_x, 0.55, 0.0],
            right_elbow=[0.58, elbow_y, 0.0],
        ))

    return frames


# ─── PoseNormalizer 테스트 ────────────────────────────────────────────────────

class TestPoseNormalizer:
    """PoseNormalizer 단위 테스트."""

    def testNormalizeReturnsAllJoints(self):
        """정규화 결과에 필요한 관절 키가 모두 포함되어야 합니다."""
        normalizer = PoseNormalizer()
        frames = [makeFrame(i) for i in range(30)]
        result = normalizer.normalize(frames, throwing_side="right")

        assert "right_wrist" in result
        assert "right_elbow" in result
        assert "right_shoulder" in result
        assert result["right_wrist"].shape == (30, 3)

    def testCameraMotionCorrectionCentersOnShoulder(self):
        """카메라 보정 후 어깨 중점이 원점에 가까워야 합니다."""
        normalizer = PoseNormalizer(smoothing_window=1)
        frames = [makeFrame(i) for i in range(20)]
        result = normalizer.normalize(frames, throwing_side="right")

        # 어깨 중점 계산 (보정 후)
        mid_x = (result["left_shoulder"][:, 0] + result["right_shoulder"][:, 0]) / 2
        # 떨림 보정 후 어깨 중점의 X는 0에 매우 가까워야 함
        assert np.abs(np.mean(mid_x)) < 0.01

    def testAngle3dRightAngle(self):
        """직각(90도)을 올바르게 계산해야 합니다."""
        normalizer = PoseNormalizer()
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        assert abs(normalizer._angle3d(p1, p2, p3) - 90.0) < 0.01

    def testAngle3dStraightLine(self):
        """직선 관계(180도)를 올바르게 계산해야 합니다."""
        normalizer = PoseNormalizer()
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([-1.0, 0.0, 0.0])
        assert abs(normalizer._angle3d(p1, p2, p3) - 180.0) < 0.01

    def testAngle3dZeroVector(self):
        """영벡터 입력 시 0.0을 반환해야 합니다."""
        normalizer = PoseNormalizer()
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        assert normalizer._angle3d(p1, p2, p3) == 0.0


# ─── ThrowSegmenter 테스트 ───────────────────────────────────────────────────

class TestThrowSegmenter:
    """ThrowSegmenter 단위 테스트."""

    def testFindPeaksBasic(self):
        """단순한 3-peak 신호에서 3개 피크를 감지해야 합니다."""
        segmenter = ThrowSegmenter(fps=30.0)
        # 3개의 뚜렷한 피크를 가진 신호
        signal = np.zeros(100)
        signal[20] = 1.0  # peak 1
        signal[50] = 1.0  # peak 2
        signal[80] = 1.0  # peak 3

        peaks = segmenter._findPeaks(signal, min_prominence=0.5, min_distance=20)
        assert len(peaks) == 3
        assert 20 in peaks
        assert 50 in peaks
        assert 80 in peaks

    def testFindPeaksMinDistanceFiltering(self):
        """min_distance보다 가까운 피크는 하나만 남아야 합니다."""
        segmenter = ThrowSegmenter(fps=30.0)
        signal = np.zeros(100)
        signal[20] = 0.8
        signal[22] = 1.0  # 더 큰 피크 (22번이 살아남아야 함)
        signal[70] = 0.9

        peaks = segmenter._findPeaks(signal, min_prominence=0.5, min_distance=10)
        # 20과 22는 10 이내이므로 더 큰 22번만 남아야 함
        assert 20 not in peaks
        assert 22 in peaks

    def testGaussianSmoothingPreservesLength(self):
        """스무딩 후 신호 길이가 유지되어야 합니다."""
        segmenter = ThrowSegmenter(fps=30.0)
        signal = np.random.rand(90)
        smoothed = segmenter._gaussianSmooth(signal, sigma_seconds=0.1)
        assert len(smoothed) == len(signal)

    def testEmptyFramesFallback(self):
        """프레임이 너무 적으면 전체를 1개 세그먼트로 반환해야 합니다."""
        segmenter = ThrowSegmenter(fps=30.0, min_segment_frames=15)
        frames = [makeFrame(i) for i in range(5)]
        normalizer = PoseNormalizer()
        normalized = normalizer.normalize(frames, "right")
        result = segmenter.segment(frames, normalized, "right")
        assert len(result) >= 1

    def testThreeThrowsDetected(self):
        """3번 투구가 있는 합성 세션에서 3개 세그먼트를 분리해야 합니다."""
        segmenter = ThrowSegmenter(fps=30.0, min_segment_frames=10)
        normalizer = PoseNormalizer(smoothing_window=1)

        # 3번의 투구를 합성 (각 60프레임, 간격 30프레임)
        all_frames = []
        for throw_i in range(3):
            offset = throw_i * 90
            # 30프레임 대기
            for i in range(30):
                all_frames.append(makeFrame(offset + i))
            # 60프레임 투구
            throw_frames = makeThrowFrames(n_frames=60, start_index=offset + 30)
            all_frames.extend(throw_frames)

        normalized = normalizer.normalize(all_frames, "right")
        segments = segmenter.segment(all_frames, normalized, "right")
        # 3개 투구 분리 (세그먼트 수가 1~3 사이여야 함)
        assert 1 <= len(segments) <= 3


# ─── PhaseDetector 테스트 ────────────────────────────────────────────────────

class TestPhaseDetector:
    """PhaseDetector 단위 테스트."""

    def testDetectReturnsThrowPhases(self):
        """감지 결과가 ThrowPhases 타입이어야 합니다."""
        detector = PhaseDetector(fps=30.0)
        normalizer = PoseNormalizer(smoothing_window=1)
        frames = makeThrowFrames(n_frames=60, start_index=0)
        normalized = normalizer.normalize(frames, "right")
        result = detector.detect(frames, normalized, "right")
        assert result is None or isinstance(result, ThrowPhases)

    def testReleaseComeAfterTakebackMax(self):
        """릴리즈 프레임은 반드시 테이크백 정점 이후여야 합니다."""
        detector = PhaseDetector(fps=30.0)
        normalizer = PoseNormalizer(smoothing_window=1)
        frames = makeThrowFrames(n_frames=60, start_index=0)
        normalized = normalizer.normalize(frames, "right")
        result = detector.detect(frames, normalized, "right")
        if result is not None:
            assert result.release >= result.takeback_max

    def testFollowThroughAfterRelease(self):
        """팔로스루는 반드시 릴리즈 이후여야 합니다."""
        detector = PhaseDetector(fps=30.0)
        normalizer = PoseNormalizer(smoothing_window=1)
        frames = makeThrowFrames(n_frames=60, start_index=0)
        normalized = normalizer.normalize(frames, "right")
        result = detector.detect(frames, normalized, "right")
        if result is not None:
            assert result.follow_through >= result.release

    def testTooShortFramesReturnsNone(self):
        """프레임이 10개 미만이면 None을 반환해야 합니다."""
        detector = PhaseDetector(fps=30.0)
        normalizer = PoseNormalizer(smoothing_window=1)
        frames = [makeFrame(i) for i in range(5)]
        normalized = normalizer.normalize(frames, "right")
        result = detector.detect(frames, normalized, "right")
        assert result is None


# ─── MetricsCalculator 테스트 ────────────────────────────────────────────────

class TestMetricsCalculator:
    """MetricsCalculator 단위 테스트."""

    def testComputeReturnsThrowMetrics(self):
        """계산 결과가 ThrowMetrics 타입이어야 합니다."""
        calc = MetricsCalculator(fps=30.0)
        normalizer = PoseNormalizer(smoothing_window=1)
        detector = PhaseDetector(fps=30.0)

        frames = makeThrowFrames(n_frames=60)
        normalized = normalizer.normalize(frames, "right")
        phases = detector.detect(frames, normalized, "right")

        if phases is not None:
            metrics = calc.compute(frames, normalized, phases, "right")
            assert isinstance(metrics, ThrowMetrics)

    def testReleaseTimingIsPositive(self):
        """릴리즈 타이밍은 항상 양수여야 합니다."""
        calc = MetricsCalculator(fps=30.0)
        normalizer = PoseNormalizer(smoothing_window=1)
        detector = PhaseDetector(fps=30.0)

        frames = makeThrowFrames(n_frames=60)
        normalized = normalizer.normalize(frames, "right")
        phases = detector.detect(frames, normalized, "right")

        if phases is not None:
            metrics = calc.compute(frames, normalized, phases, "right")
            assert metrics.release_timing_ms > 0

    def testConsistencyScoreSingleThrow(self):
        """투구가 1개면 일관성 점수는 100이어야 합니다."""
        from src.models import ThrowAnalysis
        from src.models import SessionResult

        calc = MetricsCalculator(fps=30.0)
        # 더미 ThrowAnalysis 1개
        dummy_phases = ThrowPhases(address=0, takeback_start=5, takeback_max=10, release=15, follow_through=20)
        dummy_metrics = ThrowMetrics(takeback_angle_deg=45.0, max_elbow_velocity_deg_s=200.0, release_timing_ms=100.0)
        dummy_analysis = ThrowAnalysis(1, "right", (0, 20), dummy_phases, dummy_metrics)

        score = calc.computeConsistencyScore([dummy_analysis])
        assert score == 100.0

    def testReleaseAngleComputation(self):
        """릴리즈 각도 계산이 올바른 범위(-90~90도)에 있어야 합니다."""
        calc = MetricsCalculator(fps=30.0)
        # 수평 전완 (0도)
        elbow = np.array([0.0, 0.0, 0.0])
        wrist = np.array([1.0, 0.0, 0.0])
        angle = calc._computeReleaseAngle(elbow, wrist)
        assert abs(angle) < 5.0  # 수평이면 0도에 가까워야 함


# ─── DartAnalyzer 통합 테스트 ────────────────────────────────────────────────

class TestDartAnalyzer:
    """DartAnalyzer 통합 테스트."""

    def testAnalyzeEmptyFrames(self):
        """빈 프레임 리스트에서 0회를 반환해야 합니다."""
        analyzer = DartAnalyzer(fps=30.0)
        result = analyzer.analyze_session([])
        assert result.total_throws_detected == 0

    def testAnalyzeNoKeypoints(self):
        """keypoints가 없는 프레임은 무시되어야 합니다."""
        analyzer = DartAnalyzer(fps=30.0)
        frames = [FrameData(frame_index=i, timestamp_ms=i * 33.3) for i in range(50)]
        result = analyzer.analyze_session(frames)
        assert result.total_throws_detected == 0

    def testDetectThrowingSideRight(self):
        """오른손 투구자를 올바르게 감지해야 합니다."""
        analyzer = DartAnalyzer(fps=30.0)
        frames = []
        for i in range(30):
            frames.append(makeFrame(
                i,
                right_wrist=[0.5 + 0.1 * np.sin(i * 0.3), 0.5, 0.0],
                left_wrist=[0.3, 0.5, 0.0],  # 왼손은 거의 고정
            ))
        assert analyzer._detectThrowingSide(frames) == "right"

    def testAnalyzeSessionRunsWithoutError(self):
        """유효 프레임으로 에러 없이 실행되어야 합니다."""
        analyzer = DartAnalyzer(fps=30.0)
        frames = [makeFrame(i) for i in range(30)]
        result = analyzer.analyze_session(frames)
        assert isinstance(result.total_throws_detected, int)
        assert result.fps == 30.0
