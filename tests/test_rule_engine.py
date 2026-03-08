"""Rule Engine 핵심 로직 단위 테스트."""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Keypoints, FrameData, ThrowPhases, ThrowMetrics
from src.vision.rule_engine import PoseRuleEngine


def _make_keypoints(**overrides) -> Keypoints:
    """테스트용 Keypoints를 생성합니다."""
    defaults = {
        'left_shoulder': [0.3, 0.3, 0.0],
        'right_shoulder': [0.5, 0.3, 0.0],
        'left_elbow': [0.25, 0.4, 0.0],
        'right_elbow': [0.55, 0.4, 0.0],
        'left_wrist': [0.2, 0.5, 0.0],
        'right_wrist': [0.6, 0.5, 0.0],
        'left_hip': [0.35, 0.6, 0.0],
        'right_hip': [0.45, 0.6, 0.0],
        'left_index': [0.18, 0.52, 0.0],
        'right_index': [0.62, 0.52, 0.0],
        'left_pinky': [0.22, 0.52, 0.0],
        'right_pinky': [0.58, 0.52, 0.0],
    }
    defaults.update(overrides)
    return Keypoints(**defaults)


def _make_frame(index: int, fps: float = 30.0, **kp_overrides) -> FrameData:
    """테스트용 FrameData를 생성합니다."""
    return FrameData(
        frame_index=index,
        timestamp_ms=index * (1000.0 / fps),
        keypoints=_make_keypoints(**kp_overrides),
    )


class TestAngle3D:
    """3D 각도 계산 테스트."""

    def test_right_angle(self):
        engine = PoseRuleEngine()
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([0, 1, 0])
        angle = engine._angle_3d(p1, p2, p3)
        assert abs(angle - 90.0) < 0.01

    def test_straight_angle(self):
        engine = PoseRuleEngine()
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([-1, 0, 0])
        angle = engine._angle_3d(p1, p2, p3)
        assert abs(angle - 180.0) < 0.01

    def test_none_input(self):
        engine = PoseRuleEngine()
        assert engine._angle_3d(None, np.array([0, 0, 0]), np.array([1, 0, 0])) == 0.0

    def test_zero_vector(self):
        engine = PoseRuleEngine()
        p1 = np.array([0, 0, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([1, 0, 0])
        assert engine._angle_3d(p1, p2, p3) == 0.0


class TestThrowingArmDetection:
    """투구 팔 감지 테스트."""

    def test_right_hand_more_movement(self):
        engine = PoseRuleEngine()
        frames = []
        for i in range(30):
            # 오른쪽 손목은 크게 움직이고, 왼쪽은 거의 고정
            frames.append(_make_frame(
                i,
                right_wrist=[0.5 + 0.1 * np.sin(i * 0.3), 0.5, 0.0],
                left_wrist=[0.2, 0.5, 0.0],
            ))
        assert engine._detect_throwing_arm(frames) == "right"

    def test_left_hand_more_movement(self):
        engine = PoseRuleEngine()
        frames = []
        for i in range(30):
            frames.append(_make_frame(
                i,
                right_wrist=[0.6, 0.5, 0.0],
                left_wrist=[0.2 + 0.15 * np.sin(i * 0.3), 0.5, 0.0],
            ))
        assert engine._detect_throwing_arm(frames) == "left"


class TestSessionAnalysis:
    """세션 분석 통합 테스트."""

    def test_empty_frames(self):
        engine = PoseRuleEngine(fps=30)
        result = engine.analyze_session([])
        assert result.total_throws_detected == 0

    def test_no_keypoints(self):
        engine = PoseRuleEngine(fps=30)
        frames = [FrameData(frame_index=i, timestamp_ms=i * 33.3) for i in range(50)]
        result = engine.analyze_session(frames)
        assert result.total_throws_detected == 0

    def test_analyze_with_valid_frames(self):
        """최소한의 유효 프레임으로 분석을 실행합니다."""
        engine = PoseRuleEngine(fps=30)
        frames = [_make_frame(i) for i in range(30)]
        result = engine.analyze_session(frames)
        assert result.total_frames == 30
        assert result.fps == 30.0
        # 정적 포즈이므로 투구가 감지될 수도 안 될 수도 있음
        # 핵심은 에러 없이 실행되는 것
        assert isinstance(result.total_throws_detected, int)


class TestDataModels:
    """데이터 모델 테스트."""

    def test_keypoints_get(self):
        kp = _make_keypoints()
        assert kp.get("left_shoulder") == [0.3, 0.3, 0.0]
        assert kp.get("nonexistent") is None

    def test_keypoints_to_dict(self):
        kp = _make_keypoints()
        d = kp.to_dict()
        assert "left_shoulder" in d
        assert "right_wrist" in d
        assert len(d) == 12

    def test_throw_metrics_to_dict(self):
        m = ThrowMetrics(elbow_stability_variance=0.001, takeback_min_angle_deg=45.0)
        d = m.to_dict()
        assert d["elbow_stability_variance"] == 0.001
        assert d["takeback_min_angle_deg"] == 45.0

    def test_throw_phases_to_dict(self):
        p = ThrowPhases(address=0, takeback_start=5, takeback_max=10, release=15, follow_through=20)
        d = p.to_dict()
        assert d["release"] == 15
