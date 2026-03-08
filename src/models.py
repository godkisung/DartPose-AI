"""데이터 모델 정의.

모듈 간 데이터 전달을 위한 명시적 데이터 구조를 정의합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Keypoints:
    """단일 프레임에서 추출된 관절 좌표 (정규화된 0~1 좌표)."""

    left_shoulder: list[float]
    right_shoulder: list[float]
    left_elbow: list[float]
    right_elbow: list[float]
    left_wrist: list[float]
    right_wrist: list[float]
    left_hip: list[float]
    right_hip: list[float]
    left_index: list[float]
    right_index: list[float]
    left_pinky: list[float]
    right_pinky: list[float]

    def get(self, joint_name: str) -> list[float] | None:
        """관절 이름으로 좌표를 반환합니다."""
        return getattr(self, joint_name, None)

    def to_dict(self) -> dict[str, list[float]]:
        """딕셔너리로 변환합니다."""
        return {
            'left_shoulder': self.left_shoulder,
            'right_shoulder': self.right_shoulder,
            'left_elbow': self.left_elbow,
            'right_elbow': self.right_elbow,
            'left_wrist': self.left_wrist,
            'right_wrist': self.right_wrist,
            'left_hip': self.left_hip,
            'right_hip': self.right_hip,
            'left_index': self.left_index,
            'right_index': self.right_index,
            'left_pinky': self.left_pinky,
            'right_pinky': self.right_pinky,
        }


@dataclass
class FrameData:
    """단일 프레임 데이터."""

    frame_index: int
    timestamp_ms: float
    keypoints: Keypoints | None = None


@dataclass
class ThrowPhases:
    """투구 단계별 프레임 인덱스 (세션 내 절대 인덱스)."""

    address: int           # 준비 자세 시작
    takeback_start: int    # 테이크백 시작 (팔 뒤로)
    takeback_max: int      # 테이크백 최대 (가장 뒤)
    release: int           # 릴리즈 (던지는 순간)
    follow_through: int    # 팔로스루

    def to_dict(self) -> dict[str, int]:
        return {
            'address': self.address,
            'takeback_start': self.takeback_start,
            'takeback_max': self.takeback_max,
            'release': self.release,
            'follow_through': self.follow_through,
        }


@dataclass
class ThrowMetrics:
    """단일 투구의 생체역학 수치 지표."""

    elbow_stability_variance: float = 0.0
    takeback_min_angle_deg: float = 0.0
    elbow_extension_velocity_deg_s: float = 0.0
    wrist_snap_velocity_deg_s: float = 0.0
    body_sway_x_norm: float = 0.0
    # Phase 2 확장 지표
    shoulder_stability_variance: float = 0.0
    release_height_consistency: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            'elbow_stability_variance': self.elbow_stability_variance,
            'takeback_min_angle_deg': self.takeback_min_angle_deg,
            'elbow_extension_velocity_deg_s': self.elbow_extension_velocity_deg_s,
            'wrist_snap_velocity_deg_s': self.wrist_snap_velocity_deg_s,
            'body_sway_x_norm': self.body_sway_x_norm,
            'shoulder_stability_variance': self.shoulder_stability_variance,
            'release_height_consistency': self.release_height_consistency,
        }


@dataclass
class ThrowAnalysis:
    """단일 투구 분석 결과."""

    throw_index: int
    throwing_arm: str  # 'left' or 'right'
    frame_range: tuple[int, int]  # (start_frame, end_frame) 절대 인덱스
    phases: ThrowPhases
    metrics: ThrowMetrics
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'throw_index': self.throw_index,
            'throwing_arm': self.throwing_arm,
            'frame_range': list(self.frame_range),
            'phases': self.phases.to_dict(),
            'metrics': self.metrics.to_dict(),
            'issues': self.issues,
        }


@dataclass
class SessionResult:
    """전체 세션 (여러 투구) 분석 결과."""

    total_frames: int
    fps: float
    total_throws_detected: int
    throws: list[ThrowAnalysis] = field(default_factory=list)
    llm_feedback: str = ""

    def to_dict(self) -> dict:
        return {
            'total_frames': self.total_frames,
            'fps': self.fps,
            'total_throws_detected': self.total_throws_detected,
            'throws': [t.to_dict() for t in self.throws],
            'llm_feedback': self.llm_feedback,
        }
