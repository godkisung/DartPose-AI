"""데이터 모델 정의.

모듈 간 데이터 전달을 위한 명시적 데이터 구조를 정의합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Keypoints:
    """단일 프레임에서 추출된 관절 및 손가락 좌표 (정규화된 0~1 좌표)."""

    left_shoulder: list[float]
    right_shoulder: list[float]
    left_elbow: list[float]
    right_elbow: list[float]
    left_wrist: list[float]
    right_wrist: list[float]
    left_hip: list[float]
    right_hip: list[float]
    # 손가락 끝점 (MediaPipe Hands 랜드마크 4, 8, 12번 추천)
    left_thumb_tip: Optional[list[float]] = None
    right_thumb_tip: Optional[list[float]] = None
    left_index_tip: Optional[list[float]] = None
    right_index_tip: Optional[list[float]] = None
    left_middle_tip: Optional[list[float]] = None
    right_middle_tip: Optional[list[float]] = None

    def get(self, joint_name: str) -> list[float] | None:
        """관절 이름으로 좌표를 반환합니다."""
        return getattr(self, joint_name, None)

    def to_dict(self) -> dict[str, list[float]]:
        """딕셔너리로 변환합니다."""
        d = {
            'left_shoulder': self.left_shoulder,
            'right_shoulder': self.right_shoulder,
            'left_elbow': self.left_elbow,
            'right_elbow': self.right_elbow,
            'left_wrist': self.left_wrist,
            'right_wrist': self.right_wrist,
            'left_hip': self.left_hip,
            'right_hip': self.right_hip,
        }
        if self.left_thumb_tip: d['left_thumb_tip'] = self.left_thumb_tip
        if self.right_thumb_tip: d['right_thumb_tip'] = self.right_thumb_tip
        if self.left_index_tip: d['left_index_tip'] = self.left_index_tip
        if self.right_index_tip: d['right_index_tip'] = self.right_index_tip
        if self.left_middle_tip: d['left_middle_tip'] = self.left_middle_tip
        if self.right_middle_tip: d['right_middle_tip'] = self.right_middle_tip
        return d


@dataclass
class FrameData:
    """단일 프레임 데이터."""

    frame_index: int
    timestamp_ms: float
    keypoints: Keypoints | None = None


@dataclass
class ThrowPhases:
    """투구 단계별 프레임 인덱스 (세션 내 절대 인덱스)."""

    address: int           # 준비 자세 시작 (다트 조준 시작)
    takeback_start: int    # 테이크백 시작 (팔을 뒤로 당기기 시작)
    takeback_max: int      # 테이크백 정점 (가장 뒤로 당겨진 시점)
    release: int           # 릴리즈 (다트를 놓는 찰나)
    follow_through: int    # 팔로스루 완료 (팔이 완전히 펴진 시점)

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
    """고도화된 단일 투구 생체역학 수치 지표."""

    # 1. 안정성 지표 (Stability)
    elbow_drift_norm: float = 0.0          # 투구 중 팔꿈치의 이동 거리 (정규화 단위)
    shoulder_stability: float = 0.0        # 어깨 고정성 (분산)
    body_sway: float = 0.0                 # 몸통 흔들림 (X축 변위)

    # 2. 각도 지표 (Angles)
    takeback_angle_deg: float = 0.0        # 테이크백 정점에서의 팔꿈치 각도
    release_angle_deg: float = 0.0         # 릴리즈 순간의 지면 대비 팔뚝 각도
    follow_through_angle_deg: float = 0.0  # 팔로스루 완료 시 팔 각도

    # 3. 속도/타이밍 지표 (Velocity & Timing)
    max_elbow_velocity_deg_s: float = 0.0  # 최대 팔꿈치 확장 속도
    release_timing_ms: float = 0.0         # 가속 시작부터 릴리즈까지 걸린 시간
    finger_release_speed: float = 0.0      # 릴리즈 순간 손가락이 벌어지는 속도

    # 4. 일관성 점수 (Consistency - 세션 분석 시 계산)
    consistency_score: float = 0.0         # 이전 투구들과의 유사도 (0~100)

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float))}


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
