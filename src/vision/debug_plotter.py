"""디버그 시각화 모듈.

분석 파이프라인의 중간 신호를 matplotlib 그래프로 시각화합니다.
투구 감지 및 Phase 경계를 직관적으로 확인하여 디버깅과 파라미터 튜닝에 활용합니다.

생성되는 그래프:
  Panel 1: Wrist Velocity + Segment Boundaries (ThrowSegmenter 진단)
  Panel 2: Elbow Angle + Phase Boundaries (PhaseDetector 진단)
  Panel 3: Elbow Angular Velocity + Release Point (MetricsCalculator 진단)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 디스플레이 없이 파일로 저장
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.models import FrameData, ThrowAnalysis, SessionResult


class DebugPlotter:
    """분석 파이프라인의 중간 신호를 시각화하는 디버그 도구.

    사용 예시:
        plotter = DebugPlotter(output_dir="output/debug")
        plotter.plotSessionOverview(
            frames, wrist_velocity, segments, session_result, side
        )
    """

    # 시각화 색상 팔레트
    _COLORS = {
        "wrist_velocity": "#4FC3F7",       # 밝은 파랑 — 손목 속도 시계열
        "peaks": "#FF5252",                # 빨강 — 피크 마커
        "segment": "#66BB6A",              # 녹색 — 세그먼트 배경
        "elbow_angle": "#FFB74D",          # 주황 — 팔꿈치 각도
        "elbow_velocity": "#CE93D8",       # 보라 — 팔꿈치 각속도
        "address": "#42A5F5",              # 파랑 — Address Phase
        "takeback": "#FFA726",             # 주황 — Takeback Phase
        "release": "#EF5350",              # 빨강 — Release Phase
        "follow_through": "#66BB6A",       # 녹색 — Follow-through Phase
    }

    def __init__(self, output_dir: str = "output/debug"):
        """초기화.

        Args:
            output_dir: 그래프 이미지 저장 디렉토리.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plotSessionOverview(
        self,
        frames: list[FrameData],
        wrist_velocity: np.ndarray,
        peak_indices: list[int],
        segments: list[list[FrameData]],
        session: SessionResult,
        side: str,
        sample_name: str = "sample",
    ) -> str:
        """전체 세션의 디버그 오버뷰 그래프를 생성합니다.

        3개의 패널로 구성:
        - Panel 1: 손목 속도 + 피크 + 세그먼트 구간
        - Panel 2: 팔꿈치 각도 시계열 + Phase 경계선
        - Panel 3: 팔꿈치 각속도 + 릴리즈 포인트

        Args:
            frames: 전체 세션의 유효 FrameData 리스트.
            wrist_velocity: 손목 속도 시계열 (len = len(frames)).
            peak_indices: ThrowSegmenter가 감지한 피크 인덱스 리스트.
            segments: 분리된 세그먼트 리스트.
            session: 최종 분석 결과.
            side: 투구 팔 방향 ('left' 또는 'right').
            sample_name: 저장 파일명에 사용할 샘플 이름.

        Returns:
            저장된 이미지 파일 경로.
        """
        n_frames = len(frames)
        fps = session.fps
        time_axis = np.arange(n_frames) / fps  # 시간(초) 축

        # 전체 세션에서 원시 좌표 기반 팔꿈치 각도/각속도 계산
        raw_angles, raw_velocity = self._computeRawAngleSignals(frames, side, fps)

        # ─── 그래프 생성 ───────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(
            f"🎯 Debug: {sample_name}  |  {session.total_throws_detected} throws  |  {side} arm",
            fontsize=14,
            fontweight="bold",
        )

        # ── Panel 1: 손목 속도 + 피크 + 세그먼트 ──────────────────────────
        ax1 = axes[0]
        ax1.set_title("Panel 1: Wrist Velocity + Segment Boundaries", fontsize=11, pad=8)
        ax1.plot(time_axis, wrist_velocity, color=self._COLORS["wrist_velocity"],
                 linewidth=0.8, label="Wrist Velocity")

        # 세그먼트 배경 표시
        for i, seg in enumerate(segments):
            seg_start = seg[0].frame_index / fps
            seg_end = seg[-1].frame_index / fps
            ax1.axvspan(seg_start, seg_end, alpha=0.15,
                        color=self._COLORS["segment"], label=f"Seg {i+1}" if i == 0 else None)

        # 피크 마커 표시
        if len(peak_indices) > 0 and len(peak_indices) <= len(time_axis):
            valid_peaks = [p for p in peak_indices if p < n_frames]
            if valid_peaks:
                ax1.scatter(
                    time_axis[valid_peaks], wrist_velocity[valid_peaks],
                    color=self._COLORS["peaks"], s=60, zorder=5,
                    marker="v", label="Peaks",
                )

        ax1.set_ylabel("Speed (normalized/frame)")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: 팔꿈치 각도 + Phase 경계선 ───────────────────────────
        ax2 = axes[1]
        ax2.set_title("Panel 2: Elbow Angle + Phase Boundaries", fontsize=11, pad=8)
        ax2.plot(time_axis, raw_angles, color=self._COLORS["elbow_angle"],
                 linewidth=0.8, label="Elbow Angle (°)")

        # 각 투구의 Phase 경계선 표시
        for throw in session.throws:
            self._drawPhaseLines(ax2, throw, fps)

        ax2.set_ylabel("Angle (°)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: 팔꿈치 각속도 + 릴리즈 포인트 ────────────────────────
        ax3 = axes[2]
        ax3.set_title("Panel 3: Elbow Angular Velocity + Release Point", fontsize=11, pad=8)
        ax3.plot(time_axis, raw_velocity, color=self._COLORS["elbow_velocity"],
                 linewidth=0.8, label="Elbow Angular Velocity (°/s)")
        ax3.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

        # 릴리즈 포인트 마커
        for throw in session.throws:
            rel_frame = throw.phases.release
            if rel_frame < n_frames:
                ax3.scatter(
                    rel_frame / fps, raw_velocity[rel_frame],
                    color=self._COLORS["release"], s=80, zorder=5,
                    marker="*", label=f"Release T{throw.throw_index}",
                )

        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("Angular Velocity (°/s)")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ─── 저장 ──────────────────────────────────────────────────────────
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{sample_name}_debug.png")
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"  📊 Debug plot saved: {output_path}")
        return output_path

    # ─── Private Helpers ──────────────────────────────────────────────────

    def _drawPhaseLines(
        self,
        ax: plt.Axes,
        throw: ThrowAnalysis,
        fps: float,
    ) -> None:
        """하나의 투구에 대해 Phase 경계 수직선을 그립니다.

        Args:
            ax: matplotlib Axes 객체.
            throw: ThrowAnalysis 객체.
            fps: 프레임레이트.
        """
        phases = throw.phases
        idx = throw.throw_index
        phase_map = {
            "Addr": (phases.address, self._COLORS["address"]),
            "TB":   (phases.takeback_max, self._COLORS["takeback"]),
            "Rel":  (phases.release, self._COLORS["release"]),
            "FT":   (phases.follow_through, self._COLORS["follow_through"]),
        }
        for label, (frame, color) in phase_map.items():
            t = frame / fps
            ax.axvline(x=t, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
            ax.text(t, ax.get_ylim()[1] * 0.95, f"T{idx}:{label}",
                    fontsize=7, color=color, ha="center", va="top",
                    rotation=90, alpha=0.8)

    @staticmethod
    def _computeRawAngleSignals(
        frames: list[FrameData],
        side: str,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """전체 프레임에 대해 원시 좌표 기반 팔꿈치 각도/각속도를 계산합니다.

        Args:
            frames: FrameData 리스트.
            side: 투구 팔 방향.
            fps: 프레임레이트.

        Returns:
            (angles, velocity) 튜플. 각각 shape (N,).
        """
        n = len(frames)
        dt = 1.0 / fps

        # 원시 좌표 추출 (forward fill)
        shoulder = np.zeros((n, 3))
        elbow = np.zeros((n, 3))
        wrist = np.zeros((n, 3))
        last_s = np.zeros(3)
        last_e = np.zeros(3)
        last_w = np.zeros(3)

        for i, f in enumerate(frames):
            if f.keypoints:
                s = f.keypoints.get(f"{side}_shoulder")
                e = f.keypoints.get(f"{side}_elbow")
                w = f.keypoints.get(f"{side}_wrist")
                if s is not None: last_s = np.array(s, dtype=np.float64)
                if e is not None: last_e = np.array(e, dtype=np.float64)
                if w is not None: last_w = np.array(w, dtype=np.float64)
            shoulder[i] = last_s
            elbow[i] = last_e
            wrist[i] = last_w

        # 각도 계산
        angles = np.zeros(n)
        for i in range(n):
            v1 = shoulder[i] - elbow[i]
            v2 = wrist[i] - elbow[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles[i] = np.degrees(np.arccos(cos_val))

        # 각속도 (1차 미분)
        velocity = np.gradient(angles, dt)

        return angles, velocity