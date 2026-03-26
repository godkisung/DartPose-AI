"""투구별 궤적 렌더링 모듈.

스켈레톤 영상 위에 투구별 손목 궤적을 색상으로 오버레이합니다.
다른 앱에서 영감을 받아, 각 투구를 서로 다른 색상의 궤적으로 시각화합니다.

차별점:
- 투구별 색상 궤적 (빨강→파랑→노랑 순환)
- 실시간 투구 카운터 (화면 상단)
- 현재 팔꿈치 각도 게이지 (화면 하단)
- FSM 상태 인디케이터

사용 예시:
    renderer = TrajectoryRenderer()
    renderer.render_overlay(
        video_path, output_path, frames, segments, elbow_angles, throwing_side
    )
"""

import cv2
import numpy as np
from src.models import FrameData


class TrajectoryRenderer:
    """투구별 궤적을 영상에 오버레이하는 렌더러.

    2-pass 아키텍처:
    1st pass: 분석 엔진이 투구 세그먼트를 결정
    2nd pass: 이 렌더러가 세그먼트 정보를 기반으로 궤적 오버레이 영상 생성
    """

    # 투구별 궤적 색상 (BGR 형식) — 최대 6개 순환
    _THROW_COLORS = [
        (0, 0, 255),     # 🔴 빨강 (투구 1)
        (255, 100, 0),   # 🔵 파랑 (투구 2)
        (0, 230, 255),   # 🟡 노랑 (투구 3)
        (0, 255, 100),   # 🟢 초록 (투구 4)
        (255, 0, 200),   # 🟣 핑크 (투구 5)
        (255, 200, 0),   # 🔵 시안 (투구 6)
    ]

    # UI 요소 색상
    _UI_BG_COLOR = (0, 0, 0)       # 배경
    _UI_TEXT_COLOR = (255, 255, 255)  # 기본 텍스트
    _UI_ACCENT_COLOR = (0, 200, 255)  # 강조

    def __init__(self, trail_length: int = 30, trail_thickness: int = 3):
        """초기화.

        Args:
            trail_length: 궤적 잔상 길이 (프레임 수).
            trail_thickness: 궤적 선 두께 (픽셀).
        """
        self.trail_length = trail_length
        self.trail_thickness = trail_thickness

    def renderOverlay(
        self,
        input_video_path: str,
        output_video_path: str,
        frames: list[FrameData],
        segments: list[list[FrameData]],
        elbow_angles: np.ndarray,
        throwing_side: str,
        fps: float = 30.0,
    ) -> str:
        """투구별 궤적 오버레이가 포함된 영상을 생성합니다.

        기존 스켈레톤 영상 위에 추가로:
        1. 투구별 색상 궤적 (손목 경로)
        2. 투구 카운터 UI (좌상단)
        3. 팔꿈치 각도 게이지 (우하단)
        4. FSM 상태 인디케이터

        Args:
            input_video_path: 입력 영상 경로 (스켈레톤 오버레이 포함).
            output_video_path: 출력 영상 경로.
            frames: 분석에 사용된 FrameData 리스트.
            segments: ThrowSegmenter가 감지한 세그먼트 리스트.
            elbow_angles: 프레임별 팔꿈치 각도 (도 단위).
            throwing_side: 투구 팔 방향.
            fps: 프레임레이트.

        Returns:
            출력 영상 파일 경로.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"  ⚠ 영상 열기 실패: {input_video_path}")
            return input_video_path

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 출력 영상 코덱 설정 (H.264)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # ─ 세그먼트 → 프레임 인덱스 매핑 구축 ────────────────────────────
        frame_to_throw = self._buildFrameToThrowMap(frames, segments)

        # ─ 프레임별 손목 좌표를 픽셀 좌표로 변환 ──────────────────────────
        wrist_key = f"{throwing_side}_wrist"
        wrist_pixel_coords = self._extractWristPixelCoords(
            frames, wrist_key, width, height
        )

        total_throws = len(segments)

        # ─ 프레임 순회하며 오버레이 렌더링 ────────────────────────────────
        frame_idx = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break

            # keypoints가 있는 유효 프레임 인덱스와 매칭
            if frame_idx < len(frames):
                # 1. 궤적 오버레이
                self._drawTrajectory(
                    img, frame_idx, wrist_pixel_coords, frame_to_throw
                )

                # 2. 투구 카운터 UI
                current_throw = frame_to_throw.get(frame_idx, 0)
                self._drawThrowCounter(img, current_throw, total_throws)

                # 3. 팔꿈치 각도 게이지
                if frame_idx < len(elbow_angles):
                    angle = float(elbow_angles[frame_idx])
                    self._drawAngleGauge(img, angle, width, height)

            out.write(img)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"  🎨 궤적 오버레이 영상 저장: {output_video_path}")
        return output_video_path

    # ─── Drawing Methods ─────────────────────────────────────────────────────

    def _drawTrajectory(
        self,
        img: np.ndarray,
        current_frame: int,
        wrist_coords: dict[int, tuple[int, int]],
        frame_to_throw: dict[int, int],
    ) -> None:
        """현재 프레임까지의 손목 궤적을 그립니다.

        각 투구별로 다른 색상의 라인을 그려 시각적으로 구분합니다.
        잔상 효과: 최근 trail_length 프레임만 표시하며, 오래된 부분은 투명하게.

        Args:
            img: 프레임 이미지 (in-place 수정).
            current_frame: 현재 프레임 인덱스.
            wrist_coords: 프레임→손목 픽셀 좌표 매핑.
            frame_to_throw: 프레임→투구 번호 매핑.
        """
        # 현재 프레임이 속한 투구의 궤적만 그리기and 이전 투구의 완성된 궤적도 표시
        shown_throws: set[int] = set()

        # 모든 기록된 투구의 궤적 그리기
        for throw_idx in set(frame_to_throw.values()):
            if throw_idx == 0:
                continue

            color = self._THROW_COLORS[(throw_idx - 1) % len(self._THROW_COLORS)]

            # 이 투구에 속하는 연속 프레임들의 좌표 수집
            throw_frames = sorted([
                f for f, t in frame_to_throw.items()
                if t == throw_idx and f <= current_frame and f in wrist_coords
            ])

            if len(throw_frames) < 2:
                continue

            # 궤적 라인 그리기 (잔상 효과 포함)
            for i in range(1, len(throw_frames)):
                f_prev = throw_frames[i - 1]
                f_curr = throw_frames[i]

                if f_prev not in wrist_coords or f_curr not in wrist_coords:
                    continue

                pt1 = wrist_coords[f_prev]
                pt2 = wrist_coords[f_curr]

                # 잔상 효과: 최근 trail_length 프레임 내의 점은 선명하게
                age = current_frame - f_curr
                if age > self.trail_length * 3:
                    # 너무 오래된 궤적은 얇게 표시
                    alpha_thickness = 1
                else:
                    # 최근 궤적은 두껍게
                    alpha_thickness = max(1, self.trail_thickness - age // self.trail_length)

                cv2.line(img, pt1, pt2, color, alpha_thickness, cv2.LINE_AA)

    def _drawThrowCounter(
        self,
        img: np.ndarray,
        current_throw: int,
        total_throws: int,
    ) -> None:
        """투구 카운터를 화면 좌상단에 표시합니다.

        Format: "🎯 Throw: 2 / 3"

        Args:
            img: 프레임 이미지 (in-place 수정).
            current_throw: 현재 투구 번호 (0이면 투구 구간 아님).
            total_throws: 전체 감지된 투구 수.
        """
        h, w = img.shape[:2]

        # 반투명 배경 박스
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (250, 70), self._UI_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # 텍스트
        if current_throw > 0:
            text = f"Throw: {current_throw} / {total_throws}"
            color = self._THROW_COLORS[(current_throw - 1) % len(self._THROW_COLORS)]
        else:
            text = f"Throws: {total_throws}"
            color = self._UI_TEXT_COLOR

        cv2.putText(img, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    def _drawAngleGauge(
        self,
        img: np.ndarray,
        angle: float,
        width: int,
        height: int,
    ) -> None:
        """팔꿈치 각도 게이지를 화면 우하단에 표시합니다.

        수직 바 형태로 현재 팔꿈치 각도를 시각화합니다.
        - 180° (완전 펴짐): 초록
        - 90° (중간): 노랑
        - 30° (완전 접힘): 빨강

        Args:
            img: 프레임 이미지.
            angle: 현재 팔꿈치 각도 (도).
            width: 영상 너비.
            height: 영상 높이.
        """
        # 게이지 위치 (우하단)
        gauge_x = width - 60
        gauge_y_top = height - 200
        gauge_y_bot = height - 30
        gauge_w = 30
        gauge_h = gauge_y_bot - gauge_y_top

        # 반투명 배경
        overlay = img.copy()
        cv2.rectangle(overlay, (gauge_x - 5, gauge_y_top - 30),
                     (gauge_x + gauge_w + 5, gauge_y_bot + 5),
                     self._UI_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # 배경 바
        cv2.rectangle(img, (gauge_x, gauge_y_top),
                     (gauge_x + gauge_w, gauge_y_bot),
                     (50, 50, 50), -1)

        # 채우기 비율 (0°~180° → 0~1)
        fill_ratio = np.clip(angle / 180.0, 0.0, 1.0)
        fill_h = int(gauge_h * fill_ratio)

        # 색상 그라데이션 (각도에 따라)
        if angle > 140:
            gauge_color = (0, 200, 0)     # 초록 (팔 펴짐)
        elif angle > 90:
            gauge_color = (0, 230, 255)   # 노랑 (중간)
        else:
            gauge_color = (0, 0, 255)     # 빨강 (팔 접힘)

        cv2.rectangle(img, (gauge_x, gauge_y_bot - fill_h),
                     (gauge_x + gauge_w, gauge_y_bot),
                     gauge_color, -1)

        # 각도 숫자 표시
        cv2.putText(img, f"{angle:.0f}", (gauge_x - 5, gauge_y_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._UI_TEXT_COLOR, 1, cv2.LINE_AA)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _buildFrameToThrowMap(
        frames: list[FrameData],
        segments: list[list[FrameData]],
    ) -> dict[int, int]:
        """각 프레임이 몇 번째 투구에 속하는지 매핑합니다.

        Args:
            frames: 전체 FrameData 리스트.
            segments: 투구 세그먼트 리스트.

        Returns:
            프레임 인덱스 → 투구 번호 (1-indexed, 0이면 투구 구간 외).
        """
        frame_map: dict[int, int] = {}
        for throw_idx, segment in enumerate(segments, start=1):
            for f in segment:
                frame_map[f.frame_index] = throw_idx
        return frame_map

    @staticmethod
    def _extractWristPixelCoords(
        frames: list[FrameData],
        wrist_key: str,
        width: int,
        height: int,
    ) -> dict[int, tuple[int, int]]:
        """프레임 인덱스별 손목 픽셀 좌표를 추출합니다.

        MediaPipe 정규화 좌표(0~1)를 실제 픽셀로 변환합니다.

        Args:
            frames: FrameData 리스트.
            wrist_key: 손목 관절 키 이름.
            width: 영상 너비.
            height: 영상 높이.

        Returns:
            프레임 인덱스 → (x, y) 픽셀 좌표 딕셔너리.
        """
        coords: dict[int, tuple[int, int]] = {}
        for f in frames:
            if f.keypoints:
                w = f.keypoints.get(wrist_key)
                if w is not None:
                    px = int(w[0] * width)
                    py = int(w[1] * height)
                    coords[f.frame_index] = (px, py)
        return coords
