"""MediaPipe 기반 관절 좌표 추출 모듈.

비디오 파일 또는 단일 프레임에서 인체 관절 좌표를 추출합니다.
"""

import cv2
import mediapipe as mp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models import Keypoints, FrameData
from src.config import (
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONF,
    MEDIAPIPE_MIN_TRACKING_CONF,
)
from src.utils.video_utils import ensure_compatible_video


# MediaPipe landmark 인덱스 → Keypoints 필드 매핑
_LANDMARK_MAP = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_index': 19,
    'right_index': 20,
    'left_pinky': 17,
    'right_pinky': 18,
}


class PoseExtractor:
    """MediaPipe Pose를 이용한 관절 추출기."""

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

    def extract_from_video(
        self,
        input_path: str,
        output_path: str | None = None,
    ) -> tuple[list[FrameData], float]:
        """비디오 파일에서 모든 프레임의 관절 좌표를 추출합니다.

        Args:
            input_path: 입력 비디오 경로.
            output_path: 스켈레톤 오버레이 비디오 저장 경로 (None이면 저장 안 함).

        Returns:
            (frames, fps) — 추출된 FrameData 리스트와 비디오 FPS.
        """
        compatible_path = ensure_compatible_video(input_path)

        cap = cv2.VideoCapture(compatible_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오를 열 수 없습니다: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"🎯 관절 추출 시작: {input_path} ({total_frames} 프레임, {fps:.1f} FPS)")

        frames: list[FrameData] = []
        detected_count = 0

        with self._mp_pose.Pose(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        ) as pose:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                timestamp_ms = frame_index * (1000.0 / fps)

                # MediaPipe 추론
                frame.flags.writeable = False
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                keypoints = None
                if results.pose_landmarks:
                    keypoints = self._landmarks_to_keypoints(results.pose_landmarks.landmark)
                    detected_count += 1

                    # 스켈레톤 오버레이
                    if out is not None:
                        frame.flags.writeable = True
                        image_bgr = cv2.cvtColor(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            cv2.COLOR_RGB2BGR,
                        )
                        self._mp_drawing.draw_landmarks(
                            image_bgr,
                            results.pose_landmarks,
                            self._mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
                        )
                        frame = image_bgr

                frames.append(FrameData(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    keypoints=keypoints,
                ))

                if out is not None:
                    out.write(frame)

                if frame_index % 50 == 0:
                    print(f"  진행: {frame_index}/{total_frames} ({detected_count} 감지)")

        cap.release()
        if out is not None:
            out.release()
            print(f"✓ 스켈레톤 오버레이 저장: {output_path}")

        detection_rate = (detected_count / total_frames * 100) if total_frames > 0 else 0
        print(f"✓ 추출 완료: {detected_count}/{total_frames} 프레임 감지 ({detection_rate:.1f}%)")

        return frames, fps

    def extract_from_frame(self, frame, pose_ctx) -> Keypoints | None:
        """단일 프레임에서 관절 좌표를 추출합니다 (라이브 모드용).

        Args:
            frame: BGR numpy 배열.
            pose_ctx: MediaPipe Pose 컨텍스트 (with mp_pose.Pose() as pose).

        Returns:
            Keypoints 또는 None (감지 실패).
        """
        frame.flags.writeable = False
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_ctx.process(image_rgb)

        if results.pose_landmarks:
            return self._landmarks_to_keypoints(results.pose_landmarks.landmark)
        return None

    def create_pose_context(self):
        """라이브 모드용 MediaPipe Pose 컨텍스트를 생성합니다."""
        return self._mp_pose.Pose(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        )

    @staticmethod
    def _landmarks_to_keypoints(landmarks) -> Keypoints:
        """MediaPipe 랜드마크를 Keypoints 데이터 클래스로 변환합니다."""
        data = {}
        for joint_name, idx in _LANDMARK_MAP.items():
            lm = landmarks[idx]
            data[joint_name] = [lm.x, lm.y, lm.z]
        return Keypoints(**data)


# ─── 단독 실행 (디버그용) ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    from src.vision.rule_engine import PoseRuleEngine

    parser = argparse.ArgumentParser(description="MediaPipe Pose 관절 추출기")
    parser.add_argument('--input', type=str, default='data/sample_1.mp4')
    parser.add_argument('--output', type=str, default='output/mediapipe_result.mp4')
    args = parser.parse_args()

    extractor = PoseExtractor()
    frames, fps = extractor.extract_from_video(args.input, args.output)

    # 간단한 룰 엔진 테스트
    engine = PoseRuleEngine(fps=fps)
    for f in frames:
        if f.keypoints:
            engine.feed_frame(f.frame_index, f.timestamp_ms, f.keypoints.to_dict())

    result = engine.analyze_throw()
    print("\n--- 분석 결과 ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
