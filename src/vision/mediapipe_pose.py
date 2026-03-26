"""MediaPipe 기반 통합 관절 및 손가락 추출 모듈 (프레임 시각화 강화 버전).

1. 화면 상단에 프레임 번호 크고 명확하게 표시.
2. ROI 및 랜드마크 시각화 유지 (분석용 데이터 수집).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from src.models import Keypoints, FrameData
from src.config import (
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONF,
    MEDIAPIPE_MIN_TRACKING_CONF,
)
from src.utils.video_utils import ensure_compatible_video

_POSE_MAP = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
}

_HAND_MAP = {
    'thumb_tip': 4, 'index_tip': 8, 'middle_tip': 12,
}

class PoseExtractor:
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

    def extract_from_video(self, input_path: str, output_path: str | None = None) -> tuple[list[FrameData], float]:
        compatible_path = ensure_compatible_video(input_path)
        cap = cv2.VideoCapture(compatible_path)
        if not cap.isOpened(): raise FileNotFoundError(f"비디오를 열 수 없습니다: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"🎯 프레임 시각화 모드 시작: {input_path}")

        frames: list[FrameData] = []
        
        with self._mp_pose.Pose(
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        ) as pose, self._mp_hands.Hands(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5,
            max_num_hands=1
        ) as hands:
            
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_index += 1
                
                debug_frame = frame.copy()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(image_rgb)
                
                keypoints_data = {}
                hand_detected_any = False
                
                if pose_results.pose_landmarks:
                    # Pose 그리기
                    self._mp_drawing.draw_landmarks(
                        debug_frame, pose_results.pose_landmarks, self._mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    for name, idx in _POSE_MAP.items():
                        lm = pose_results.pose_landmarks.landmark[idx]
                        keypoints_data[name] = [lm.x, lm.y, lm.z, lm.visibility]
                    
                    wrist_l = pose_results.pose_landmarks.landmark[15]
                    wrist_r = pose_results.pose_landmarks.landmark[16]
                    
                    for label, wrist in [('left', wrist_l), ('right', wrist_r)]:
                        if wrist.visibility > 0.5:
                            roi_x, roi_y = int(wrist.x * w), int(wrist.y * h)
                            margin = int(min(w, h) * 0.15)
                            x1, y1 = max(0, roi_x - margin), max(0, roi_y - margin)
                            x2, y2 = min(w, roi_x + margin), min(h, roi_y + margin)
                            
                            # ROI 박스 시각화 (빨간색)
                            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            if x2 > x1 and y2 > y1:
                                hand_roi = image_rgb[y1:y2, x1:x2]
                                hand_results = hands.process(hand_roi)
                                
                                if hand_results.multi_hand_landmarks:
                                    hand_detected_any = True
                                    h_lms = hand_results.multi_hand_landmarks[0].landmark
                                    self._mp_drawing.draw_landmarks(
                                        debug_frame[y1:y2, x1:x2], hand_results.multi_hand_landmarks[0], 
                                        self._mp_hands.HAND_CONNECTIONS)
                                    
                                    for h_name, h_idx in _HAND_MAP.items():
                                        lm = h_lms[h_idx]
                                        global_x = (x1 + lm.x * (x2 - x1)) / w
                                        global_y = (y1 + lm.y * (y2 - y1)) / h
                                        keypoints_data[f"{label}_{h_name}"] = [global_x, global_y, lm.z]

                # 💡 프레임 번호 및 상태 시각화 (매우 크게 표시)
                status_text = f"FRAME: {frame_index}"
                cv2.putText(debug_frame, status_text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 0), 4)
                
                hand_status = "HAND: YES" if hand_detected_any else "HAND: NO"
                cv2.putText(debug_frame, hand_status, (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0) if hand_detected_any else (0, 0, 255), 3)

                final_kp = self._create_keypoints_obj(keypoints_data)
                frames.append(FrameData(frame_index=frame_index, timestamp_ms=frame_index*(1000/fps), keypoints=final_kp))
                
                if out is not None:
                    out.write(debug_frame)
                    
                if frame_index % 100 == 0:
                    print(f"  진행: {frame_index}/{total_frames}")

        cap.release()
        if out is not None:
            out.release()
            print(f"✅ 시각화 영상 저장 완료: {output_path}")
        return frames, fps

    def _create_keypoints_obj(self, data: dict) -> Keypoints:
        fields = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_thumb_tip', 'right_thumb_tip', 'left_index_tip', 'right_index_tip', 'left_middle_tip', 'right_middle_tip']
        for f in fields:
            if f not in data: data[f] = None
        return Keypoints(**data)
