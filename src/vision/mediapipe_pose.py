import cv2
import mediapipe as mp
import argparse
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가하여 src 모듈 임포트 가능하게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.vision.rule_engine import PoseRuleEngine


def process_video_mediapipe(input_path, output_path):
    """
    MediaPipe Pose를 사용하여 동영상의 관절(Keypoints)을 추출하고 결과를 저장합니다.
    """
    if not os.path.exists(input_path):
        print(f"Error: 입력 파일을 찾을 수 없습니다. 경로를 확인하세요: {input_path}")
        return

    # MediaPipe 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 비디오 캡처 객체 초기화
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: 동영상을 열 수 없습니다: {input_path}")
        return

    # 원본 비디오 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 비디오 저장 객체 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 코덱
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Processing video with MediaPipe: {input_path} ({total_frames} frames)...")

    rule_engine = PoseRuleEngine(fps=fps)
    frame_count = 0
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2 # 가장 무겁지만 정확한 모델 (0, 1, 2)
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # BGR 이미지를 RGB로 변환 (MediaPipe는 RGB를 사용)
            frame.flags.writeable = False # 성능 향상을 위해 쓰기 불가 설정
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 이미지를 다시 BGR로 변환하여 그리기 가능하게 설정
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Pose 랜드마크 그리기 및 룰 엔진 데이터 전달
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 추출된 관절 데이터를 딕셔너리로 변환 (정규화된 0~1 좌표 사용)
                landmarks = results.pose_landmarks.landmark
                keypoints = {
                    'left_shoulder': [landmarks[11].x, landmarks[11].y, landmarks[11].z],
                    'right_shoulder': [landmarks[12].x, landmarks[12].y, landmarks[12].z],
                    'left_elbow': [landmarks[13].x, landmarks[13].y, landmarks[13].z],
                    'right_elbow': [landmarks[14].x, landmarks[14].y, landmarks[14].z],
                    'left_wrist': [landmarks[15].x, landmarks[15].y, landmarks[15].z],
                    'right_wrist': [landmarks[16].x, landmarks[16].y, landmarks[16].z],
                    'left_hip': [landmarks[23].x, landmarks[23].y, landmarks[23].z],
                    'right_hip': [landmarks[24].x, landmarks[24].y, landmarks[24].z],
                    'left_index': [landmarks[19].x, landmarks[19].y, landmarks[19].z],
                    'right_index': [landmarks[20].x, landmarks[20].y, landmarks[20].z],
                    'left_pinky': [landmarks[17].x, landmarks[17].y, landmarks[17].z],
                    'right_pinky': [landmarks[18].x, landmarks[18].y, landmarks[18].z],
                }
                timestamp_ms = frame_count * (1000.0 / fps)
                rule_engine.feed_frame(frame_count, timestamp_ms, keypoints)

            out.write(image)

            if frame_count % 30 == 0:
                print(f"Progress: {frame_count}/{total_frames} frames processed.")

    cap.release()
    out.release()
    print(f"\n완료! MediaPipe 결과 영상이 저장되었습니다: {output_path}")

    # Rule Engine 평가 수행
    print("\n--- 다트 투구 분석 결과 (Rule Engine) ---")
    analysis_result = rule_engine.analyze_throw()
    import json
    print(json.dumps(analysis_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Pose 관절 추출기")
    parser.add_argument('--input', type=str, default='data/KakaoTalk_20260306_232406312.mp4', help='입력 동영상 경로')
    parser.add_argument('--output', type=str, default='output/mediapipe_result.mp4', help='출력 동영상 경로')
    
    args = parser.parse_args()
    
    # output 폴더가 없으면 생성
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    if os.path.dirname(args.output) == '':
       pass # current dir
        
    process_video_mediapipe(args.input, args.output)
