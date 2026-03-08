import cv2
import os
import argparse
from ultralytics import YOLO

def process_video_yolov8(input_path, output_path, model_name='models/yolov8n-pose.pt', conf=0.5):
    """
    YOLOv8-Pose를 사용하여 동영상의 관절(Keypoints)을 추출하고 결과를 저장합니다.

    Args:
        input_path: 입력 동영상 경로
        output_path: 결과 동영상 저장 경로
        model_name: 사용할 YOLO 모델 (기본값: yolov8n-pose.pt - n, s, m, l, x 중 선택)
        conf: confidence threshold (이 값보다 높은 신뢰도의 객체만 검출)
    """
    if not os.path.exists(input_path):
        print(f"Error: 입력 파일을 찾을 수 없습니다. 경로를 확인하세요: {input_path}")
        return

    # YOLO 모델 로드 (최초 실행 시 인터넷에서 모델 파일 다운로드)
    print(f"Loading '{model_name}' model...")
    model = YOLO(model_name)

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

    print(f"Processing video: {input_path} ({total_frames} frames)...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLOv8 추론 시작 (단일 프레임)
        # conf: 0.5 이상의 확실한 객체만 추출, save=False: 이미지 자체 저장은 안 함
        results = model.predict(source=frame, conf=conf, save=False, verbose=False)

        # 결과 렌더링 (원래 이미지 위에 점과 선을 그림)
        # results[0].plot() 함수가 Bounding Box와 Keypoints를 오버레이 해줍니다.
        annotated_frame = results[0].plot()

        # 만약 특정 관절 좌표만 빼서 계산하고 싶다면 아래와 같이 접근 가능합니다.
        # if results[0].keypoints is not None:
        #     keypoints = results[0].keypoints.xy[0] # 첫 번째 사람의 좌표 (x, y) 리스트
        #     # COCO Keypoints 인덱스: 
        #     # 5: 왼쪽 어깨, 7: 왼쪽 팔꿈치, 9: 왼쪽 손목
        #     # 6: 오른쪽 어깨, 8: 오른쪽 팔꿈치, 10: 오른쪽 손목
        #     if len(keypoints) > 10:
        #         r_shoulder = keypoints[6]
        #         r_elbow = keypoints[8]
        #         r_wrist = keypoints[10]
        #         # 여기서 각도 계산 등의 '규칙 엔진'을 적용할 수 있습니다.

        out.write(annotated_frame)
        
        # 진행 상황 출력
        if frame_count % 30 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames processed.")

    # 자원 해제
    cap.release()
    out.release()
    print(f"\n완료! 결과 영상이 저장되었습니다: {output_path}")

if __name__ == "__main__":
    # 실행 인자 설정
    parser = argparse.ArgumentParser(description="YOLOv8-Pose 관절 추출기")
    parser.add_argument('--input', type=str, default='data/KakaoTalk_20260306_232406312.mp4', help='입력 동영상 경로')
    parser.add_argument('--output', type=str, default='output/yolov8_result.mp4', help='출력 동영상 경로')
    parser.add_argument('--model', type=str, default='models/yolov8n-pose.pt', help='모델 종류 (models/yolov8n-pose.pt 등)')
    
    args = parser.parse_args()
    
    process_video_yolov8(args.input, args.output, args.model)
