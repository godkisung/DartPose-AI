import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="AI Darts Coaching System")
    parser.add_argument('--mode', type=str, choices=['video', 'live'], default='video',
                        help="Run in 'video' mode for a file, or 'live' mode for camera+sensor integration")
    parser.add_argument('--video-path', type=str, default='data/KakaoTalk_20260306_232406312.mp4')
    args = parser.parse_args()

    print("Initializing AI Darts Platform...")
    
    if args.mode == 'live':
        print("Starting Live Mode: Waiting for Hardware Sensor Trigger...")
        # 1. Initialize Serial Receiver
        # 2. Initialize Camera Buffers
        # 3. Enter Event Loop
    else:
        print(f"Starting Video Processing Mode: {args.video_path}")
        # 1. Load Video
        # 2. Extract Poses frame-by-frame
        # 3. Run Rule Engine on extracted poses
        # 4. Generate LLM Feedback
        print("\nNote: For actual implementation, please refer to src/vision/yolo_pose.py and mediapipe_pose.py for keypoint extraction.")

if __name__ == "__main__":
    main()
