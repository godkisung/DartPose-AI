"""AI Dart Coach — 스마트 실행 엔트리 포인트.

인자가 없으면 data/ 폴더의 최신 영상을 자동으로 분석합니다.
출력 경로는 입력 파일명을 기반으로 자동 생성됩니다.
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.mediapipe_pose import PoseExtractor
from src.vision.advanced_engine import AdvancedPoseEngine

def get_latest_video(search_dir="data"):
    """data 폴더에서 가장 최신 mp4 파일을 찾습니다."""
    files = glob.glob(os.path.join(search_dir, "*.mp4"))
    if not files:
        return None
    # 수정 시간 기준 정렬
    return max(files, key=os.path.getmtime)

def run_video_mode(input_path: str, output_video: str | None, output_json: str | None):
    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return

    # 1. 파일명 기반 자동 경로 설정
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%H%M")
    
    output_video = output_video or f"output/{base_name}_skeleton_{timestamp}.mp4"
    output_json = output_json or f"output/{base_name}_report_{timestamp}.json"

    print("=" * 60)
    print(f"🎯 분석 시작: {input_path}")
    print(f"📹 출력 영상: {output_video}")
    print("=" * 60)

    # Step 1: 관절 및 손가락 추출
    extractor = PoseExtractor()
    frames, fps = extractor.extract_from_video(input_path, output_video)

    # Step 2: 투구 분석
    engine = AdvancedPoseEngine(fps=fps)
    session = engine.analyze_session(frames)

    # Step 3: 결과 출력 및 저장
    print(f"\n📊 분석 결과 ({session.total_throws_detected}회 감지)")
    for t in session.throws:
        print(f"  [투구 {t.throw_index}] 릴리즈: {t.metrics.release_angle_deg:.1f}°, 가속: {t.metrics.release_timing_ms:.1f}ms")

    os.makedirs("output", exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 완료! 리포트 저장됨: {output_json}")

def main():
    parser = argparse.ArgumentParser(description="🎯 AI Dart Coach")
    parser.add_argument('--input', type=str, help="입력 비디오 (생략 시 최신 파일)")
    parser.add_argument('--all', action='store_true', help="data 폴더의 모든 영상 분석")
    args = parser.parse_args()

    if args.all:
        videos = glob.glob("data/*.mp4")
        for v in sorted(videos):
            run_video_mode(v, None, None)
    else:
        # 인자가 없으면 최신 영상 선택
        input_file = args.input or get_latest_video()
        if not input_file:
            print("❌ data/ 폴더에 분석할 mp4 파일이 없습니다.")
            sys.exit(1)
        run_video_mode(input_file, None, None)

if __name__ == "__main__":
    main()
