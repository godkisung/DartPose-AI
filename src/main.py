"""AI Dart Coach — 시스템 통합 엔트리 포인트.

비디오 모드: 비디오 → 관절 추출 → 투구 분리 → 분석 → 피드백 → JSON 리포트
라이브 모드: 카메라 + 하드웨어 센서 → 실시간 분석 (개발 예정)
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.mediapipe_pose import PoseExtractor
from src.vision.rule_engine import PoseRuleEngine
from src.llm.feedback_generator import FeedbackGenerator


def run_video_mode(input_path: str, output_video: str | None, output_json: str | None):
    """비디오 모드: 녹화된 영상을 분석합니다."""
    if not os.path.exists(input_path):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("🎯 AI Dart Coach — 비디오 분석 모드")
    print("=" * 60)

    # Step 1: 관절 추출
    print("\n[1/4] 관절 추출 중...")
    extractor = PoseExtractor()
    frames, fps = extractor.extract_from_video(input_path, output_video)

    frames_with_kp = sum(1 for f in frames if f.keypoints is not None)
    if frames_with_kp == 0:
        print("❌ 관절이 감지되지 않았습니다. 영상에 사람이 보이는지 확인하세요.")
        sys.exit(1)

    # Step 2: 투구 분석
    print("\n[2/4] 투구 분석 중...")
    engine = PoseRuleEngine(fps=fps)
    session = engine.analyze_session(frames)

    print(f"  → {session.total_throws_detected}개의 투구가 감지되었습니다.")

    # Step 3: 코칭 피드백 생성
    print("\n[3/4] 피드백 생성 중...")
    feedback_gen = FeedbackGenerator()
    feedback = feedback_gen.generate(session)
    session.llm_feedback = feedback

    # Step 4: 결과 출력
    print("\n[4/4] 결과 출력")
    print("\n" + feedback)

    # JSON 리포트 저장
    if output_json:
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n💾 JSON 리포트 저장: {output_json}")

    print("\n" + "=" * 60)
    print("✅ 분석 완료!")
    print("=" * 60)

    return session


def run_live_mode(port: str, simulate: bool):
    """라이브 모드: 카메라 + 하드웨어 센서 실시간 분석."""
    print("=" * 60)
    print("🎯 AI Dart Coach — 라이브 모드")
    print("=" * 60)

    if simulate:
        from src.hardware.serial_receiver import KeyboardSimulator

        def on_hit(event):
            print(f"  🎯 타격 감지: {event}")

        def on_reset():
            print("  🔄 리셋")

        sim = KeyboardSimulator(on_hit=on_hit, on_reset=on_reset)
        sim.start()

        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            sim.stop()
            print("\n종료.")
    else:
        from src.hardware.serial_receiver import ArduinoReceiver

        def on_hit(event):
            print(f"  🎯 타격 감지: {event}")

        def on_reset():
            print("  🔄 리셋")

        receiver = ArduinoReceiver(port=port, on_hit=on_hit, on_reset=on_reset)
        if not receiver.connect():
            print("❌ Arduino 연결 실패. --simulate 옵션을 사용해보세요.")
            sys.exit(1)

        receiver.start_listening()

        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            receiver.stop()
            print("\n종료.")


def main():
    parser = argparse.ArgumentParser(
        description="🎯 AI Dart Coach — 다트 투구 자세 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 비디오 분석
  uv run src/main.py --mode video --input data/sample_1.mp4

  # 비디오 분석 + 스켈레톤 오버레이 + JSON 리포트
  uv run src/main.py --mode video --input data/sample_1.mp4 --output-video output/result.mp4 --output-json output/report.json

  # 라이브 모드 (시뮬레이터)
  uv run src/main.py --mode live --simulate

  # 라이브 모드 (실제 하드웨어)
  uv run src/main.py --mode live --port /dev/ttyUSB0
        """,
    )

    parser.add_argument(
        '--mode', type=str, choices=['video', 'live'], default='video',
        help="실행 모드: 'video'(녹화 분석) 또는 'live'(실시간)",
    )
    parser.add_argument(
        '--input', type=str, default='data/sample_1.mp4',
        help="[video 모드] 입력 비디오 경로",
    )
    parser.add_argument(
        '--output-video', type=str, default=None,
        help="[video 모드] 스켈레톤 오버레이 비디오 저장 경로",
    )
    parser.add_argument(
        '--output-json', type=str, default='output/report.json',
        help="[video 모드] JSON 리포트 저장 경로",
    )
    parser.add_argument(
        '--port', type=str, default='/dev/ttyUSB0',
        help="[live 모드] Arduino 시리얼 포트",
    )
    parser.add_argument(
        '--simulate', action='store_true',
        help="[live 모드] 키보드 시뮬레이터 사용",
    )

    args = parser.parse_args()

    if args.mode == 'video':
        run_video_mode(args.input, args.output_video, args.output_json)
    else:
        run_live_mode(args.port, args.simulate)


if __name__ == "__main__":
    main()
