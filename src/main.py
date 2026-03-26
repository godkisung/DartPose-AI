"""AI Dart Coach — 실행 엔트리 포인트.

인자가 없으면 data/ 폴더의 최신 영상을 자동으로 분석합니다.
출력 경로는 입력 파일명을 기반으로 자동 생성됩니다.
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.mediapipe_pose import PoseExtractor
from src.vision.dart_analyzer import DartAnalyzer


def getLatestVideo(search_dir: str = "data") -> str | None:
    """data 폴더에서 가장 최신 mp4 파일을 찾습니다."""
    files = glob.glob(os.path.join(search_dir, "*.mp4"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def runVideoMode(
    input_path: str,
    output_video: str | None,
    output_json: str | None,
    debug_plot: bool = False,
) -> str | None:
    """단일 영상 파일을 분석합니다 (2-pass 아키텍처).

    1st pass: 포즈 추출 + 투구 분석 (세그먼트, 메트릭 계산)
    2nd pass: 투구별 궤적 오버레이 영상 생성

    Args:
        input_path: 입력 영상 파일 경로.
        output_video: 스켈레톤 시각화 출력 경로 (None이면 자동 생성).
        output_json: JSON 리포트 출력 경로 (None이면 자동 생성).
        debug_plot: True면 디버그 시각화 그래프를 생성.
        
    Returns:
        LLM 디버깅용으로 압축된 한 줄 요약 문자열 (str)
    """
    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return None

    # 파일명 기반 자동 출력 경로 설정
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%H%M")

    skeleton_video = output_video or f"output/{base_name}_skeleton_{timestamp}.mp4"
    output_json = output_json or f"output/{base_name}_report_{timestamp}.json"
    trajectory_video = f"output/{base_name}_trajectory_{timestamp}.mp4"

    print("=" * 60)
    print(f"🎯 분석 시작: {input_path}")
    print(f"📹 스켈레톤 영상: {skeleton_video}")
    print(f"🎨 궤적 영상: {trajectory_video}")
    print("=" * 60)

    # ━━━ 1st Pass: 포즈 추출 + 투구 분석 ━━━
    extractor = PoseExtractor()
    frames, fps = extractor.extract_from_video(input_path, skeleton_video)
    print(f"  ✓ 프레임 추출 완료: {len(frames)}개 / {fps:.1f}fps")

    sample_name = os.path.splitext(os.path.basename(input_path))[0]
    analyzer = DartAnalyzer(fps=fps, debug_plot=debug_plot)
    session = analyzer.analyze_session(frames, sample_name=sample_name)

    # Step 3: 결과 출력
    print(f"\n📊 분석 결과 ({session.total_throws_detected}회 감지)")
    for t in session.throws:
        print(
            f"  [투구 {t.throw_index}] "
            f"테이크백: {t.metrics.takeback_angle_deg:.1f}°, "
            f"릴리즈: {t.metrics.release_angle_deg:.1f}°, "
            f"가속: {t.metrics.release_timing_ms:.0f}ms, "
            f"최대속도: {t.metrics.max_elbow_velocity_deg_s:.0f}°/s"
        )

    if len(session.throws) >= 2:
        print(f"  일관성 점수: {session.throws[0].metrics.consistency_score:.1f}/100")

    # ━━━ 2nd Pass: 궤적 오버레이 영상 생성 ━━━
    if session.total_throws_detected > 0 and os.path.exists(skeleton_video):
        try:
            from src.vision.trajectory_renderer import TrajectoryRenderer

            # ThrowSegmenter에서 팔꿈치 각도 데이터 가져오기
            segmenter = analyzer._segmenter
            elbow_angles = getattr(segmenter, "_last_elbow_angles", None)
            if elbow_angles is None or len(elbow_angles) == 0:
                # fallback: 유효 프레임에서 직접 계산
                elbow_angles = np.zeros(len(frames))

            # 세그먼트 정보 가져오기
            valid_frames = [f for f in frames if f.keypoints is not None]
            segments = segmenter.segment(
                valid_frames,
                {},
                analyzer._detectThrowingSide(valid_frames),
            )

            throwing_side = analyzer._detectThrowingSide(valid_frames)
            renderer = TrajectoryRenderer()
            renderer.renderOverlay(
                skeleton_video, trajectory_video,
                valid_frames, segments, elbow_angles,
                throwing_side, fps,
            )
        except Exception as e:
            print(f"  ⚠ 궤적 오버레이 생성 실패: {e}")

    # Step 4: JSON 리포트 저장
    os.makedirs("output", exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n✅ 완료! 리포트 저장됨: {output_json}")

    # --- 🤖 LLM 토큰 최적화용 한 줄 요약 생성 ---
    llm_summary = f"{sample_name}: throws={session.total_throws_detected}"
    if session.total_throws_detected > 0:
        metrics_list = []
        for t in session.throws:
            m = t.metrics
            metrics_list.append(f"[tb={m.takeback_angle_deg:.1f},rel={m.release_angle_deg:.1f},vel={m.max_elbow_velocity_deg_s:.0f},time={m.release_timing_ms:.0f}ms]")
        llm_summary += f", metrics={''.join(metrics_list)}"
    else:
        llm_summary += ", metrics=[]"
    
    # 단일 파일 요약본 따로 저장
    llm_output_path = f"output/{base_name}_llm.txt"
    with open(llm_output_path, 'w', encoding='utf-8') as f:
        f.write(llm_summary)

    return llm_summary


def main() -> None:
    """CLI 엔트리 포인트."""
    parser = argparse.ArgumentParser(description="🎯 AI Dart Coach")
    parser.add_argument('--input', type=str, help="입력 비디오 경로 (생략 시 최신 파일)")
    parser.add_argument('--output-video', type=str, help="스켈레톤 영상 출력 경로")
    parser.add_argument('--output-json', type=str, help="JSON 리포트 출력 경로")
    parser.add_argument('--debug-plot', action='store_true',
                        help="분석 디버그 시각화 그래프 생성 (output/debug/에 저장)")
    parser.add_argument('--all', action='store_true', help="data 폴더의 모든 영상 분석")
    args = parser.parse_args()

    if args.all:
        videos = sorted(glob.glob("data/*.mp4"))
        if not videos:
            print("❌ data/ 폴더에 mp4 파일이 없습니다.")
            sys.exit(1)
        print(f"📂 {len(videos)}개 영상 일괄 분석 시작")
        
        # --- 🤖 배치 요약본 수집 리스트 ---
        batch_summaries = []
        
        for v in videos:
            summary = runVideoMode(v, None, None, debug_plot=args.debug_plot)
            if summary:
                batch_summaries.append(summary)
                
        # --- 🤖 최종 LLM 배치 리포트 저장 ---
        if batch_summaries:
            batch_report_path = "output/batch_llm_report.txt"
            with open(batch_report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(batch_summaries))
            print(f"\n🚀 전체 일괄 분석 완료! LLM 전용 압축 리포트 생성: {batch_report_path}")
            
    else:
        input_file = args.input or getLatestVideo()
        if not input_file:
            print("❌ data/ 폴더에 분석할 mp4 파일이 없습니다.")
            sys.exit(1)
        runVideoMode(input_file, args.output_video, args.output_json, debug_plot=args.debug_plot)


if __name__ == "__main__":
    main()