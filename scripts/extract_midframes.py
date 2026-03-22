"""각 sample 영상의 중간 프레임 스냅샷을 추출합니다.

용도: 촬영 각도별 인식률을 문서화하여 클라이언트 가이드라인을 만들기 위한 자료 수집.
각 영상의 중간 프레임(50%), 1/4 프레임(25%), 3/4 프레임(75%)을 추출합니다.
MediaPipe 스켈레톤 오버레이 포함/미포함 두 버전을 저장합니다.
"""

import cv2
import glob
import os
import sys
import mediapipe as mp

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.video_utils import ensure_compatible_video


def extractMidframes(
    video_path: str,
    output_dir: str,
    with_skeleton: bool = True,
) -> list[str]:
    """영상에서 중간 프레임을 추출합니다.

    Args:
        video_path: 입력 영상 경로.
        output_dir: 출력 디렉토리.
        with_skeleton: True면 MediaPipe 스켈레톤 오버레이 포함.

    Returns:
        저장된 이미지 파일 경로 리스트.
    """
    compatible_path = ensure_compatible_video(video_path)
    cap = cv2.VideoCapture(compatible_path)
    if not cap.isOpened():
        print(f"❌ 열 수 없음: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_name = os.path.splitext(os.path.basename(video_path))[0]

    # 추출 위치: 25%, 50%, 75%
    positions = {
        "25pct": int(total_frames * 0.25),
        "50pct": int(total_frames * 0.50),
        "75pct": int(total_frames * 0.75),
    }

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # MediaPipe 설정 (스켈레톤용)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,  # 속도 우선
    ) as pose:
        for label, frame_idx in positions.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # 원본 프레임 저장
            raw_path = os.path.join(output_dir, f"{sample_name}_{label}_raw.jpg")
            cv2.imwrite(raw_path, frame)
            saved.append(raw_path)

            # 스켈레톤 오버레이 프레임 저장
            if with_skeleton:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                annotated = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                    # 감지 신뢰도 표시
                    visibility_sum = sum(
                        lm.visibility for lm in results.pose_landmarks.landmark
                    )
                    avg_visibility = visibility_sum / len(results.pose_landmarks.landmark)
                    cv2.putText(
                        annotated,
                        f"Avg Visibility: {avg_visibility:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                else:
                    cv2.putText(
                        annotated,
                        "NO POSE DETECTED",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )

                # 프레임 정보 표시
                time_sec = frame_idx / fps if fps > 0 else 0
                cv2.putText(
                    annotated,
                    f"Frame {frame_idx} / {total_frames}  ({time_sec:.1f}s)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                )

                skel_path = os.path.join(output_dir, f"{sample_name}_{label}_skeleton.jpg")
                cv2.imwrite(skel_path, annotated)
                saved.append(skel_path)

    cap.release()
    print(f"  ✅ {sample_name}: {len(saved)}장 저장 → {output_dir}")
    return saved


def main():
    """data/ 폴더의 모든 영상에서 중간 프레임을 추출합니다."""
    videos = sorted(glob.glob("data/*.mp4"))
    if not videos:
        print("❌ data/ 폴더에 mp4 파일이 없습니다.")
        sys.exit(1)

    output_dir = "output/midframes"
    print(f"📸 {len(videos)}개 영상에서 중간 프레임 추출")
    print(f"📂 출력: {output_dir}/")

    all_saved = []
    for v in videos:
        saved = extractMidframes(v, output_dir)
        all_saved.extend(saved)

    print(f"\n✅ 총 {len(all_saved)}장 추출 완료")


if __name__ == "__main__":
    main()
