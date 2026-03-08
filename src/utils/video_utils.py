"""비디오 코덱 호환성 유틸리티.

OpenCV가 처리할 수 없는 코덱(AV1 등)의 비디오를 H.264로 자동 변환합니다.
"""

import subprocess
import tempfile
import os
import cv2


def get_video_codec(path: str) -> str:
    """ffprobe를 이용하여 비디오의 코덱 이름을 반환합니다."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip().lower()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _can_opencv_read(path: str) -> bool:
    """OpenCV로 첫 프레임을 읽을 수 있는지 확인합니다."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret


def transcode_to_h264(input_path: str, output_path: str) -> None:
    """ffmpeg를 이용하여 비디오를 H.264/mp4로 변환합니다."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ],
        capture_output=True, text=True, check=True, timeout=120,
    )


def ensure_compatible_video(input_path: str) -> str:
    """OpenCV가 읽을 수 있는 비디오 경로를 반환합니다.

    이미 호환되는 코덱이면 원본 경로를 그대로 반환하고,
    호환되지 않으면 H.264로 변환한 임시 파일 경로를 반환합니다.

    Returns:
        읽을 수 있는 비디오 파일의 경로.
    """
    if _can_opencv_read(input_path):
        return input_path

    codec = get_video_codec(input_path)
    print(f"⚠ 비디오 코덱 '{codec}'은 OpenCV와 호환되지 않습니다. H.264로 변환합니다...")

    # 원본과 같은 디렉토리에 _h264 접미사로 변환 파일 저장
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_h264.mp4"

    # 이미 변환된 파일이 있으면 재사용
    if os.path.exists(output_path) and _can_opencv_read(output_path):
        print(f"✓ 기존 변환 파일을 재사용합니다: {output_path}")
        return output_path

    transcode_to_h264(input_path, output_path)
    print(f"✓ 변환 완료: {output_path}")
    return output_path
