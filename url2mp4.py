import yt_dlp
import os
import sys

# --- Configuration ---
SCRIPT_DIR = os.getcwd() 
LINKS_FILE_PATH = os.path.join(SCRIPT_DIR, 'link.txt')
SAVE_PATH = os.path.join(SCRIPT_DIR, 'result')

# 리눅스에서는 보통 PATH에 등록되므로 None으로 설정해도 무방합니다.
# 만약 특정 위치에 직접 설치했다면 그 경로를 적어주세요.
FFMPEG_PATH = None 

# --- Main Script ---

print(f"Link file path: {LINKS_FILE_PATH}")
print(f"Save path: {SAVE_PATH}")

# 저장 디렉토리 생성
try:
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Ensured save directory exists: {SAVE_PATH}")
except OSError as e:
    print(f"Error creating directory {SAVE_PATH}: {e}")
    sys.exit(1)

# URL 목록 읽기
try:
    # 리눅스에서는 인코딩에 더 민감할 수 있으므로 utf-8 명시 유지
    with open(LINKS_FILE_PATH, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"Read {len(urls)} URLs from {LINKS_FILE_PATH}")
    if not urls:
        print("No URLs found in the file. Exiting.")
        sys.exit(0)
except FileNotFoundError:
    print(f"Error: Links file not found at {LINKS_FILE_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading links file {LINKS_FILE_PATH}: {e}")
    sys.exit(1)

# yt-dlp 옵션 설정
ydl_opts = {
    # 리눅스 파일 시스템에서 유효하지 않은 문자 처리를 위해 아래 템플릿 권장
    'outtmpl': os.path.join(SAVE_PATH, '%(title)s.%(ext)s'),
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'merge_output_format': 'mp4',
    'noplaylist': True,
}

# FFmpeg 경로 설정 (리눅스 환경 대응)
if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
    ydl_opts['ffmpeg_location'] = FFMPEG_PATH
elif FFMPEG_PATH:
    print(f"Warning: Specified FFmpeg path not found: {FFMPEG_PATH}")

# 다운로드 시작
print("\n--- Starting Downloads ---")
download_count = 0
fail_count = 0

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for i, url in enumerate(urls):
        # Shorts 및 기본 URL 처리
        clean_url = url.replace("youtube.com/shorts/", "youtube.com/watch?v=") if "shorts" in url else url
        
        print(f"[{i+1}/{len(urls)}] Processing: {clean_url}")

        try:
            ydl.download([clean_url])
            print(f"    → Success!")
            download_count += 1
        except Exception as e:
            print(f"    → Error: {e}")
            fail_count += 1

print(f"\nSummary: {download_count} success, {fail_count} failed.")
print(f"Files saved to: {SAVE_PATH}")