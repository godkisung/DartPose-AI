#!/usr/bin/env python3
"""
sync_config.py
Python의 src/config.py 설정을 iOS의 Config.swift로 자동 동기화하는 도구입니다.
"""

import os
import re
import sys

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
SRC_CONFIG_PATH = os.path.join(BASE_DIR, "src", "config.py")
DEST_CONFIG_PATH = os.path.join(WORKSPACE_DIR, "darts-ios", "DartPose", "Models", "Config.swift")

def parse_python_config(content):
    """src/config.py에서 변수명과 값을 추출합니다."""
    # 전역 변수 또는 클래스 내 변수 매칭
    # 예: NORMALIZER_SMOOTHING_WINDOW = 5
    pattern = re.compile(r"^\s*([A-Z_0-9]+)\s*:\s*[^=]*=\s*([^#\n]+)", re.MULTILINE)
    matches = pattern.findall(content)
    
    # 타입 힌트가 없는 경우 대응
    if not matches:
        pattern = re.compile(r"^\s*([A-Z_0-9]+)\s*=\s*([^#\n]+)", re.MULTILINE)
        matches = pattern.findall(content)
        
    return matches

def to_camel_case(snake_str):
    """SNAKE_CASE를 camelCase로 변환합니다."""
    components = snake_str.lower().split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def generate_swift_config(pairs):
    """추출된 쌍을 바탕으로 Swift enum을 생성합니다."""
    lines = [
        "// Config.swift",
        "// 자동 생성된 파일입니다. 수동으로 수정하지 마세요.",
        "// 생성 도구: tools/sync_config.py",
        "",
        "import Foundation",
        "",
        "enum DartConfig {",
    ]
    
    for key, value in pairs:
        swift_key = to_camel_case(key)
        val = value.strip()
        
        # 타입 추론 (Double vs Int)
        if "." in val or "e" in val.lower():
            swift_line = f"    static let {swift_key}: Double = {val}"
        else:
            # 숫자가 아닌 경우 (예: "right") 처리
            if '"' in val or "'" in val:
                clean_val = val.replace("'", '"')
                swift_line = f'    static let {swift_key}: String = {clean_val}'
            else:
                swift_line = f"    static let {swift_key}: Int = {val}"
                
        lines.append(swift_line)
        
    lines.append("}")
    return "\n".join(lines) + "\n"

def main():
    if not os.path.exists(SRC_CONFIG_PATH):
        print(f"Error: {SRC_CONFIG_PATH}를 찾을 수 없습니다.")
        sys.exit(1)
        
    print(f"Reading {SRC_CONFIG_PATH}...")
    with open(SRC_CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 전역 변수 추출
    pairs = parse_python_config(content)
    
    if not pairs:
        print("Warning: 설정을 하나도 찾지 못했습니다.")
        
    print(f"Found {len(pairs)} config items.")
    
    # Swift 파일 생성
    swift_content = generate_swift_config(pairs)
    
    os.makedirs(os.path.dirname(DEST_CONFIG_PATH), exist_ok=True)
    with open(DEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(swift_content)
        
    print(f"Successfully synced to {DEST_CONFIG_PATH}")

if __name__ == "__main__":
    main()
