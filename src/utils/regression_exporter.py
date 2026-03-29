# regression_exporter.py
# 파이썬 엔진의 중간 결과와 최종 결과를 JSON으로 내보내어
# iOS Swift 엔진의 정확성을 검증하기 위한 데이터를 생성합니다.

import json
import os
from dataclasses import asdict

class RegressionExporter:
    """분석 데이터를 JSON으로 내보내는 유틸리티"""
    
    def __init__(self, output_dir="tests/regression_data"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export_session(self, frames, session_result, filename="regression_case_1.json"):
        """전체 세션 데이터와 기대되는 분석 결과를 저장합니다."""
        
        # 1. 입력 프레임 데이터 (Keypoints 포함)
        # asdict()는 dataclass를 dict로 변환합니다.
        frames_data = [asdict(f) for f in frames]
        
        # 2. 기대되는 최종 결과
        expected_result = asdict(session_result)
        
        data = {
            "metadata": {
                "source": "Python Analysis Engine",
                "fps": session_result.fps,
                "total_frames": session_result.total_frames
            },
            "input_frames": frames_data,
            "expected_output": expected_output_filter(expected_result)
        }
        
        target_path = os.path.join(self.output_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Regression data exported to {target_path}")

def expected_output_filter(result_dict):
    """테스트에 필요한 핵심 필드만 남깁니다."""
    # UUID 등은 스위프트에서 보장하기 어려우므로 제외
    filtered_throws = []
    for t in result_dict.get("throws", []):
        filtered_t = {
            "throw_index": t["throw_index"],
            "throwing_arm": t["throwing_arm"],
            "frame_range": t["frame_range"],
            "phases": t["phases"],
            "metrics": t["metrics"]
        }
        filtered_throws.append(filtered_t)
        
    return {
        "total_throws_detected": result_dict["total_throws_detected"],
        "throws": filtered_throws,
        "consistency_score": result_dict.get("consistency_score", 0)
    }
