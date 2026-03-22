"""고도화된 다트 투구 분석 엔진 (릴리즈 타이밍 및 각도 보정 버전).

1. 릴리즈 탐색 시점을 테이크백 정점 '이후'로 고정 (가속 시간 0ms 방지).
2. 30fps 환경에 맞게 가속 시간 임계값 최적화.
3. 릴리즈 각도 계산 시 좌표계 반전 및 방향성 보정.
"""

import numpy as np
from typing import List, Optional, Dict
from src.models import (
    FrameData, Keypoints, ThrowPhases, ThrowMetrics, ThrowAnalysis, SessionResult
)
from src.config import THROW_MIN_FRAMES

class AdvancedPoseEngine:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps

    def analyze_session(self, frames: List[FrameData]) -> SessionResult:
        valid_frames = [f for f in frames if f.keypoints is not None]
        if len(valid_frames) < THROW_MIN_FRAMES:
            return SessionResult(len(frames), self.fps, 0)

        side = self._detect_active_side_robust(valid_frames)
        print(f"  ℹ 최종 판정 투구 팔: {side}")
        
        candidates = self._segment_throws_advanced(valid_frames, side)
        candidates = self._merge_close_segments(candidates)
        
        analyses = []
        for i, segment in enumerate(candidates):
            f_start, f_end = segment[0].frame_index, segment[-1].frame_index
            
            # 1. 분석 수행 (릴리즈 감지 로직 포함)
            analysis = self._analyze_single_throw_advanced(segment, side, len(analyses) + 1)
            if not analysis:
                print(f"  ℹ 후보 {i+1} ({f_start}-{f_end}): 분석 불가")
                continue

            # 2. 움직임 및 타이밍 검증
            is_valid, reason = self._validate_throw_movement(analysis, segment, side)
            if is_valid:
                analyses.append(analysis)
                print(f"  ✅ 투구 {len(analyses)} 확정 ({f_start}-{f_end})")
            else:
                print(f"  ℹ 후보 {i+1} ({f_start}-{f_end}): 거절 (사유: {reason})")

        return SessionResult(
            total_frames=len(frames), fps=self.fps,
            total_throws_detected=len(analyses), throws=analyses
        )

    def _merge_close_segments(self, segments: List[List[FrameData]]) -> List[List[FrameData]]:
        if len(segments) < 2: return segments
        merged = []
        curr = segments[0]
        for next_seg in segments[1:]:
            gap = next_seg[0].frame_index - curr[-1].frame_index
            if gap < 20:
                print(f"  ℹ 세그먼트 병합 (간격: {gap}f)")
                curr.extend(next_seg)
            else:
                merged.append(curr)
                curr = next_seg
        merged.append(curr)
        return merged

    def _validate_throw_movement(self, analysis: ThrowAnalysis, segment: List[FrameData], side: str) -> tuple[bool, str]:
        wrist_key = f"{side}_wrist"
        try:
            wrist_coords = np.array([f.keypoints.get(wrist_key) for f in segment if f.keypoints.get(wrist_key)])
            max_disp = np.max(np.linalg.norm(wrist_coords[:, :2] - wrist_coords[0, :2], axis=1))
            
            # 1. 변위 체크 (0.12로 약간 완화)
            if max_disp < 0.12: return False, f"변위 부족 ({max_disp:.3f})"
            
            # 2. 가속 시간 체크 (10ms 이상이면 1프레임이라도 전진한 것)
            if analysis.metrics.release_timing_ms < 10: 
                return False, f"가속 시간 부족 ({analysis.metrics.release_timing_ms:.1f}ms)"
            
            return True, ""
        except:
            return False, "검증 오류"

    def _analyze_single_throw_advanced(self, frames: List[FrameData], side: str, index: int) -> Optional[ThrowAnalysis]:
        scaled_data = self._preprocess_throw_data(frames, side)
        local_p = self._detect_phases_fsm_local(scaled_data, side)
        if not local_p: return None

        metrics = self._calculate_metrics_advanced_local(scaled_data, local_p, side)
        
        global_p = ThrowPhases(
            address=frames[local_p['address']].frame_index,
            takeback_start=frames[local_p['takeback_start']].frame_index,
            takeback_max=frames[local_p['takeback_max']].frame_index,
            release=frames[local_p['release']].frame_index,
            follow_through=frames[local_p['follow_through']].frame_index
        )
        return ThrowAnalysis(index, side, (frames[0].frame_index, frames[-1].frame_index), global_p, metrics)

    def _preprocess_throw_data(self, frames: List[FrameData], side: str) -> Dict[str, np.ndarray]:
        joints = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", f"{side}_hip", f"{side}_thumb_tip", f"{side}_index_tip"]
        raw = {j: np.array([f.keypoints.get(j) if f.keypoints.get(j) else [0.0,0.0,0.0] for f in frames]) for j in joints}
        dist_sh = np.linalg.norm(raw[f"{side}_shoulder"] - raw[f"{side}_hip"], axis=1)
        scale = np.mean(dist_sh) if np.mean(dist_sh) > 0.01 else 1.0
        return {j: (raw[j] - raw[f"{side}_shoulder"]) / scale for j in joints}

    def _detect_phases_fsm_local(self, data: Dict[str, np.ndarray], side: str) -> dict:
        wrist, thumb = data[f"{side}_wrist"], data.get(f"{side}_thumb_tip")
        # 정점: 가장 뒤로 당겨진 시점
        tb_max = int(np.argmax(np.abs(wrist[:, 0] - wrist[0, 0])))
        
        release = -1
        # 💡 핵심: 정점 '다음' 프레임부터 손가락 출현을 탐색 (가속 구간 확보)
        if thumb is not None and len(thumb) > tb_max + 1:
            for i in range(tb_max + 1, len(thumb)):
                if not np.all(thumb[i] == 0): release = i; break
        
        if release == -1:
            vel = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
            # 정점 이후 최대 속도 지점을 릴리즈로 (fallback)
            if len(vel) > tb_max + 1:
                release = int(tb_max + 1 + np.argmax(vel[tb_max+1:]))
            else:
                release = len(wrist) - 1
                
        return {'address': 0, 'takeback_start': int(max(0, tb_max-5)), 'takeback_max': tb_max, 'release': release, 'follow_through': int(min(len(wrist)-1, release+10))}

    def _calculate_metrics_advanced_local(self, data: Dict[str, np.ndarray], p: dict, side: str) -> ThrowMetrics:
        wrist, elbow = data[f"{side}_wrist"], data[f"{side}_elbow"]
        rel_idx = min(len(wrist)-1, p['release'])
        
        # 릴리즈 각도 보정: 지면 대비 팔뚝의 각도 (Y축 반전 고려)
        # MediaPipe Y는 아래가 +, 위가 - 임을 고려하여 계산
        forearm_vec = wrist[rel_idx] - elbow[rel_idx]
        # x좌표의 절대값을 취해 투구 방향(좌/우)에 상관없이 0~180도 사이로 산출
        angle = np.degrees(np.arctan2(-forearm_vec[1], np.abs(forearm_vec[0])))
        
        # 가속 시간 (정점 ~ 릴리즈)
        timing = (rel_idx - p['takeback_max']) * self.dt * 1000
        
        return ThrowMetrics(release_angle_deg=float(angle), release_timing_ms=float(timing))

    def _detect_active_side_robust(self, frames: List[FrameData]) -> str:
        l_hand = sum(1 for f in frames if f.keypoints.left_thumb_tip is not None)
        r_hand = sum(1 for f in frames if f.keypoints.right_thumb_tip is not None)
        if abs(l_hand - r_hand) > 5: return "left" if l_hand > r_hand else "right"
        lw = [f.keypoints.left_wrist[0] for f in frames if f.keypoints.left_wrist]
        rw = [f.keypoints.right_wrist[0] for f in frames if f.keypoints.right_wrist]
        return "left" if np.var(lw) > np.var(rw) else "right"

    def _segment_throws_advanced(self, frames: List[FrameData], side: str) -> List[List[FrameData]]:
        wrist_key = f"{side}_wrist"
        
        # 💡 None 값 방어 로직 추가: 데이터가 없으면 [0,0,0]으로 채우거나 이전 값 유지
        positions = []
        last_valid = [0.0, 0.0, 0.0]
        for f in frames:
            pos = f.keypoints.get(wrist_key)
            if pos is not None:
                last_valid = pos
                positions.append(pos)
            else:
                positions.append(last_valid)
        
        wrist_pos = np.array(positions)
        if len(wrist_pos) < 2: return []
        
        vels = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
        thresh = max(np.percentile(vels, 92), 0.015)
        segments, current, idle = [], [], 0
        for i, v in enumerate(vels):
            if v > thresh: idle = 0; current.append(frames[i])
            else:
                if current:
                    idle += 1
                    if idle > 15:
                        if len(current) > 25: segments.append(current)
                        current, idle = [], 0
                    else: current.append(frames[i])
        if len(current) > 25: segments.append(current)
        return segments
