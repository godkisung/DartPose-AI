"""투구 세분화 모듈 — FSM(상태 기계) 기반.

연속 영상에서 개별 투구 구간을 자동으로 분리합니다.
팔꿈치 각도(어깨-팔꿈치-손목) 사이클을 FSM으로 추적하여
투구를 카운트합니다.

다른 앱 참고:
- 손목 각도(=팔꿈치 각도) 변화로 투구 판정
- 각 투구마다 궤적을 색상별로 시각화

차별점:
- FSM + hysteresis로 노이즈 강건성 확보
- 각속도 보조 신호 활용
- 무효 투구 자동 필터링

핵심 알고리즘:
1. 프레임별 팔꿈치 각도 계산 (카메라 거리 불변)
2. 가우시안 스무딩으로 노이즈 제거
3. FSM 상태 전환으로 투구 사이클 감지:
   IDLE → COCKING → RELEASING → FOLLOW_THROUGH → IDLE
4. 각 사이클을 하나의 투구 세그먼트로 반환

iOS Swift(Accelerate/vDSP) 포팅을 고려하여 numpy 만으로 구현합니다.
"""

from enum import Enum
import numpy as np
from src.models import FrameData
from src.config import (
    SEGMENTER_SMOOTHING_SIGMA,
    SEGMENTER_ANGLE_DROP_THRESHOLD,
    SEGMENTER_RELEASE_ANGLE_RISE,
    SEGMENTER_IDLE_STABILITY_FRAMES,
    SEGMENTER_MIN_COCKING_ANGLE,
    SEGMENTER_MIN_SEGMENT_FRAMES,
    SEGMENTER_MIN_THROW_INTERVAL_S,
    SEGMENTER_SEGMENT_PAD_S,
    SEGMENTER_MERGE_GAP_FRAMES,
)


class _FSMState(Enum):
    """투구 감지 FSM 상태.

    IDLE: 대기 상태 (팔이 펴져 있거나 정지)
    COCKING: 팔 접기 진행 중 (테이크백)
    RELEASING: 팔 펴기 진행 중 (릴리즈)
    FOLLOW_THROUGH: 팔로스루 진행 중 (팔이 다 펴진 후 안정화 대기)
    """
    IDLE = "idle"
    COCKING = "cocking"
    RELEASING = "releasing"
    FOLLOW_THROUGH = "follow_through"


class ThrowSegmenter:
    """팔꿈치 각도 사이클 FSM으로 투구 구간을 분리하는 클래스.

    기존 방식의 문제점:
    - 손목 2D 속도 피크: 카메라 거리/각도에 따라 신호 크기 변동
    - 적응적 prominence: 영상마다 임계값이 달라 불안정

    개선된 방식 (FSM 기반):
    - 팔꿈치 각도는 카메라 거리에 불변 (벡터 내적 기반)
    - FSM 상태 전환: IDLE → COCKING → RELEASING → FOLLOW_THROUGH → IDLE
    - 각 전환에 hysteresis 적용하여 노이즈에 강건
    - 투구의 물리적 동작("팔 접기 → 팔 펴기")을 직접 감지
    """

    def __init__(
        self,
        fps: float = 30.0,
        smoothing_sigma: float = SEGMENTER_SMOOTHING_SIGMA,
        angle_drop_threshold: float = SEGMENTER_ANGLE_DROP_THRESHOLD,
        release_angle_rise: float = SEGMENTER_RELEASE_ANGLE_RISE,
        idle_stability_frames: int = SEGMENTER_IDLE_STABILITY_FRAMES,
        min_cocking_angle: float = SEGMENTER_MIN_COCKING_ANGLE,
        min_segment_frames: int = SEGMENTER_MIN_SEGMENT_FRAMES,
        min_throw_interval_s: float = SEGMENTER_MIN_THROW_INTERVAL_S,
        segment_pad_s: float = SEGMENTER_SEGMENT_PAD_S,
        merge_gap_frames: int = SEGMENTER_MERGE_GAP_FRAMES,
    ):
        """초기화.

        Args:
            fps: 영상 프레임레이트.
            smoothing_sigma: 가우시안 스무딩 표준편차 (초 단위).
            angle_drop_threshold: IDLE→COCKING 전환 각도 감소량 (도).
            release_angle_rise: 테이크백 최저 각도 대비 릴리즈 판정 각도 증가량 (도).
            idle_stability_frames: FOLLOW_THROUGH→IDLE 전환 안정화 프레임 수.
            min_cocking_angle: 유효 투구 인정 최소 각도 변화 (도).
            min_segment_frames: 유효 세그먼트 최소 프레임 수.
            min_throw_interval_s: 연속 투구 간 최소 시간 간격 (초).
            segment_pad_s: 세그먼트 전후 패딩 시간 (초).
            merge_gap_frames: 병합 간격 프레임 수.
        """
        self.fps = fps
        self.smoothing_sigma = smoothing_sigma
        self.angle_drop_threshold = angle_drop_threshold
        self.release_angle_rise = release_angle_rise
        self.idle_stability_frames = idle_stability_frames
        self.min_cocking_angle = min_cocking_angle
        self.min_segment_frames = min_segment_frames
        self.min_throw_interval = max(1, int(min_throw_interval_s * fps))
        self.segment_pad = max(1, int(segment_pad_s * fps))
        self.merge_gap_frames = merge_gap_frames

        # 디버그 시각화용 데이터 (analyze 후 접근 가능)
        self._last_elbow_angles: np.ndarray = np.array([])
        self._last_smoothed_angles: np.ndarray = np.array([])
        self._last_fsm_states: list[str] = []
        self._last_cycle_boundaries: list[dict] = []
        # 기존 호환용: 디버그 플롯에서 wrist velocity 대신 elbow angle 사용
        self._last_velocity: np.ndarray = np.array([])
        self._last_peaks: list[int] = []

    # ─── Public API ─────────────────────────────────────────────────────────

    def segment(
        self,
        frames: list[FrameData],
        normalized_data: dict[str, np.ndarray],
        throwing_side: str,
    ) -> list[list[FrameData]]:
        """투구 세그먼트를 분리합니다.

        팔꿈치 각도(어깨-팔꿈치-손목)의 사이클을 FSM으로 추적하여
        개별 투구 구간을 감지합니다.

        Args:
            frames: keypoints가 있는 FrameData 리스트.
            normalized_data: PoseNormalizer.normalize()의 출력 (API 호환용).
            throwing_side: 투구 팔 방향 ('left' 또는 'right').

        Returns:
            투구 세그먼트 리스트. 각 원소는 [FrameData, ...] 리스트.
        """
        if len(frames) < self.min_segment_frames:
            self._clearDebugData()
            return [frames] if frames else []

        # ─ Step 1: 원시 좌표에서 관절 좌표 추출 ──────────────────────────
        shoulder_coords = self._extractRawJointCoords(frames, f"{throwing_side}_shoulder")
        elbow_coords = self._extractRawJointCoords(frames, f"{throwing_side}_elbow")
        wrist_coords = self._extractRawJointCoords(frames, f"{throwing_side}_wrist")

        if shoulder_coords is None or elbow_coords is None or wrist_coords is None:
            self._clearDebugData()
            return [frames]

        # ─ Step 2: 프레임별 팔꿈치 각도 계산 (카메라 거리 불변) ──────────
        elbow_angles = self._computeElbowAngles(shoulder_coords, elbow_coords, wrist_coords)

        # ─ Step 3: 가우시안 스무딩 ──────────────────────────────────────
        smoothed_angles = self._gaussianSmooth(elbow_angles, self.smoothing_sigma)

        # 디버그 데이터 저장
        self._last_elbow_angles = elbow_angles
        self._last_smoothed_angles = smoothed_angles
        # 기존 호환용: 디버그 플롯에서 사용
        self._last_velocity = smoothed_angles

        # ─ Step 4: FSM으로 투구 사이클 감지 ──────────────────────────────
        cycles = self._runFSM(smoothed_angles)
        self._last_cycle_boundaries = cycles

        # 피크 위치 (디버그 호환용: 각 사이클의 최저 각도 프레임)
        self._last_peaks = [c["min_angle_frame"] for c in cycles]

        if len(cycles) == 0:
            print("  ⚠ FSM 사이클 감지 실패 — 전체를 1개 투구로 처리")
            return [frames]

        print(f"  ℹ FSM 감지된 투구 사이클: {len(cycles)}개")
        for i, c in enumerate(cycles):
            print(f"    사이클 {i+1}: frames {c['start']}~{c['end']}, "
                  f"각도 변화: {c['angle_range']:.1f}°, "
                  f"최저 각도: {c['min_angle']:.1f}° (frame {c['min_angle_frame']})")

        # ─ Step 5: 사이클을 세그먼트로 변환 (패딩 포함) ──────────────────
        raw_segments = self._cyclesToSegments(frames, cycles)

        # ─ Step 6: 무효 세그먼트 필터링 ──────────────────────────────────
        filtered = self._filterInvalidSegments(raw_segments, frames, throwing_side)

        # ─ Step 7: 가까운 세그먼트 병합 ──────────────────────────────────
        merged = self._mergeCloseSegments(filtered)

        return merged if merged else [frames]

    # ─── FSM Core ────────────────────────────────────────────────────────────

    def _runFSM(self, smoothed_angles: np.ndarray) -> list[dict]:
        """FSM을 실행하여 투구 사이클 경계를 감지합니다.

        개선된 로직:
        - IDLE: '최근 윈도우 내 최대 각도'를 추적. 현재 각도가 이 최대값에서
                angle_drop_threshold 이상 감소하면 → COCKING
        - COCKING: 각도 감소 추적 (최저 각도 갱신).
                   각도가 최저점에서 release_angle_rise 이상 증가하면 → RELEASING
        - RELEASING: 각도 증가 추적 (팔 펴기).
                     각도가 안정되거나 cocking_start 각도에 근접하면 → FOLLOW_THROUGH
        - FOLLOW_THROUGH: idle_stability_frames 동안 안정되면 → IDLE (1회 완료)

        각도 변화(angle_range)는 'cocking 직전 최대각도 - 최저 각도'로 계산하여
        스무딩에 의한 극값 손실 문제를 방지합니다.

        Args:
            smoothed_angles: (약하게) 스무딩된 팔꿈치 각도 시계열 (도).

        Returns:
            투구 사이클 경계 리스트.
        """
        n = len(smoothed_angles)
        state = _FSMState.IDLE
        cycles: list[dict] = []

        # 로컬 최대 각도 추적 윈도우 크기 (약 0.5초)
        _LOOKBACK = max(5, int(self.fps * 0.5))

        # FSM 상태 변수
        recent_max_angle = float(smoothed_angles[0])  # 최근 윈도우 최대 각도
        cocking_entry_angle = 0.0   # COCKING 진입 시점의 각도 (= 직전 IDLE 최대)
        cycle_start = 0             # 현재 사이클 시작 프레임
        min_angle = float('inf')    # COCKING 중 최저 각도
        min_angle_frame = 0         # 최저 각도 프레임
        stability_count = 0         # FOLLOW_THROUGH 안정화 카운터
        last_cycle_end = -self.min_throw_interval  # 마지막 사이클 종료 프레임

        # hysteresis 최소 체류 프레임 (상태당 최소 3프레임)
        _MIN_STATE_FRAMES = max(3, int(self.fps * 0.1))
        state_entry_frame = 0

        # FSM 상태 기록 (디버그용)
        fsm_states: list[str] = []

        for i in range(n):
            angle = float(smoothed_angles[i])
            frames_in_state = i - state_entry_frame
            fsm_states.append(state.value)

            if state == _FSMState.IDLE:
                # 최근 윈도우 내 최대 각도 추적
                lookback_start = max(0, i - _LOOKBACK)
                recent_max_angle = float(np.max(smoothed_angles[lookback_start:i + 1]))

                # 현재 각도가 최근 최대에서 drop_threshold 이상 떨어지면 → COCKING
                drop = recent_max_angle - angle
                if drop > self.angle_drop_threshold:
                    if i - last_cycle_end >= self.min_throw_interval:
                        state = _FSMState.COCKING
                        state_entry_frame = i
                        cocking_entry_angle = recent_max_angle
                        cycle_start = max(0, i - _MIN_STATE_FRAMES)
                        min_angle = angle
                        min_angle_frame = i

            elif state == _FSMState.COCKING:
                # 최저 각도 갱신
                if angle < min_angle:
                    min_angle = angle
                    min_angle_frame = i

                if frames_in_state >= _MIN_STATE_FRAMES:
                    # 각도가 최저점에서 release_angle_rise 이상 증가하면 → RELEASING
                    if angle - min_angle > self.release_angle_rise:
                        state = _FSMState.RELEASING
                        state_entry_frame = i

                    # 타임아웃: 2초 이상 COCKING이면 무효 → IDLE
                    elif frames_in_state > int(2.0 * self.fps):
                        state = _FSMState.IDLE
                        state_entry_frame = i

            elif state == _FSMState.RELEASING:
                if frames_in_state >= _MIN_STATE_FRAMES:
                    # 각도가 cocking 진입 각도 근처(±15°)로 돌아오면 → FOLLOW_THROUGH
                    if angle >= cocking_entry_angle - 15.0:
                        state = _FSMState.FOLLOW_THROUGH
                        state_entry_frame = i
                        stability_count = 0

                    # 타임아웃: 1.5초 이상 지속되면 강제 전환
                    elif frames_in_state > int(1.5 * self.fps):
                        state = _FSMState.FOLLOW_THROUGH
                        state_entry_frame = i
                        stability_count = 0

            elif state == _FSMState.FOLLOW_THROUGH:
                # 각도 변화가 안정(±2°/프레임 이하)이면 카운터 증가
                if i > 0:
                    angle_delta = abs(angle - float(smoothed_angles[i - 1]))
                    if angle_delta < 2.0:
                        stability_count += 1
                    else:
                        stability_count = max(0, stability_count - 1)

                # 안정 프레임 수 도달 → 1회 투구 완료
                if stability_count >= self.idle_stability_frames or \
                   frames_in_state > int(1.5 * self.fps):
                    # angle_range = cocking 진입 최대 각도 - 최저 각도
                    angle_range = cocking_entry_angle - min_angle
                    if angle_range >= self.min_cocking_angle:
                        cycles.append({
                            "start": cycle_start,
                            "end": i,
                            "min_angle_frame": min_angle_frame,
                            "min_angle": min_angle,
                            "angle_range": angle_range,
                        })
                        last_cycle_end = i
                    else:
                        print(f"    ⚠ 무효 사이클 기각 (각도 변화 {angle_range:.1f}° < "
                              f"{self.min_cocking_angle}°)")

                    state = _FSMState.IDLE
                    state_entry_frame = i

        # 마지막 미완료 사이클 처리
        if state in (_FSMState.RELEASING, _FSMState.FOLLOW_THROUGH):
            angle_range = cocking_entry_angle - min_angle
            if angle_range >= self.min_cocking_angle:
                cycles.append({
                    "start": cycle_start,
                    "end": n - 1,
                    "min_angle_frame": min_angle_frame,
                    "min_angle": min_angle,
                    "angle_range": angle_range,
                })

        self._last_fsm_states = fsm_states
        return cycles

    # ─── Segment Construction ────────────────────────────────────────────────

    def _cyclesToSegments(
        self,
        frames: list[FrameData],
        cycles: list[dict],
    ) -> list[list[FrameData]]:
        """FSM 사이클 경계를 프레임 세그먼트로 변환합니다.

        각 사이클의 시작/끝에 패딩을 추가하여
        테이크백 준비 ~ 팔로스루 완료까지 충분한 프레임을 확보합니다.

        Args:
            frames: 전체 FrameData 리스트.
            cycles: FSM이 감지한 사이클 경계 리스트.

        Returns:
            세그먼트 리스트.
        """
        n = len(frames)
        segments: list[list[FrameData]] = []

        for idx, cycle in enumerate(cycles):
            start = max(0, cycle["start"] - self.segment_pad)
            end = min(n - 1, cycle["end"] + self.segment_pad)

            # 인접 사이클과 패딩이 겹치지 않도록 중간점으로 클리핑
            if idx > 0:
                midpoint = (cycles[idx - 1]["end"] + cycle["start"]) // 2
                start = max(start, midpoint + 1)
            if idx < len(cycles) - 1:
                midpoint = (cycle["end"] + cycles[idx + 1]["start"]) // 2
                end = min(end, midpoint)

            segments.append(frames[start:end + 1])

        return segments

    def _filterInvalidSegments(
        self,
        segments: list[list[FrameData]],
        all_frames: list[FrameData],
        throwing_side: str,
    ) -> list[list[FrameData]]:
        """무효 세그먼트를 필터링합니다.

        조건:
        1. 최소 프레임 수 미달
        2. 손목 이동 변위 부족 (단순 자세 교정)

        Args:
            segments: 후보 세그먼트 리스트.
            all_frames: 전체 프레임 (참조용).
            throwing_side: 투구 팔 방향.

        Returns:
            필터링된 세그먼트 리스트.
        """
        valid: list[list[FrameData]] = []

        for seg in segments:
            # 조건 1: 최소 프레임 수
            if len(seg) < self.min_segment_frames:
                print(f"    ⚠ 세그먼트 기각 (프레임 부족: {len(seg)})")
                continue

            # 조건 2: 손목 이동 변위 확인
            wrist_key = f"{throwing_side}_wrist"
            wrist_coords = []
            for f in seg:
                if f.keypoints:
                    w = f.keypoints.get(wrist_key)
                    if w:
                        wrist_coords.append(w[:2])

            if len(wrist_coords) >= 3:
                coords = np.array(wrist_coords)
                max_disp = float(np.max(np.linalg.norm(
                    coords - coords[0], axis=1
                )))
                if max_disp < 0.05:  # 정규화 좌표에서 5% 미만 이동
                    print(f"    ⚠ 세그먼트 기각 (변위 부족: {max_disp:.3f})")
                    continue

            valid.append(seg)

        return valid

    # ─── Core Algorithms ────────────────────────────────────────────────────

    @staticmethod
    def _extractRawJointCoords(
        frames: list[FrameData],
        joint_name: str,
    ) -> np.ndarray | None:
        """원시 keypoints에서 특정 관절 좌표를 추출합니다.

        MediaPipe 0~1 좌표를 직접 사용합니다.
        누락된 프레임은 직전 값으로 채웁니다 (forward fill).

        Args:
            frames: FrameData 리스트.
            joint_name: 찾을 관절의 키 이름 (예: 'left_wrist')

        Returns:
            shape (N, 3) numpy 배열. 좌표가 전혀 없으면 None.
        """
        n = len(frames)
        coords = np.zeros((n, 3))
        last_valid = None

        for i, f in enumerate(frames):
            if f.keypoints:
                w = f.keypoints.get(joint_name)
                if w is not None:
                    last_valid = np.array(w[:3], dtype=np.float64)
            if last_valid is not None:
                coords[i] = last_valid

        # 유효 좌표가 하나도 없으면 None 반환
        if last_valid is None:
            return None

        return coords

    @staticmethod
    def _computeElbowAngles(
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray,
    ) -> np.ndarray:
        """어깨-팔꿈치-손목 각도를 계산합니다.

        각도는 카메라 거리에 불변합니다 (벡터 내적 기반).
        이것이 다른 앱이 손목 각도로 투구를 판정하는 핵심 원리입니다.

        팔이 완전히 펴지면 ~180°, 접으면 ~30~60°.
        다트 투구 사이클: ~160° → ~60° → ~170° (접기 → 펴기)

        Args:
            shoulder: shape (N, 3) 어깨 좌표.
            elbow: shape (N, 3) 팔꿈치 좌표.
            wrist: shape (N, 3) 손목 좌표.

        Returns:
            shape (N,) 각도 배열 (도 단위).
        """
        # 팔꿈치에서 어깨/손목 방향 벡터
        vec_upper = shoulder - elbow   # 상완 벡터
        vec_forearm = wrist - elbow    # 전완 벡터

        # 벡터 크기
        norm_upper = np.linalg.norm(vec_upper, axis=1)
        norm_forearm = np.linalg.norm(vec_forearm, axis=1)

        # 유효한 벡터만 각도 계산 (길이가 0에 가까운 경우 방어)
        valid = (norm_upper > 1e-6) & (norm_forearm > 1e-6)
        angles = np.full(len(elbow), 180.0)  # 기본값: 팔 완전 펴짐

        # 내적 → 각도 변환
        dot_product = np.sum(vec_upper[valid] * vec_forearm[valid], axis=1)
        cos_theta = np.clip(
            dot_product / (norm_upper[valid] * norm_forearm[valid]),
            -1.0, 1.0
        )
        angles[valid] = np.degrees(np.arccos(cos_theta))

        return angles

    def _gaussianSmooth(self, signal: np.ndarray, sigma_seconds: float) -> np.ndarray:
        """1D 신호에 가우시안 스무딩을 적용합니다.

        scipy 없이 numpy 컨볼루션으로 구현합니다.
        iOS Swift(vDSP) 포팅을 위해 커널 크기를 홀수로 유지합니다.

        Args:
            signal: 1D numpy 배열.
            sigma_seconds: 가우시안 표준편차 (초 단위).

        Returns:
            스무딩된 1D numpy 배열 (같은 길이).
        """
        # FPS 기반 동적 프레임 환산
        sigma_frames = max(1.0, sigma_seconds * self.fps)
        # 커널 크기: 홀수 보장 (vDSP.convolve 1:1 매칭 요건)
        kernel_size = int(6 * sigma_frames) | 1
        half = kernel_size // 2

        # 가우시안 커널 생성
        x = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (x / sigma_frames) ** 2)
        kernel /= kernel.sum()  # 정규화

        # 'same' 모드로 컨볼루션 — 길이 유지
        return np.convolve(signal, kernel, mode="same")

    def _mergeCloseSegments(
        self,
        segments: list[list[FrameData]],
    ) -> list[list[FrameData]]:
        """인접한 세그먼트 간격이 너무 가까우면 하나로 병합합니다.

        Args:
            segments: 세그먼트 리스트.

        Returns:
            병합된 세그먼트 리스트.
        """
        if len(segments) < 2:
            return segments

        merged: list[list[FrameData]] = []
        current = list(segments[0])

        for next_seg in segments[1:]:
            gap = next_seg[0].frame_index - current[-1].frame_index
            if gap <= self.merge_gap_frames:
                print(f"  ℹ 세그먼트 병합 (간격: {gap}f)")
                current.extend(next_seg)
            else:
                merged.append(current)
                current = list(next_seg)

        merged.append(current)
        return merged

    # ─── Debug Accessors ─────────────────────────────────────────────────────

    def _clearDebugData(self) -> None:
        """디버그 데이터를 초기화합니다."""
        self._last_elbow_angles = np.array([])
        self._last_smoothed_angles = np.array([])
        self._last_fsm_states = []
        self._last_cycle_boundaries = []
        self._last_velocity = np.array([])
        self._last_peaks = []

    def getDebugData(self) -> dict:
        """디버그 시각화에 필요한 데이터를 반환합니다.

        Returns:
            딕셔너리:
            - 'elbow_angles': 원시 팔꿈치 각도 시계열
            - 'smoothed_angles': 스무딩된 각도
            - 'fsm_states': 프레임별 FSM 상태 문자열
            - 'cycle_boundaries': 사이클 경계 정보
        """
        return {
            "elbow_angles": self._last_elbow_angles,
            "smoothed_angles": self._last_smoothed_angles,
            "fsm_states": self._last_fsm_states,
            "cycle_boundaries": self._last_cycle_boundaries,
        }
