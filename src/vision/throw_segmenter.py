"""투구 세분화 모듈.

연속 영상에서 개별 투구 구간을 자동으로 분리합니다.
scipy를 사용하지 않고 numpy 만으로 핵심 알고리즘을 구현하여
iOS Swift(Accelerate/vDSP)로의 1:1 포팅을 용이하게 합니다.

핵심 알고리즘:
1. 손목 2D 속도(카메라 보정 후) 계산
2. 가우시안 스무딩으로 노이즈 제거
3. 자체 구현 peak detection: prominence + min_distance 조건 적용
4. 각 피크 주변으로 세그먼트 확장
"""

import numpy as np
from src.models import FrameData
from src.config import (
    SEGMENTER_SMOOTHING_SIGMA,
    SEGMENTER_MIN_PEAK_PROMINENCE,
    SEGMENTER_MIN_PEAK_DISTANCE,
    SEGMENTER_SEGMENT_EXPAND_FRAMES,
    SEGMENTER_MIN_SEGMENT_FRAMES,
    SEGMENTER_MERGE_GAP_FRAMES,
)


class ThrowSegmenter:
    """연속 영상에서 개별 투구 구간을 분리하는 클래스.

    기존 방식의 문제점:
    - 92 percentile 임계값: 영상마다 임계값이 달라 불안정
    - 유휴 카운터 방식: 느린/빠른 투구자에 동일하게 적용 불가

    개선된 방식:
    - 손목 속도 피크(peak)를 감지하여 투구의 '최고 속도 순간'을 찾음
    - prominence 조건: 주변 값 대비 얼마나 두드러지는지 → 노이즈 억제
    - min_distance 조건: 피크 간 최소 간격 → 같은 투구를 두 번 세지 않음
    - 피크 중심으로 ±expand 프레임을 세그먼트로 확장
    """

    def __init__(
        self,
        fps: float = 30.0,
        smoothing_sigma: float = SEGMENTER_SMOOTHING_SIGMA,
        min_peak_prominence: float = SEGMENTER_MIN_PEAK_PROMINENCE,
        min_peak_distance_s: float = SEGMENTER_MIN_PEAK_DISTANCE,
        segment_expand_s: float = SEGMENTER_SEGMENT_EXPAND_FRAMES,
        min_segment_frames: int = SEGMENTER_MIN_SEGMENT_FRAMES,
        merge_gap_frames: int = SEGMENTER_MERGE_GAP_FRAMES,
    ):
        """초기화.

        Args:
            fps: 영상 프레임레이트.
            smoothing_sigma: 가우시안 스무딩 표준편차 (초 단위).
                            클수록 노이즈 제거 강하지만 시간 해상도 감소.
            min_peak_prominence: 피크 돌출도 최소값 (속도 단위).
                                 주변 값의 이 비율 이상이어야 유효 피크로 판정.
            min_peak_distance_s: 연속 피크 간 최소 시간 간격 (초).
                                 투구 주기의 최솟값으로 설정.
            segment_expand_s: 피크 전후로 세그먼트를 확장하는 시간 (초).
            min_segment_frames: 유효 세그먼트의 최소 프레임 수.
            merge_gap_frames: 이 프레임 수 이하의 간격은 하나로 병합.
        """
        self.fps = fps
        self.smoothing_sigma = smoothing_sigma
        self.min_peak_prominence = min_peak_prominence
        # 초 단위를 프레임 단위로 변환
        self.min_peak_distance = max(1, int(min_peak_distance_s * fps))
        self.segment_expand = max(1, int(segment_expand_s * fps))
        self.min_segment_frames = min_segment_frames
        self.merge_gap_frames = merge_gap_frames

    # ─── Public API ─────────────────────────────────────────────────────────

    def segment(
        self,
        frames: list[FrameData],
        normalized_data: dict[str, np.ndarray],
        throwing_side: str,
    ) -> list[list[FrameData]]:
        """투구 세그먼트를 분리합니다.

        원시 keypoints의 손목 좌표에서 속도를 계산합니다.
        정규화 좌표는 어깨 중점 기준 상대 좌표이므로,
        실제 손목 이동량이 상쇄되어 투구 피크를 놓칠 수 있습니다.

        Args:
            frames: keypoints가 있는 FrameData 리스트.
            normalized_data: PoseNormalizer.normalize()의 출력 (API 호환용).
            throwing_side: 투구 팔 방향 ('left' 또는 'right').

        Returns:
            투구 세그먼트 리스트. 각 원소는 [FrameData, ...] 리스트.
        """
        if len(frames) < self.min_segment_frames:
            self._last_velocity = np.array([])
            self._last_peaks = []
            return [frames] if frames else []

        # 원시 좌표 추출 (어깨, 팔꿈치, 손목)
        shoulder_coords = self._extractRawJointCoords(frames, f"{throwing_side}_shoulder")
        elbow_coords = self._extractRawJointCoords(frames, f"{throwing_side}_elbow")
        wrist_coords = self._extractRawJointCoords(frames, f"{throwing_side}_wrist")

        if wrist_coords is None or elbow_coords is None or shoulder_coords is None or len(wrist_coords) < 2:
            self._last_velocity = np.array([])
            self._last_peaks = []
            return [frames]

        # 1. 손목 2D 속도(프레임 간 변위 크기) 계산 및 스무딩
        wrist_velocity = self._computeWristVelocity(wrist_coords)
        smoothed_wrist = self._gaussianSmooth(wrist_velocity, self.smoothing_sigma)

        # 2. 팔꿈치 각도 및 각속도 계산 (3D)
        vec_a = shoulder_coords - elbow_coords
        vec_b = wrist_coords - elbow_coords
        norm_a = np.linalg.norm(vec_a, axis=1)
        norm_b = np.linalg.norm(vec_b, axis=1)
        
        valid = (norm_a > 1e-6) & (norm_b > 1e-6)
        angles = np.zeros(len(frames))
        angles[~valid] = 180.0
        dot_product = np.sum(vec_a[valid] * vec_b[valid], axis=1)
        cos_theta = np.clip(dot_product / (norm_a[valid] * norm_b[valid]), -1.0, 1.0)
        angles[valid] = np.degrees(np.arccos(cos_theta))
        
        # 프레임 간 각도 변화율 (각속도 대용)
        angular_vel = np.concatenate([[0.0], np.diff(angles)])
        smoothed_angular = self._gaussianSmooth(angular_vel, self.smoothing_sigma)
        
        # 3. 신호 정규화 및 융합 (Multi-Signal Fusion)
        # left arm처럼 한쪽 신호가 약해도 보완할 수 있도록 융합.
        # 비디오별 최대값 정규화는 노이즈를 증폭시키므로, 물리적 스케일에 맞춰 고정 가중치 융합
        # (손목 속도: 0.01~0.08, 각속도: 10~40도/프레임 → 각속도에 0.002 곱해 스케일 맞춤)
        # 릴리즈 시 팔꿈치가 '펴지는' 양수 각속도만 사용.
        positive_angular = np.maximum(smoothed_angular, 0.0)
        fused_signal = smoothed_wrist + (positive_angular * 0.002)

        # 4. 피크 감지 — 융합 신호의 적응적 prominence + distance 조건
        # 고정 prominence는 카메라 거리/위치에 따라 불안정하므로,
        # 신호의 동적 범위(max - median)의 비율로 자동 결정
        adaptive_prominence = self._computeAdaptiveProminence(fused_signal)
        effective_prominence = max(adaptive_prominence, self.min_peak_prominence)

        peak_indices = self._findPeaks(
            fused_signal,
            min_prominence=effective_prominence,
            min_distance=self.min_peak_distance,
        )

        # ※ 디버그 시각화용 융합 신호 저장
        self._last_velocity = fused_signal
        self._last_peaks = list(peak_indices)

        # 피크가 없으면 전체를 하나의 투구로 반환 (fallback)
        if len(peak_indices) == 0:
            print("  ⚠ 피크 감지 실패 — 전체를 1개 투구로 처리")
            return [frames]

        print(f"  ℹ 감지된 속도 피크: {len(peak_indices)}개 "
              f"(prominence={effective_prominence:.4f}) "
              f"at frames {[frames[i].frame_index for i in peak_indices]}")

        # 4. 피크 주변으로 세그먼트 확장
        raw_segments = self._expandPeaksToSegments(frames, peak_indices)

        # 5. 너무 짧은 세그먼트 제거
        filtered = [s for s in raw_segments if len(s) >= self.min_segment_frames]

        # 6. 가까운 세그먼트 병합
        merged = self._mergeCloseSegments(filtered)

        return merged if merged else [frames]

    # ─── Core Algorithms ────────────────────────────────────────────────────

    @staticmethod
    def _extractRawJointCoords(
        frames: list[FrameData],
        joint_name: str,
    ) -> np.ndarray | None:
        """원시 keypoints에서 특정 관절 좌표를 추출합니다.

        정규화 좌표는 어깨 중점 기준 상대 좌표이므로,
        카메라 움직임과 함께 관절의 실제 이동량이 상쇄됩니다.
        원시 좌표(MediaPipe 0~1 범위)를 직접 사용하면
        높이/위치에 무관하게 안정적으로 투구 속도를 측정할 수 있습니다.

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
                    last_valid = np.array(w, dtype=np.float64)
            if last_valid is not None:
                coords[i] = last_valid

        # 유효 좌표가 하나도 없으면 None 반환
        if last_valid is None:
            return None

        return coords

    @staticmethod
    def _computeAdaptiveProminence(smoothed_velocity: np.ndarray) -> float:
        """신호의 동적 범위를 기반으로 적응적 prominence 임계값을 계산합니다.

        고정 prominence는 카메라 거리, 촬영 각도, 해상도에 따라 불안정합니다.
        대신 신호의 (max - median)에 비례하는 임계값을 사용하면,
        영상 조건에 무관하게 안정적으로 투구 피크만 감지할 수 있습니다.

        비율은 0.25 (25%)를 기본값으로 사용합니다.
        → 투구 피크는 보통 median 대비 3~10배 높으므로 25%면 충분히 분별 가능.
        → 자세 조정이나 팔 흔들기 등의 잔잔한 움직임은 25% 이하이므로 필터링됨.

        Args:
            smoothed_velocity: 스무딩된 손목 속도 시계열.

        Returns:
            적응적 prominence 임계값.
        """
        if len(smoothed_velocity) < 3:
            return 0.010  # fallback

        signal_max = float(np.max(smoothed_velocity))
        signal_median = float(np.median(smoothed_velocity))
        dynamic_range = signal_max - signal_median

        # 동적 범위의 25%를 prominence로 사용
        adaptive = dynamic_range * 0.25

        # 최소 하한 (노이즈가 너무 작은 경우 방어)
        return max(adaptive, 0.002)

    def _computeWristVelocity(self, wrist_coords: np.ndarray) -> np.ndarray:
        """손목 좌표 시계열에서 프레임 간 속도(변위 크기)를 계산합니다.

        X, Y 축만 사용합니다 (Z는 깊이로 MediaPipe 추정값이 부정확).

        Args:
            wrist_coords: shape (N, 3) 손목 3D 좌표 배열.

        Returns:
            shape (N,) 속도 배열. 첫 값은 0으로 패딩됨.
        """
        # 프레임 간 XY 변위 계산
        displacements = np.diff(wrist_coords[:, :2], axis=0)  # shape: (N-1, 2)
        velocity = np.linalg.norm(displacements, axis=1)       # shape: (N-1,)

        # 첫 프레임에 0 패딩 (원래 프레임 수 유지)
        return np.concatenate([[0.0], velocity])

    def _gaussianSmooth(self, signal: np.ndarray, sigma_seconds: float) -> np.ndarray:
        """1D 신호에 가우시안 스무딩을 적용합니다.

        scipy 없이 numpy 컨볼루션으로 구현합니다.
        커널 크기는 sigma의 6배(±3σ 범위)로 설정하고,
        vDSP 포팅을 위해 반드시 홀수로 유지합니다.

        σ는 초 단위로 입력받으므로 FPS에 따라 자동으로
        프레임 단위로 변환됩니다:
        - 30fps, σ=0.06s → σ_frames=1.8 → kernel=11
        - 60fps, σ=0.06s → σ_frames=3.6 → kernel=23

        Args:
            signal: 1D numpy 배열.
            sigma_seconds: 가우시안 표준편차 (초 단위).

        Returns:
            스무딩된 1D numpy 배열 (같은 길이).
        """
        # FPS 기반 동적 프레임 환산 (검토 의견: σ를 FPS에 따라 동적 설정)
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

    def _findPeaks(
        self,
        signal: np.ndarray,
        min_prominence: float,
        min_distance: int,
    ) -> np.ndarray:
        """신호에서 로컬 최대값(피크)을 감지합니다.

        scipy.signal.find_peaks 와 동일한 로직을 numpy로 직접 구현합니다.
        iOS Swift 포팅 시 이 함수를 vDSP 기반으로 대체할 수 있습니다.

        알고리즘:
        1. 로컬 최대값 후보 탐색 (양쪽 이웃보다 큰 값)
        2. prominence 필터: 피크 값이 주변 골짜기보다 min_prominence 이상 높아야 함
        3. min_distance 필터: 피크 간 최소 거리 유지 (가까운 피크 중 큰 것만 남김)

        Args:
            signal: 1D numpy 배열.
            min_prominence: 최소 돌출도 (절대값 기준).
            min_distance: 피크 간 최소 인덱스 간격.

        Returns:
            피크 인덱스 배열 (정렬됨).
        """
        n = len(signal)
        if n < 3:
            return np.array([], dtype=int)

        # 1. 로컬 최대값 후보: 양쪽 이웃보다 크거나 같은 점
        candidates = []
        for i in range(1, n - 1):
            if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
                # 단, 평탄한 구간의 첫 점만 택함
                if signal[i] > signal[i - 1] or signal[i] > signal[i + 1]:
                    candidates.append(i)

        if not candidates:
            return np.array([], dtype=int)

        # 2. prominence 필터 (scipy.signal.find_peaks 정확한 정의)
        # prominence = 피크 값 - contour_height
        # contour_height = 양쪽으로 내려가서 '더 높은 피크를 만나기 전까지'
        #                  구간에서의 최저점 중 큰 쪽 값
        # (검토 의견: 전체 구간 min이 아닌, 인접 골짜기 기준으로 계산해야
        #  의미 있는 투구 가속만 걸러낼 수 있음)
        prominent = []
        for idx in candidates:
            # 왼쪽: idx에서 시작하여 왼쪽으로 내려가다 더 높은 피크를 만나면 정지
            left_min = signal[idx]
            for j in range(idx - 1, -1, -1):
                left_min = min(left_min, signal[j])
                if signal[j] > signal[idx]:  # 더 높은 피크 발견 → 정지
                    break

            # 오른쪽: idx에서 시작하여 오른쪽으로 내려가다 더 높은 피크를 만나면 정지
            right_min = signal[idx]
            for j in range(idx + 1, n):
                right_min = min(right_min, signal[j])
                if signal[j] > signal[idx]:  # 더 높은 피크 발견 → 정지
                    break

            prominence = signal[idx] - max(left_min, right_min)
            if prominence >= min_prominence:
                prominent.append(idx)

        if not prominent:
            # prominence 조건을 만족하는 피크가 없으면 가장 큰 피크 1개만 반환
            best = max(candidates, key=lambda i: signal[i])
            return np.array([best], dtype=int)

        # 3. min_distance 필터 (작은 피크 제거)
        # 값이 큰 피크를 우선적으로 유지하고, 그 주변 min_distance 내의 피크를 제거
        # 피크를 신호 강도 내림차순으로 정렬
        prominent_sorted = sorted(prominent, key=lambda i: signal[i], reverse=True)
        kept = []
        suppressed = set()

        for idx in prominent_sorted:
            if idx in suppressed:
                continue
            kept.append(idx)
            # 이 피크 주변 min_distance 내의 다른 피크를 억제
            for other in prominent_sorted:
                if abs(other - idx) < min_distance and other != idx:
                    suppressed.add(other)

        return np.array(sorted(kept), dtype=int)

    def _expandPeaksToSegments(
        self,
        frames: list[FrameData],
        peak_indices: np.ndarray,
    ) -> list[list[FrameData]]:
        """각 피크 주변으로 세그먼트를 확장합니다.

        각 피크(최고 속도 프레임)를 중심으로 ±expand_frames 범위의
        프레임들을 하나의 투구 세그먼트로 묶습니다.

        Args:
            frames: FrameData 리스트.
            peak_indices: 피크 인덱스 배열.

        Returns:
            세그먼트 리스트.
        """
        n = len(frames)
        segments = []

        for peak_idx in peak_indices:
            start = max(0, peak_idx - self.segment_expand)
            end = min(n - 1, peak_idx + self.segment_expand)
            segments.append(frames[start:end + 1])

        return segments

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

        merged = []
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
