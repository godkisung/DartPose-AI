"""포즈 정규화 모듈.

핸드헬드 촬영에 의한 카메라 흔들림 보정 및 몸통 크기 기반 좌표 정규화를
수행합니다. 측면/측후면/측전면 등 다양한 촬영 각도에서도 동일한
분석 신호를 추출하기 위해 원시 좌표 대신 상대 좌표계로 변환합니다.

iOS Swift 포팅 시 이 모듈의 로직은 Accelerate(vDSP) 로 1:1 구현 가능합니다.
"""

import numpy as np
from src.models import FrameData


# ─── 관절 이름 상수 ──────────────────────────────────────────────────────────

# 핵심 포즈 관절 (MediaPipe Pose 랜드마크 인덱스 기준)
_UPPER_BODY_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
]

# 손가락 관절 (MediaPipe Hands — 선택적)
_FINGER_JOINTS = [
    "left_thumb_tip", "right_thumb_tip",
    "left_index_tip", "right_index_tip",
    "left_middle_tip", "right_middle_tip",
]


class PoseNormalizer:
    """관절 좌표 시계열을 정규화하는 클래스.

    1. 어깨 중점(mid-shoulder) 기준 상대 좌표 변환 → 카메라 흔들림 제거
    2. 어깨-엉덩이 거리 기준 스케일 정규화 → 촬영 거리 차이 제거
    3. 가우시안 스무딩 → 잔여 노이즈 제거

    사용 예시:
        normalizer = PoseNormalizer()
        normalized_data = normalizer.normalize(frames, throwing_side="right")
    """

    def __init__(self, smoothing_window: int = 5):
        """초기화.

        Args:
            smoothing_window: 시간축 이동평균 스무딩 윈도우 크기.
                              클수록 떨림이 제거되지만 시간 해상도가 낮아짐.
        """
        self.smoothing_window = smoothing_window

    # ─── Public API ─────────────────────────────────────────────────────────

    def normalize(
        self,
        frames: list[FrameData],
        throwing_side: str,
    ) -> dict[str, np.ndarray]:
        """프레임 리스트를 정규화된 관절 좌표 딕셔너리로 변환합니다.

        Args:
            frames: FrameData 리스트.
            throwing_side: 투구 팔 방향 ('left' 또는 'right').

        Returns:
            관절 이름 → shape (N, 3) numpy 배열 딕셔너리.
            N은 프레임 수, 3은 (x, y, z) 정규화 좌표.
        """
        all_joints = _UPPER_BODY_JOINTS + _FINGER_JOINTS

        # 1단계: 원시 좌표 추출 (없는 값은 이전 프레임으로 채우기)
        raw = self._extractRawCoordinates(frames, all_joints)

        # 2단계: 핸드헬드 흔들림 보정 (어깨 중점 기준 상대 좌표)
        camera_corrected = self._correctCameraMotion(raw)

        # 3단계: 몸통 크기 기반 스케일 정규화
        scale_normalized = self._normalizeByTorsoLength(
            camera_corrected, throwing_side
        )

        # 4단계: 시간축 이동평균 스무딩
        smoothed = self._applyTemporalSmoothing(scale_normalized)

        return smoothed

    def extractAngleSeries(
        self,
        normalized_data: dict[str, np.ndarray],
        side: str,
    ) -> dict[str, np.ndarray]:
        """정규화된 좌표에서 관절 각도 시계열을 추출합니다.

        관절 각도는 촬영 각도에 상대적으로 불변하므로,
        측면/측후면/측전면 영상에서도 일관된 신호를 제공합니다.

        Args:
            normalized_data: normalize()의 출력.
            side: 투구 팔 방향 ('left' 또는 'right').

        Returns:
            'elbow_angle': 어깨-팔꿈치-손목 각도 시계열 (도, shape: N)
            'wrist_angle': 팔꿈치-손목-손가락 각도 시계열 (도, shape: N) — 손가락 가용 시
        """
        n_frames = len(next(iter(normalized_data.values())))

        shoulder = normalized_data.get(f"{side}_shoulder", np.zeros((n_frames, 3)))
        elbow    = normalized_data.get(f"{side}_elbow",    np.zeros((n_frames, 3)))
        wrist    = normalized_data.get(f"{side}_wrist",    np.zeros((n_frames, 3)))
        index    = normalized_data.get(f"{side}_index_tip", None)

        # 어깨-팔꿈치-손목 각도 시계열
        elbow_angles = np.array([
            self._angle3d(shoulder[i], elbow[i], wrist[i])
            for i in range(n_frames)
        ])

        result = {"elbow_angle": elbow_angles}

        # 손가락(index_tip)이 유효한 경우 손목 각도 계산
        if index is not None:
            wrist_angles = np.array([
                self._angle3d(elbow[i], wrist[i], index[i])
                for i in range(n_frames)
            ])
            result["wrist_angle"] = wrist_angles

        return result

    # ─── Private Methods ────────────────────────────────────────────────────

    def _extractRawCoordinates(
        self,
        frames: list[FrameData],
        joints: list[str],
    ) -> dict[str, np.ndarray]:
        """프레임 리스트에서 관절별 원시 좌표 배열을 추출합니다.

        누락된 값(None)은 직전 유효값으로 채웁니다(forward fill).
        모두 누락이면 [0,0,0]으로 처리합니다.

        Args:
            frames: FrameData 리스트.
            joints: 추출할 관절 이름 리스트.

        Returns:
            관절 이름 → shape (N, 3) numpy 배열 딕셔너리.
        """
        n = len(frames)
        result: dict[str, np.ndarray] = {}

        for joint in joints:
            coords = np.zeros((n, 3), dtype=np.float32)
            last_valid = np.zeros(3, dtype=np.float32)

            for i, frame in enumerate(frames):
                if frame.keypoints:
                    val = frame.keypoints.get(joint)
                    if val is not None:
                        last_valid = np.array(val, dtype=np.float32)
                coords[i] = last_valid

            result[joint] = coords

        return result

    def _correctCameraMotion(
        self,
        raw: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """핸드헬드 카메라 흔들림을 제거합니다.

        어깨 중점(mid-shoulder)의 이동 궤적을 계산하고,
        모든 관절에서 이 이동량을 차감합니다.
        이렇게 하면 사람의 신체가 화면에 고정된 것처럼 좌표가 변환됩니다.

        Args:
            raw: 원시 관절 좌표 딕셔너리.

        Returns:
            카메라 움직임이 제거된 좌표 딕셔너리.
        """
        # 어깨 중점 계산 (카메라 위치의 프록시)
        l_shoulder = raw.get("left_shoulder", np.zeros_like(raw.get("right_shoulder", np.zeros((1, 3)))))
        r_shoulder = raw.get("right_shoulder", l_shoulder)
        mid_shoulder = (l_shoulder + r_shoulder) / 2.0  # shape: (N, 3)

        corrected = {}
        for joint, coords in raw.items():
            # 손가락은 어깨 중점 보정의 영향을 받으면 오히려 부정확해질 수 있음
            # (손가락은 원래 상대 좌표가 더 신뢰도 높음)
            corrected[joint] = coords - mid_shoulder

        return corrected

    def _normalizeByTorsoLength(
        self,
        data: dict[str, np.ndarray],
        side: str,
    ) -> dict[str, np.ndarray]:
        """몸통 길이(어깨-엉덩이 거리)로 좌표를 정규화합니다.

        촬영 거리(카메라에서 피사체까지 거리)에 따라 관절 좌표의
        절대값이 달라지는 문제를 해결합니다.

        Args:
            data: 카메라 보정된 좌표 딕셔너리.
            side: 투구 팔 방향.

        Returns:
            스케일 정규화된 좌표 딕셔너리.
        """
        shoulder = data.get(f"{side}_shoulder", np.zeros_like(data.get("left_shoulder")))
        hip = data.get(f"{side}_hip", data.get("left_hip", np.zeros_like(shoulder)))

        # 어깨-엉덩이 거리 (프레임별 계산 후 중앙값 사용)
        torso_lengths = np.linalg.norm(shoulder - hip, axis=1)  # shape: (N,)
        median_torso = np.median(torso_lengths)

        # 유효하지 않은 스케일 방지 (너무 작으면 정규화 안함)
        scale = median_torso if median_torso > 0.01 else 1.0

        normalized = {}
        for joint, coords in data.items():
            normalized[joint] = coords / scale

        return normalized

    def _applyTemporalSmoothing(
        self,
        data: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """시간축 이동평균(moving average) 스무딩을 적용합니다.

        잔여 떨림 노이즈를 제거합니다.
        'same' 모드로 경계 처리하여 길이를 그대로 유지합니다.

        Args:
            data: 정규화된 좌표 딕셔너리.

        Returns:
            스무딩된 좌표 딕셔너리.
        """
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed = {}

        for joint, coords in data.items():
            # 각 축(x, y, z)에 독립적으로 이동평균 적용
            out = np.zeros_like(coords)
            for axis in range(coords.shape[1]):
                out[:, axis] = np.convolve(coords[:, axis], kernel, mode="same")
            smoothed[joint] = out

        return smoothed

    # ─── Static Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _angle3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """p2를 꼭짓점으로 하는 3점 각도를 도(degree) 단위로 반환합니다.

        Args:
            p1, p2, p3: 3D 또는 2D 좌표 numpy 배열.

        Returns:
            각도 (도). 유효하지 않은 입력이면 0.0 반환.
        """
        v1 = p1 - p2
        v2 = p3 - p2
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))
