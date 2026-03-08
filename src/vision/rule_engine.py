import numpy as np
import math

class PoseRuleEngine:
    def __init__(self, fps=30):
        self.history = []
        self.fps = fps
        # Keypoints definition expects dict like: { 'left_shoulder': [x, y, (z)], ... }
        
    def feed_frame(self, frame_index, timestamp_ms, keypoints):
        """
        Feeds a single frame's extracted keypoints into the engine.
        keypoints is a dictionary mapping joint names to their coordinates.
        Coordinates should ideally be normalized or in consistent pixel space.
        """
        self.history.append({
            'frame': frame_index,
            'time': timestamp_ms,
            'keypoints': keypoints
        })

    def get_joint(self, index, joint_name):
        """Helper to get a joint coordinate at a specific history index. Returns numpy array."""
        kp = self.history[index]['keypoints'].get(joint_name)
        return np.array(kp) if kp is not None else None

    def calculate_angle_3d(self, p1, p2, p3):
        """
        Calculate 3D angle at p2 formed by lines p1-p2 and p3-p2.
        Input: numpy arrays [x, y, z].
        Returns angle in degrees (0 to 180).
        """
        if p1 is None or p2 is None or p3 is None:
            return 0.0
            
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def _detect_throwing_arm(self):
        """
        Automatically detects whether the player is left or right handed based on
        which wrist has the most movement along the X and Y axes throughout the clip.
        Returns 'right' or 'left'.
        """
        right_wrist_positions = []
        left_wrist_positions = []
        
        for record in self.history:
            rw = record['keypoints'].get('right_wrist')
            lw = record['keypoints'].get('left_wrist')
            if rw is not None: right_wrist_positions.append(rw)
            if lw is not None: left_wrist_positions.append(lw)
            
        if not right_wrist_positions or not left_wrist_positions:
            return 'right' # Default fallback
            
        rw_var = np.var([p[0] for p in right_wrist_positions]) + np.var([p[1] for p in right_wrist_positions])
        lw_var = np.var([p[0] for p in left_wrist_positions]) + np.var([p[1] for p in left_wrist_positions])
        
        return 'right' if rw_var > lw_var else 'left'

    def detect_phases(self, throwing_side):
        """
        Phase Detection based on wrist movement along the X-axis.
        """
        if len(self.history) < 10:
            return None
            
        wrist_key = f'{throwing_side}_wrist'
        x_positions = []
        
        for rec in self.history:
            wrist = rec['keypoints'].get(wrist_key)
            if wrist is not None:
                x_positions.append(wrist[0])
            else:
                x_positions.append(x_positions[-1] if x_positions else 0.0)
                
        # Smooth the trajectory to ignore jitter
        window_size = 5
        smoothed_x = np.convolve(x_positions, np.ones(window_size)/window_size, mode='same')
        
        # Determine throwing direction based on overall start to end displacement
        # Or more robustly, find the global min and max. 
        # The takeback is the furthest point backwards, release is furthest forwards.
        # Assuming camera is generally in front of the board/player, 
        # throwing right means X goes from small to large.
        
        takeback_max_idx = 0
        release_idx = 0
        
        diff = smoothed_x[-1] - smoothed_x[0]
        if diff > 0:
            # Throwing Right: Takeback is Minimum X, Release is Maximum X
            takeback_max_idx = np.argmin(smoothed_x)
            release_idx = np.argmax(smoothed_x[takeback_max_idx:]) + takeback_max_idx
        else:
            # Throwing Left: Takeback is Maximum X, Release is Minimum X
            takeback_max_idx = np.argmax(smoothed_x)
            release_idx = np.argmin(smoothed_x[takeback_max_idx:]) + takeback_max_idx

        # Address is before takeback where movement is minimal. Use 0 for now.
        # Follow through is after release where movement stops. Use end for now.
        n = len(self.history)
        
        # Fallback if detection fails
        if takeback_max_idx == release_idx or takeback_max_idx == 0:
             takeback_max_idx = int(n * 0.4)
             release_idx = int(n * 0.6)
             
        takeback_start_idx = max(0, takeback_max_idx - int(n*0.1)) # Rough approximation
        follow_through_idx = min(n-1, release_idx + int(n*0.1))

        return {
            'address': 0,
            'takeback_start': takeback_start_idx,
            'takeback_max': takeback_max_idx,
            'release': release_idx,
            'follow_through': follow_through_idx
        }

    def analyze_throw(self):
        """
        Calculates biomechanical features based on Huang et al., 2024.
        Returns a dict of extracted features and detected issues.
        """
        if len(self.history) == 0:
            return {"error": "No data fed to engine"}
            
        side = self._detect_throwing_arm()
        phases = self.detect_phases(side)
        
        if not phases:
            return {"error": "Could not detect throw phases"}
            
        wrist = f'{side}_wrist'
        elbow = f'{side}_elbow'
        shoulder = f'{side}_shoulder'
        hip = f'{side}_hip'
        index = f'{side}_index'
        
        # 1. Elbow Position Stability (Variance) using normalized Y 
        elbow_y_positions = []
        for i in range(phases['address'], phases['follow_through']):
            kp = self.get_joint(i, elbow)
            if kp is not None:
                elbow_y_positions.append(kp[1])
                
        elbow_stability = np.var(elbow_y_positions) if elbow_y_positions else 0
        
        # 2. Takeback Angle (Minimum 3D angle during takeback phase)
        takeback_angles = []
        for i in range(phases['takeback_start'], phases['release']):
            p_shoulder = self.get_joint(i, shoulder)
            p_elbow = self.get_joint(i, elbow)
            p_wrist = self.get_joint(i, wrist)
            angle = self.calculate_angle_3d(p_shoulder, p_elbow, p_wrist)
            if angle > 0:
                takeback_angles.append(angle)
                
        min_takeback_angle = min(takeback_angles) if takeback_angles else 0

        # Helpers for angular velocity at Release
        release_idx = phases['release']
        vel_elbow_ext = 0
        vel_wrist_flex = 0
        
        if release_idx > 2 and release_idx < len(self.history) - 2:
            dt = (self.history[release_idx+2]['time'] - self.history[release_idx-2]['time']) / 1000.0
            if dt > 0:
                # 3. Elbow Extension Velocity
                e_ang_before = self.calculate_angle_3d(self.get_joint(release_idx-2, shoulder), self.get_joint(release_idx-2, elbow), self.get_joint(release_idx-2, wrist))
                e_ang_after = self.calculate_angle_3d(self.get_joint(release_idx+2, shoulder), self.get_joint(release_idx+2, elbow), self.get_joint(release_idx+2, wrist))
                vel_elbow_ext = (e_ang_after - e_ang_before) / dt
                
                # 4. Wrist Palmar Flexion Velocity (Snap)
                w_ang_before = self.calculate_angle_3d(self.get_joint(release_idx-2, elbow), self.get_joint(release_idx-2, wrist), self.get_joint(release_idx-2, index))
                w_ang_after = self.calculate_angle_3d(self.get_joint(release_idx+2, elbow), self.get_joint(release_idx+2, wrist), self.get_joint(release_idx+2, index))
                vel_wrist_flex = abs(w_ang_after - w_ang_before) / dt # Absolute snap rate

        # 5. Body Sway (X displacement)
        sway_amount = 0
        s_start = self.get_joint(phases['address'], shoulder)
        s_end = self.get_joint(phases['release'], shoulder)
        if s_start is not None and s_end is not None:
            sway_amount = abs(s_end[0] - s_start[0])

        issues = []
        # Adjusted Thresholds for normalized coordinates
        if elbow_stability > 0.005: 
            issues.append("elbow_unstable_y")
        if min_takeback_angle < 30:
            issues.append("takeback_too_deep")
        if min_takeback_angle > 110:
            issues.append("takeback_too_shallow")
        if vel_elbow_ext < 150: 
            issues.append("slow_elbow_extension")
        if sway_amount > 0.05:
            issues.append("body_sway_detected")
            
        result = {
            "throwing_arm": side,
            "phases_frames": phases,
            "metrics": {
                "elbow_stability_variance": float(elbow_stability),
                "takeback_min_angle_deg": float(min_takeback_angle),
                "elbow_extension_velocity_deg_s": float(vel_elbow_ext),
                "wrist_snap_velocity_deg_s": float(vel_wrist_flex),
                "body_sway_x_norm": float(sway_amount)
            },
            "issues": issues
        }
        return result
