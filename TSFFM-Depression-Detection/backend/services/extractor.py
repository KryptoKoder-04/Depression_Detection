import os
import cv2
import numpy as np
import mediapipe as mp
from config import SEQUENCE_LENGTH, FPS

def normalize_landmarks(seq):
    """
    Normalizes a sequence of landmarks to be scale and translation invariant.
    seq shape: (seq_len, num_points, 4)
    """
    if seq.shape[1] == 0:
        return seq
        
    seq_norm = seq.copy()
    # Centering: subtract mean (x,y,z) for each frame
    mean_xyz = np.mean(seq_norm[:, :, :3], axis=1, keepdims=True)
    seq_norm[:, :, :3] -= mean_xyz
    
    # Scaling: divide by max distance from center for each frame
    max_dist = np.max(np.linalg.norm(seq_norm[:, :, :3], axis=2, keepdims=True), axis=1, keepdims=True)
    max_dist[max_dist == 0] = 1.0 # Prevent division by zero
    seq_norm[:, :, :3] /= max_dist
    
    return seq_norm

# Define standard 68 landmarks indices subset from MediaPipe FaceMesh (468 landmarks total)
# This keeps the landmarks consistent and matches the 68 shape of the training dataset.
FACEMESH_68_INDICES = [
    # Jawline (17 points) - idx 0..16
    162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    # Eyebrows (10 points) - idx 17..26
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Nose (9 points) - idx 27..35
    168, 6, 197, 195, 5, 4, 45, 275, 94,
    # Eyes (12 points) - idx 36..47
    33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373,
    # Mouth Outer (12 points) - idx 48..59
    61, 37, 267, 291, 321, 314, 17, 84, 91, 146, 12, 269,
    # Mouth Inner (8 points) - idx 60..67
    78, 191, 80, 81, 82, 312, 311, 310
]


def compute_expression_features(face_seq_raw):
    """
    Compute expression features from raw (pre-normalization) face landmarks.
    These features capture actual emotional cues (smiling, movement, expressiveness)
    that the LSTM model alone cannot reliably detect on out-of-distribution videos.
    
    face_seq_raw: numpy array of shape (seq_len, 68, 4)
    
    Returns dict with:
      - smile_score:      0..1 (0=no smile, 1=strong smile)
      - movement_score:   0..1 (0=flat/still, 1=very expressive)
      - expression_score: 0..1 weighted combination (0=likely depressed, 1=likely not)
    """
    # Only analyze frames where face was actually detected (not zero-padded)
    valid_mask = np.any(face_seq_raw[:, :, :3] != 0, axis=(1, 2))
    valid_frames = face_seq_raw[valid_mask]
    
    if len(valid_frames) < 5:
        return {"smile_score": 0.5, "movement_score": 0.5, "expression_score": 0.5}
    
    # --- 1. SMILE DETECTION ---
    # Mouth corners: index 48 (left), 51 (right)
    # Top lip center: index 58 (MediaPipe 12)
    # Bottom lip center: index 54 (MediaPipe 17)
    # Nose tip: index 33 (MediaPipe 5)
    # Jaw edges: index 0, 16
    smile_ratios = []
    corner_lifts = []
    
    for frame in valid_frames:
        left_corner  = frame[48, :2]   # (x, y)
        right_corner = frame[51, :2]
        top_lip      = frame[58, :2]
        bottom_lip   = frame[54, :2]
        jaw_left     = frame[0, :2]
        jaw_right    = frame[16, :2]
        
        mouth_width = np.linalg.norm(right_corner - left_corner)
        mouth_height = np.linalg.norm(top_lip - bottom_lip) + 1e-6
        face_width = np.linalg.norm(jaw_right - jaw_left) + 1e-6
        
        # Mouth Aspect Ratio normalized by face width
        # Smiling → mouth gets wider relative to face
        mar = mouth_width / face_width
        smile_ratios.append(mar)
        
        # Corner lift: are mouth corners higher than mouth center?
        # In image coords, y=0 is top, so "higher" = smaller y value.
        mouth_center_y = (top_lip[1] + bottom_lip[1]) / 2
        corner_avg_y = (left_corner[1] + right_corner[1]) / 2
        # Positive value = corners are ABOVE center = smile
        lift = mouth_center_y - corner_avg_y
        corner_lifts.append(lift)
    
    avg_mar = np.mean(smile_ratios)
    avg_lift = np.mean(corner_lifts)
    
    # MAR typically: ~0.25 neutral, ~0.35+ smiling (normalized by face width)
    smile_from_mar = np.clip((avg_mar - 0.22) / 0.15, 0, 1)
    # Corner lift typically: ~0 neutral, ~0.01+ smiling
    smile_from_lift = np.clip(avg_lift / 0.015, 0, 1)
    
    smile_score = 0.5 * smile_from_mar + 0.5 * smile_from_lift
    
    # --- 2. MOVEMENT / EXPRESSIVENESS ---
    # Standard deviation of landmark positions over time.
    # Depressed individuals show "flat affect" = very little facial movement.
    # Healthy individuals are more animated.
    xy_positions = valid_frames[:, :, :2]  # (N, 68, 2)
    
    # Per-landmark temporal std, averaged across all landmarks
    temporal_std = np.std(xy_positions, axis=0).mean()
    
    # Also compute velocity (frame-to-frame movement)
    if len(valid_frames) > 1:
        diffs = np.diff(xy_positions, axis=0)
        avg_velocity = np.mean(np.abs(diffs))
    else:
        avg_velocity = 0.0
    
    # Movement: combine std and velocity
    # Typical ranges: std ~0.002-0.02, velocity ~0.001-0.01
    movement_from_std = np.clip(temporal_std / 0.015, 0, 1)
    movement_from_vel = np.clip(avg_velocity / 0.005, 0, 1)
    movement_score = 0.5 * movement_from_std + 0.5 * movement_from_vel
    
    # --- 3. COMBINED EXPRESSION SCORE ---
    # Higher = more expressive/positive = less likely depressed
    expression_score = 0.55 * smile_score + 0.45 * movement_score
    
    return {
        "smile_score": float(np.clip(smile_score, 0, 1)),
        "movement_score": float(np.clip(movement_score, 0, 1)),
        "expression_score": float(np.clip(expression_score, 0, 1)),
    }


def extract_features_from_video(video_path):
    """
    Reads a video, extracts face (68 points) and pose (2 shoulder points) 
    using MediaPipe, and returns padded/truncated arrays of shape (360, 272) and (360, 8).
    Also computes expression features for inference calibration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate step size to achieve target FPS (5 FPS)
    step = max(1, int(round(video_fps / FPS)))
    
    # Initialize MediaPipe solutions
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    face_sequence = []
    pose_sequence = []
    face_detected_count = 0
    pose_detected_count = 0
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample frames based on our target FPS step
        if frame_idx % step == 0:
            # Convert color space for MediaPipe (BGR -> RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Face Features
            face_results = mp_face_mesh.process(rgb_frame)
            frame_face = np.zeros((68, 4), dtype=np.float64) # x, y, z, confidence
            
            if face_results.multi_face_landmarks:
                face_detected_count += 1
                landmarks = face_results.multi_face_landmarks[0].landmark
                # Extract the 68 standard subset landmarks
                for idx, mesh_idx in enumerate(FACEMESH_68_INDICES):
                    lm = landmarks[mesh_idx]
                    frame_face[idx] = [lm.x, lm.y, lm.z, 1.0] # 1.0 confidence as fallback
            else:
                # If no face is detected, we reuse the last detected frame's features if available
                if len(face_sequence) > 0:
                    frame_face = face_sequence[-1].copy()
                    
            face_sequence.append(frame_face)
            
            # 2. Pose Features (Shoulders: Left shoulder 11, Right shoulder 12)
            pose_results = mp_pose.process(rgb_frame)
            frame_pose = np.zeros((2, 4), dtype=np.float64) # x, y, z, visibility
            
            if pose_results.pose_landmarks:
                pose_detected_count += 1
                landmarks = pose_results.pose_landmarks.landmark
                # Left Shoulder
                ls = landmarks[11]
                frame_pose[0] = [ls.x, ls.y, ls.z, ls.visibility]
                # Right Shoulder
                rs = landmarks[12]
                frame_pose[1] = [rs.x, rs.y, rs.z, rs.visibility]
            else:
                if len(pose_sequence) > 0:
                    frame_pose = pose_sequence[-1].copy()
                    
            pose_sequence.append(frame_pose)
            
        frame_idx += 1
        
    cap.release()
    mp_face_mesh.close()
    mp_pose.close()
    
    # --- Compute expression features BEFORE normalization/padding ---
    # These operate on the raw landmark coordinates where facial geometry is meaningful.
    raw_face_for_expression = np.array(face_sequence, dtype=np.float64) if face_sequence else np.zeros((1, 68, 4))
    expression_metrics = compute_expression_features(raw_face_for_expression)
    
    # Pad or truncate to target SEQUENCE_LENGTH (360 frames)
    sampled_frames = len(face_sequence)
    current_length = sampled_frames
    if current_length == 0:
        face_seq_np = np.zeros((SEQUENCE_LENGTH, 68, 4), dtype=np.float64)
        pose_seq_np = np.zeros((SEQUENCE_LENGTH, 2, 4), dtype=np.float64)
    elif current_length < SEQUENCE_LENGTH:
        # Zero-pad with neutral frames (NOT last-frame repeat, which causes bias)
        pad_size = SEQUENCE_LENGTH - current_length
        pad_face = np.zeros((pad_size, 68, 4), dtype=np.float64)
        pad_pose = np.zeros((pad_size, 2, 4), dtype=np.float64)
        
        face_seq_np = np.concatenate([np.array(face_sequence, dtype=np.float64), pad_face], axis=0)
        pose_seq_np = np.concatenate([np.array(pose_sequence, dtype=np.float64), pad_pose], axis=0)
    else:
        # If longer than needed, take the MIDDLE portion (most expressive part of interview)
        start = (current_length - SEQUENCE_LENGTH) // 2
        face_seq_np = np.array(face_sequence[start:start+SEQUENCE_LENGTH], dtype=np.float64)
        pose_seq_np = np.array(pose_sequence[start:start+SEQUENCE_LENGTH], dtype=np.float64)
        
    # Apply Scale & Translation Normalization
    face_seq_np = normalize_landmarks(face_seq_np)
    pose_seq_np = normalize_landmarks(pose_seq_np)
        
    # Flatten features to 2D matching model inputs
    # (360, 68, 4) -> (360, 272)
    face_features_flat = face_seq_np.reshape(SEQUENCE_LENGTH, -1)
    # (360, 2, 4) -> (360, 8)
    pose_features_flat = pose_seq_np.reshape(SEQUENCE_LENGTH, -1)
    
    metadata = {
        "sampled_frames": sampled_frames,
        "frames_processed": int(min(sampled_frames, SEQUENCE_LENGTH)),
        "total_video_frames": int(total_frames),
        "video_fps": float(video_fps or 0.0),
        "target_fps": int(FPS),
        "sequence_length": int(SEQUENCE_LENGTH),
        "face_detection_rate": float(face_detected_count / sampled_frames) if sampled_frames else 0.0,
        "body_detection_rate": float(pose_detected_count / sampled_frames) if sampled_frames else 0.0,
        "expression_metrics": expression_metrics,
    }
    
    return face_features_flat, pose_features_flat, metadata
