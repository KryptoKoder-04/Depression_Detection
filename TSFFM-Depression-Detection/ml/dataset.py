import os
import numpy as np
import torch
from torch.utils.data import Dataset

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


class DepressionDataset(Dataset):
    """
    PyTorch Dataset to load pre-extracted features from the DAIC-WOZ-E dataset splits.
    Each participant directory contains:
    - {split}_ft_fkps_{id}.npy : (6, 1800, 68, 4) -> face keypoint sequence
    - {split}_ft_pose_conf_{id}.npy : (6, 1800, 2, 4) -> pose features sequence
    - {split}_phq_binary_{id}.npy : (6,) -> binary label
    """
    def __init__(self, split_dir, split_name="train"):
        super(DepressionDataset, self).__init__()
        self.split_dir = split_dir
        self.split_name = split_name
        self.samples = []
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Directory {split_dir} does not exist.")
            
        participants = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for p in participants:
            p_dir = os.path.join(split_dir, p)
            
            # Formulate filenames based on the split name (train / valid / test)
            fkps_file = os.path.join(p_dir, f"{split_name}_ft_fkps_{p}.npy")
            pose_file = os.path.join(p_dir, f"{split_name}_ft_pose_conf_{p}.npy")
            label_file = os.path.join(p_dir, f"{split_name}_phq_binary_{p}.npy")
            
            # In some cases test set labels might be in a different file or missing, but DAIC-WOZ-E test set typically has them or we skip if unavailable
            if os.path.exists(fkps_file) and os.path.exists(pose_file) and os.path.exists(label_file):
                try:
                    labels = np.load(label_file)
                    num_segments = len(labels)
                    for idx in range(num_segments):
                        self.samples.append({
                            'participant_id': p,
                            'fkps_path': fkps_file,
                            'pose_path': pose_file,
                            'label_path': label_file,
                            'segment_idx': idx,
                            'label': int(labels[idx])
                        })
                except Exception as e:
                    print(f"Error reading npy files for participant {p}: {e}")
                    
        print(f"Loaded {len(self.samples)} segments from {len(participants)} participants in split '{split_name}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load the specific segment's pre-extracted features
        # Note: loading files inside __getitem__ is memory efficient for large datasets, 
        # but since numpy files are small, we can load them quickly.
        fkps_all = np.load(sample_info['fkps_path'])
        pose_all = np.load(sample_info['pose_path'])
        
        seg_idx = sample_info['segment_idx']
        
        # Temporal Jittering: Randomly shift the starting frame during training
        start_idx = np.random.randint(0, 5) if self.split_name == "train" else 0
        
        # Extract features for this segment and downsample by taking every 5th frame (1800 -> 360)
        fkps = fkps_all[seg_idx][start_idx::5] # Shape: (360, 68, 4)
        pose = pose_all[seg_idx][start_idx::5] # Shape: (360, 2, 4)
        
        # Apply Scale & Translation Normalization
        fkps = normalize_landmarks(fkps)
        pose = normalize_landmarks(pose)
        
        # Spatial Jittering: Add Gaussian noise during training
        if self.split_name == "train":
            # Add small random noise (std=0.01)
            fkps = fkps + np.random.normal(0, 0.01, size=fkps.shape)
            pose = pose + np.random.normal(0, 0.01, size=pose.shape)
            
            # Temporal Masking: Randomly zero out a block of 10-30 frames to prevent overfitting
            seq_len = fkps.shape[0]
            mask_len = np.random.randint(10, 31)
            if seq_len > mask_len:
                mask_start = np.random.randint(0, seq_len - mask_len)
                fkps[mask_start:mask_start+mask_len] = 0.0
                pose[mask_start:mask_start+mask_len] = 0.0
            
        label = sample_info['label']
        
        # Flatten the features over landmark and coordinate channels:
        # (1800, 68, 4) -> (1800, 272)
        fkps_flat = fkps.reshape(fkps.shape[0], -1)
        # (1800, 2, 4) -> (1800, 8)
        pose_flat = pose.reshape(pose.shape[0], -1)
        
        # Convert to PyTorch FloatTensors
        fkps_tensor = torch.tensor(fkps_flat, dtype=torch.float32)
        pose_tensor = torch.tensor(pose_flat, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return fkps_tensor, pose_tensor, label_tensor
