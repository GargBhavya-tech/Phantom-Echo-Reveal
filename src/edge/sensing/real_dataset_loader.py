"""
PHANTOM-ECHO REVEAL — Real Dataset Loader
Reads RGB, Depth, and Pose data from a folder.
"""

import os
import json
import numpy as np
import cv2
import glob
import time
from typing import List, Dict, Optional
import logging

from src.edge.sensing.arkit_depth import DepthFrame

logger = logging.getLogger(__name__)

class RealDepthGenerator:
    """
    Reads real sensor data from a directory structure.
    Expected format in dataset_path:
      - color/*.jpg or *.png
      - depth/*.png (16-bit, scaled) or *.npy
      - pose/*.txt (4x4 matrices)
      - intrinsics.json
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self._frame_id = 0
        
        self.color_files = sorted(glob.glob(os.path.join(dataset_path, "color", "*.*")))
        self.depth_files = sorted(glob.glob(os.path.join(dataset_path, "depth", "*.*")))
        self.pose_files  = sorted(glob.glob(os.path.join(dataset_path, "pose", "*.txt")))
        
        intrinsics_file = os.path.join(dataset_path, "intrinsics.json")
        if os.path.exists(intrinsics_file):
            with open(intrinsics_file, "r") as f:
                self.intrinsics = json.load(f)
        else:
            # Fallback to standard 640x480 RealSense/Kinect intrinsics
            self.intrinsics = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5}

        if not self.color_files:
            logger.warning(f"No color files found in {dataset_path}/color")
            
    def _load_pose(self, filepath: str) -> np.ndarray:
        try:
            pose = np.loadtxt(filepath)
            if pose.shape == (4, 4):
                return pose
        except Exception as e:
            logger.error(f"Failed to load pose {filepath}: {e}")
        return np.eye(4)

    def _load_depth(self, filepath: str) -> np.ndarray:
        if filepath.endswith('.npy'):
            return np.load(filepath).astype(np.float32)
        else:
            # Assume 16-bit PNG depth where 1000 = 1 meter
            depth_img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                return depth_img.astype(np.float32) / 1000.0
        return np.zeros((480, 640), dtype=np.float32)
        
    def generate_walk_sequence(self, n_frames: int = 10, **kwargs) -> List[DepthFrame]:
        """
        Reads sequence from disk instead of generating it.
        Ignores start_pos and axis, since poses are fixed by the dataset.
        """
        frames = []
        n_avail = min(len(self.color_files), len(self.depth_files), len(self.pose_files))
        n_load = min(n_frames, n_avail)
        
        if n_load == 0:
            logger.error("No valid real frames found to load.")
            return frames
            
        for i in range(n_load):
            rgb = cv2.imread(self.color_files[i])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            depth = self._load_depth(self.depth_files[i])
            pose = self._load_pose(self.pose_files[i])
            
            # Simple confidence map: 2 for close, 1 for far, 0 for missing/invalid
            conf_map = np.zeros_like(depth, dtype=np.uint8)
            conf_map[depth > 0.1] = 2
            conf_map[depth > 3.0] = 1
            conf_map[depth == 0] = 0
            
            frame = DepthFrame(
                depth_map=depth,
                confidence_map=conf_map,
                rgb_image=rgb,
                camera_intrinsics=self.intrinsics,
                camera_to_world=pose,
                timestamp_s=time.time() + i,
                frame_id=self._frame_id
            )
            frames.append(frame)
            self._frame_id += 1
            
        return frames
