import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, Dict, List
from core.structures import Pose, SCALE, ARROW_LENGTH_MM, LOCAL_ARROW_DIR

class SpatialManager:
    
    def __init__(self, pose: Pose, pos=None, quat=None, align_quat=None):

        self.pose = pose 
        self.pos = pos
        self.quat = quat
        self.align_quat = align_quat

        self.world_to_napari = np.array([
            [0,  0, -1, 0],  
            [0,  1,  0, 0],  
            [1,  0,  0, 0],  
            [0,  0,  0, 1]
        ], dtype=np.float64)

        #probe calibration 
        self.probe_to_image = np.eye(4)

        #needle calibration
        self.needle_to_tip = np.eye(4)

    @staticmethod
    def create_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(quat).as_matrix()
        mat[:3, 3] = pos
        return mat

    def get_needle_tip_in_world(self, needle_pose_raw: np.ndarray) -> np.ndarray:
        #T_world_tip = T_world_needle * T_needle_tip
        return needle_pose_raw @ self.needle_to_tip

    def get_image_origin_in_world(self, probe_pose_raw: np.ndarray) -> np.ndarray:
        #T_world_image = T_world_probe * T_probe_image
        return probe_pose_raw @ self.probe_to_image

    def pixel_to_world(self, u: float, v: float, probe_pose_raw: np.ndarray, mm_per_px: float) -> np.ndarray:
       
        pixel_in_image = np.array([u * mm_per_px / 1000.0, v * mm_per_px / 1000.0, 0, 1])
        t_world_image = self.get_image_origin_in_world(probe_pose_raw)
        world_pt = t_world_image @ pixel_in_image
        return world_pt[:3]

    def transform_to_napari(self, world_matrix: np.ndarray) -> np.ndarray:
        return self.world_to_napari @ world_matrix

   