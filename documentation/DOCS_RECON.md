# recon.py

> **Overview:** Achieves 3D volumetric reconstruction from a sequence of 2D ultrasound frames with corresponding spatial poses. Transforms camera/image space to 3D world space using **Trilinear Splatting**. 

## Functions 

| Function Name | Input | Output | Logic Summary | 
| :--- | :--- | :--- | :--- | 
| `_quat_to_R_wxyz`, `_quat_to_R_xyzw` | sequence of `4 floats` | `3x3 mat` | Converts a quaternion into a standard 3x3 rotation matrix. | 
| `_pose_to_matrix` | `pose` (object with position and rotation data) | `4x4 mat` | Used for Napari visualization. Maps a Pose-like object to a 4x4 homogenous world matrix in mm. | 
| `reconstruct_volume_from_frames` | `recorded_frames` (timestamp, image/frame, pose) | `volume, weights, metadata` | Calculates the global bounding box of all frames in world space, initializes a 3D grid, and goes through each frame. Maps 2D pixels to 3D world coordinates. | 
| `_splat` (in `reconstruct_volume_from_frames`) | `pts_vox` (point coordinates in voxel space), `vals` (value per point) | `None` | Trilinear Splatting implementation. Reduces "stair-stepping" artifacts using 8 neighboring voxels and distributes pixel intensity based on fractional distance to each voxel. Result is a smoother 3D volume after linear interpolation. 

## Additional Information 

**Metadata** consists of the following: 
1. voxel size (mm)
2. origin (mm)
3. minimum bounding box (mm)
4. maximum bounding box (mm)
5. volume shape 
6. pixels per mm (image resolution)
7. pixel step
8. frame step 
9. inplane origin (center, top center, or top left)
10. intenisty scale 

