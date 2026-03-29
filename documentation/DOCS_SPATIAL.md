# spatial.py

> **Overview:** `SpatialManager` acts as the geometric coordinator for the surgical navigation system. It translates raw sensor data **(Poses)** into clinically-relevant coordinates **(Needle Tip and Ultrasound Image Origin)** and reformats them for 3D visualization in Napari.

## Functions 

| Function Name | Input | Output | Logic Summary |
| :--- | :--- | :--- | :--- |
| `create_matrix` | `pos, quat` | `4x4 mat` | Converts raw tracking data to a homogeneous transform (transation and rotation data). |
| `pixel_to_world` | `u, v, pose` | `x, y, z` | **Core:** Maps 2D pixels to 3D world coordinates in meters. |
| `get_needle_tip_in_world`| `needle_pose_raw` | `4x4 mat` | Translates the needle handle pose to the physical needle tip. | 
| `get_image_origin_in_world` | `probe_pose_raw` | 4`x4 mat` | Maps the ultrasound image origin to the global coordinate system. | 
| `transform_to_napari` | `world_matrix` | `4x4 mat` | Re-aligns world coordinates for 3D visualization in Napari. | 
<!-- | `from_pos_rot` | `pos, quat` | `Pose` | **Utility:** Creates a standardized Pose object for data passing. |  -->

## Additional Information 

* `world_to_napari` Axis-swapping matrix. It ensures _X, Y, Z_ orientation in the lab matches the viewer's perspective. 

$$
\text{I} = \begin{bmatrix} 
0 & 0 & -1 & 0 \\
0 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 
\end{bmatrix}
$$
  
* `mm_per_px / 1000.0` Unit conversion from Image space (pixels) to World space (meters). 

