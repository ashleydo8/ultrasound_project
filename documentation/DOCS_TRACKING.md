# tracking.py 

>**Overview:** Bridges 3D motion capture data (i.e. rigid bodies and markers) and Napari visualization. The NatNet SDK allows OptiTrack's Motive software to send data over UDP (User Datagram Protocol) to the Napari client application. 

## Classes 

| Class Name | Logic Summary | 
| :--- | :--- | 
| `NatNetClientWrapper` | Interface for NatNetSDK. Connects to live hardware to obtain 3D motion capture data for optical tracking. | 

## Functions 

| Function Name | Input | Output | Logic Summary |
| :--- | :--- | :--- | :--- | 
| `_permute_position_to_napari` | `(x, y, z)` | `(-z, y, x)` | To ensure accurate visualization in Napari. | 
| `_permute_quat_to_napari` | `raw quat` | `napari quat` | Performs a basis transformation on the rotation to ensure consistent tool orientation in the new coordinate system. | 
| `get_rigid_body_poses` | `None` | `Dict` | Collects tracked objects and applies permutaiton math. Returns a dictionary of IDs mapped to `position`, `rotation` tuples. | 

## Additional Information 

1. When initializing the NatNetClientWrapper, server and client addresses are the same if running on the same machine. 
