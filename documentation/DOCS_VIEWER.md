# viewer.py 

>**Overview:** `RealTimeUSViewer` is the custom, Qt-based widget that manages the life-cycle of the entire system. It implemnents thread safety, coordinate synchronization, and memory efficiency. 

## Functions 

| Function Name | Input | Output | Logic Summary |
| :--- | :--- | :--- | :--- | 
| `__init__` | `frame_q`, `stop_event` | `None` | Initializes the UI and pre-allocates image buffers. Starts the master `QTimer` for frame updates. | 
| `update_frame` | `None` | `None` | Runs 60 times per second. Pulls `RenderPacket` from the queue, updates layer coordinates, and refreshes Napari-based viewer. | 
| `calibrate_ultrasound_transform` | `None` | `None` | Records a series of frames then solves for the spaital offset matrix. | 




