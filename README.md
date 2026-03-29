# Ultrasound_Project Structure 

## core/
*Responsible for the mathematical heavy-lifting and spatial logic.*
* **`spatial.py`**: Coordinate system transformations and 4x4 matrix math.
* **`calibration.py`**: Logic for needle pivot calibration and Probe-to-Image (P2I) registration.
* **`recon.py`**: Algorithms for 3D volume reconstruction from tracked 2D slices.

## drivers/
*The hardware abstraction layer.*
* **`tracking.py`**: Wrapper for the NatNet (OptiTrack) client to fetch rigid body poses.
* **`imaging.py`**: Wrapper for the Clarius SDK (`pyclariuscast`) to handle the ultrasound stream.

## ui/
*The presentation and user interaction layer.*
* **`viewer.py`**: Manages Napari viewers, image layers, and real-time contrast/rendering settings.
* **`widgets.py`**: Custom Qt buttons and controls for triggering calibration, recording, and volume reconstruction.


```mermaid
graph TD
    A[app.py] --> B[ui/viewer.py]
    A --> C[drivers/imaging.py]
    A --> D[drivers/tracking.py]
    
    B --> E[core/spatial.py]
    B --> F[core/calibration.py]
    
    subgraph "Hardware Layer"
    C
    D
    end
    
    subgraph "Processing Layer"
    E
    F
    G[core/recon.py]
    end
    
    B -- "Records Frames" --> G
```
