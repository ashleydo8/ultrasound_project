UltrasoundProject/
├── app.py                
├── core/
│   ├── spatial.py         # Coordinate systems & Matrix math
│   ├── calibration.py     # Pivot and Probe-to-Image logic
│   └── recon.py           # Volume reconstruction 
├── drivers/
│   ├── tracking.py        # NatNet wrapper 
│   └── imaging.py         # Clarius wrapper 
└── ui/
    ├── viewer.py          # Napari layers & Window management
    └── widgets.py         # Custom buttons for calibration/recording
