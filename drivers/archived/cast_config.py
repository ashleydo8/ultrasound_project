import os
import sys
import ctypes
import cv2
import numpy as np

libcast_path = "/Users/ashleydo/ultrasound_project/dependencies/libcast.dylib"

if not os.path.exists(libcast_path):
    print(f"Error: libcast.dylib not found at {libcast_path}")
    sys.exit(1)

try:
    ctypes.CDLL(libcast_path, ctypes.RTLD_GLOBAL)
    print(f"Loaded libcast.dylib from {libcast_path}")
except OSError as e:
    print("Failed to load libcast.dylib:", e)
    print("Consider copying it to /usr/local/lib or setting DYLD_LIBRARY_PATH before running Python")
    sys.exit(1)

try:
    import pyclariuscast
    print("Successfully imported pyclariuscast")
except ImportError as e:
    print("Failed to import pyclariuscast:", e)
    sys.exit(1)

# sudo cp /Users/ashleydo/ultrasound_project/dependencies/libcast.dylib /usr/local/lib/
# PYTHONPATH=/Users/ashleydo/ultrasound_project/dependencies python3 cast_config.py