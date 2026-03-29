import ctypes
import os
import sys
import time
import threading
import numpy as np
from typing import Optional, Tuple

import argparse
import ctypes
import os.path
import sys

class UltrasoundImage:
    
    US_2D_SHAPE = (480, 640)
    FAKE_MICRONS_PER_PIXEL = 100.0
    TOP_BLACK_ROWS = 24

# _IMPORT_ERROR = None
# pyclariuscast = None
# _LIBCAST_HANDLE = None

#for mac 

libcast_path = "/Users/ashleydo/ultrasound_project/dependencies/libcast.dylib"
ctypes.CDLL(libcast_path, ctypes.RTLD_GLOBAL)
deps_path = "/Users/ashleydo/ultrasound_project/dependencies"
if deps_path not in sys.path:
    sys.path.insert(0, deps_path)
import pyclariuscast 
 
# try:
#     import pyclariuscast
# except ImportError as e:
#     print(f"Error: Module 'pyclariuscast' not found in {libcast_path}")
#     print(f"Check your .so filename. Detailed error: {e}")
#     sys.exit(1)

# try:
#     if sys.platform.startswith("linux"):
#         _LIBCAST_HANDLE = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle
#         ctypes.cdll.LoadLibrary("./pyclariuscast.so")
#     import pyclariuscast as _pyclariuscast 
# except (ImportError, OSError) as primary_exc:
#     try:
#         from dependencies import pyclariuscast as _pyclariuscast  
#     except (ImportError, OSError) as fallback_exc:
#         pyclariuscast = None
#     else:
#         pyclariuscast = _pyclariuscast
#         _IMPORT_ERROR = None
# else:
#     pyclariuscast = _pyclariuscast
#     _IMPORT_ERROR = None

#store latest us frame, timestamp, spatial calibration 

parser = argparse.ArgumentParser()
parser.add_argument("--address", "-a", dest="ip", help="ip address of probe.")
parser.add_argument("--port", "-p", dest="port", type=int, help="port of the probe")
parser.add_argument("--width", "-w", dest="width", type=int, help="image output width in pixels")
parser.add_argument("--height", "-ht", dest="height", type=int, help="image output height in pixels")
parser.set_defaults(ip="192.168.1.1")
parser.set_defaults(port=5828)
parser.set_defaults(width=640)
parser.set_defaults(height=480)
args = parser.parse_args()

class UltrasoundState:
    def __init__(self):
        self.image = np.zeros(UltrasoundImage.US_2D_SHAPE, dtype=np.float32)
        self.timestamp: Optional[float] = None
        self.microns_per_pixel: Optional[float] = None
        self.imu_buffer = []
        self.lock = threading.Lock()

STATE = UltrasoundState() #one object 

_fake_thread: Optional[threading.Thread] = None
_fake_running = False
cast = None
printStream = True


#reduce noise 

def _mask_top_rows(frame: np.ndarray) -> np.ndarray:
    rows = int(UltrasoundImage.TOP_BLACK_ROWS)
    if rows <= 0 or frame.ndim < 2:
        return frame
    rows = min(rows, frame.shape[0])
    frame[:rows, ...] = 0.0
    return frame

#clarius callbacks 

def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    
    #convert to int first to prevent callback error 
    try: 
        w = int(width)
        h = int(height)
        s = int(sz)
    
        num_pixels = w * h 
        if num_pixels == 0:
            print(f"Warning: Received image with zero pixels (width: {w}, height: {h}). Skipping processing.")
            return 
        bpp = s // num_pixels 

        if printStream:
                # Format micronsPerPixel as a float for printing, but keep it safe
                print(f"image: {timestamp}, {width}x{height} @ {bpp} bpp, {micronsPerPixel:.2f} um/px, imu: {len(imu)} pts",
                    end="\r",)

        if bpp == 1:
            arr = np.frombuffer(image, dtype=np.uint8, count=num_pixels)
            gray = arr.reshape((h, w)).astype(np.float32) / 255.0
        elif bpp == 4:
            # RGBA data
            arr = np.frombuffer(image, dtype=np.uint8, count=num_pixels * 4)
            # Use integer dimensions for reshape
            gray = arr.reshape((h, w, 4)).mean(axis=2).astype(np.float32) / 255.0
        else:
            # Fallback for other formats
            arr = np.frombuffer(image, dtype=np.uint8, count=s)
            return

        gray = _mask_top_rows(gray)

        with STATE.lock:
            STATE.image = gray
            STATE.timestamp = float(timestamp)
            STATE.microns_per_pixel = float(micronsPerPixel)

    except Exception as e:
        # This will now tell you exactly WHICH line failed
        print(f"\n[US CALLBACK ERROR] {e} | Types: w:{type(width)} sz:{type(sz)}")

    # try:
    #     num_pixels = max(1, width * height)
    #     bpp = max(1, sz / num_pixels) if sz else 1

    #     if printStream:
    #         print(
    #             f"image: {timestamp}, {width}x{height} @ {bpp} bpp, {micronsPerPixel:.2f} um/px, imu: {len(imu)} pts",
    #             end="\r",
    #         )

    #     if bpp == 1:
    #         arr = np.frombuffer(image, dtype=np.uint8, count=num_pixels)
    #         gray = arr.reshape((height, width)).astype(np.float32)
    #         gray /= 255.0

    #     elif bpp == 2:
    #         arr = np.frombuffer(image, dtype=np.uint16, count=num_pixels)
    #         gray = arr.reshape((height, width)).astype(np.float32)
    #         gray /= 65535.0

    #     else:
    #         arr = np.frombuffer(image, dtype=np.uint8, count=num_pixels * bpp)
    #         arr = arr.reshape((height, width, bpp))
    #         gray = arr.mean(axis=2).astype(np.float32)
    #         gray /= 255.0

    #     gray = _mask_top_rows(gray)

    #     with STATE.lock:
    #         STATE.image = gray
    #         STATE.timestamp = timestamp

    #         try:
    #             mpp_val = float(micronsPerPixel)
    #             if np.isfinite(mpp_val) and mpp_val > 0:
    #                 STATE.microns_per_pixel = mpp_val
    #         except Exception:
    #             pass

    # except Exception as e:
    #     print(f"[US CALLBACK ERROR] {e}", flush=True)


def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    return


def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


def newImuData(imu):
     with STATE.lock:
        for s in imu:
            STATE.imu_buffer.append((
                time.monotonic(),
                s.qw, s.qx, s.qy, s.qz,
                s.ax, s.ay, s.az,
                s.gx, s.gy, s.gz
            ))

        if len(STATE.imu_buffer) > 1000:
            STATE.imu_buffer = STATE.imu_buffer[-1000:]


def freezeFn(frozen):
    if frozen:
        print("\nimaging frozen")
    else:
        print("imaging running")


def buttonsFn(button, clicks):
    print(f"button pressed: {button}, clicks: {clicks}")

#fake streaming implementation 

def _fake_ultrasound_stream(fps=120):
    global _fake_running

    dt = 1.0 / fps
    h, w = UltrasoundImage.US_2D_SHAPE

    line_center = max(0, h - 100)
    half_thickness = 7
    top = max(0, line_center - half_thickness)
    bottom = min(h, line_center + half_thickness + 1)

    while _fake_running:
        frame = np.zeros((h, w), dtype=np.float32)
        frame[top:bottom, :] = 1.0
        frame = _mask_top_rows(frame)

        with STATE.lock:
            STATE.image = frame
            STATE.timestamp = time.monotonic()
            STATE.microns_per_pixel = UltrasoundImage.FAKE_MICRONS_PER_PIXEL

        time.sleep(dt)

#public api 

def start_ultrasound_stream(ip="192.168.1.1", port=5828, fake_data=False):
    
    global _fake_thread, _fake_running, cast

    if fake_data:
        if _fake_running:
            return
        _fake_running = True
        _fake_thread = threading.Thread(target=_fake_ultrasound_stream, daemon=True)
        _fake_thread.start()
        return

    if pyclariuscast is None:
        msg = "pyclariuscast not available"
        # if _IMPORT_ERROR:
        #     msg += f" ({_IMPORT_ERROR})"
        # raise RuntimeError(msg)

    if cast is not None:
        return

    # path = os.path.abspath(os.path.expanduser("~/")) 
    path = "/Users/ashleydo/ultrasound_project/dependencies"  
    print(f"DEBUG: Initializing Caster with path: {path}") 
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, newImuData, freezeFn, buttonsFn)
    ret = cast.init(path, args.width, args.height)

    if ret:
        print("initialization succeeded")
        ret = cast.connect(ip, port, "research")

        if ret:
            print(f"connected to {ip} on port {port}")
            cast.enableParam("imu", 1)
        else:
            print(f"connection failed. IP address is {ip} and port is {port}")
            return
    else:
        print("initialization failed")
        return
    
def stop_ultrasound_stream():
    global _fake_running, _fake_thread, cast

    if _fake_running:
        _fake_running = False
        if _fake_thread is not None:
            _fake_thread.join(timeout=1.0)
            _fake_thread = None

    if cast is not None:
        try:
            cast.disconnect()
            cast.destroy()
        except Exception:
            pass
        finally:
            cast = None


def get_new_us_frame() -> Tuple[Optional[float], Optional[np.ndarray], Optional[float]]:
    with STATE.lock:
        if STATE.timestamp is None:
            return None, None, None
        return (
            STATE.timestamp,
            STATE.image.copy(),
            STATE.microns_per_pixel,
        )
    
def get_latest_imu():
    with STATE.lock:
        return list(STATE.imu_buffer)


start_ultrasound_stream( ip=args.ip, port=args.port, fake_data=False)