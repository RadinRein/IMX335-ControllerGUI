# IMX335-ControllerGUI
Python GUI APP for controlling Sony IMX335 5MP USB Camera (C)
=======
# IMX335 Camera Controller — USBcam.py

**USBcam.py** — a compact PyQt6 GUI for previewing and controlling V4L2/UVC cameras.  
Single-file reference implementation with dynamic controls (from `v4l2-ctl`), FPS control, burst capture, selectable save folder, and **user-definable filename patterns**.

---

## Quick links
- Single-file python: 'USBcam.py' # Go for the Patch code by RadinRein for updated GUI
- Supported platforms: Linux (full features with `v4l2-ctl`)
- Recommended: use with UVC cameras and V4L2-compatible drivers on Linux

---

## Features
- Live preview via OpenCV + PyQt6
- Automatic enumeration of V4L2 devices and formats (via `v4l2-ctl`)
- Dynamic control panel built from `v4l2-ctl --list-ctrls` (sliders, checkboxes, menus)
- Set FPS (via `v4l2-ctl --set-parm` + OpenCV fallback)
- Save single frames and burst capture (configurable count & interval)
- Flexible filename patterns with tokens:
  - `{timestamp}`, `{timestamp:%Y-%m-%d_%H-%M-%S}`
  - `{index}`, `{index:03d}`, `{count}`
  - `{exposure}`, `{gain}`, or other enumerated control names

## Requirements

- Python 3.8 or newer
- `PyQt6`
- `opencv-python` (cv2)
- `numpy`
- On Linux: `v4l-utils` (for `v4l2-ctl`) — recommended to enumerate devices and controls.

Install Python packages:
```bash/Terminal
pip install -r requirements.txt

# P.S. This code USBcam.py is made for Linux system. User can make a Desktop App image for their Linux OS using pyinstaller, which goes like :
pip install -r requirements.txt pyinstaller
pyinstaller --onefile USBcam.py

#The created app image will be in a folder /dist/USBcam
