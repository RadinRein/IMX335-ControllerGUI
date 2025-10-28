#!/usr/bin/env python3
"""
USBcam.py

PyQt6 GUI for controlling a V4L2 / UVC camera with dynamic controls, FPS, burst capture,
selectable save folder, and **user-definable filename pattern**.

Filename pattern tokens:
  {timestamp}                    -> current time using default "%Y%m%d_%H%M%S"
  {timestamp:%Y-%m-%d_%H-%M-%S}  -> timestamp with custom strftime format
  {index}                        -> frame index in burst (0..N-1)
  {index:03d}                    -> index with format spec (Python format spec)
  {count}                        -> total count
  {exposure}                     -> exposure (if available) else "NA"
  {gain}                         -> gain (if available) else "NA"

Examples:
  capture_{timestamp}.jpg
  burst_{timestamp}_{index:03d}.jpg
  img_{timestamp:%Y%m%d}_{index:02d}_exp{exposure}.png

Usage:
    python3 USBcam.py
"""
import sys
import os
import re
import subprocess
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

# ---------- helper (v4l2-ctl wrappers, parsing) ----------
def run_cmd(cmd: List[str]) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except FileNotFoundError:
        return ""
    except subprocess.CalledProcessError as e:
        return (e.stdout or "") + "\n" + (e.stderr or "")

def v4l2_available() -> bool:
    return bool(run_cmd(["which", "v4l2-ctl"]).strip())

def list_v4l2_devices() -> Dict[str, List[str]]:
    out = run_cmd(["v4l2-ctl", "--list-devices"])
    devices = {}
    if not out:
        return devices
    cur = None
    for line in out.splitlines():
        if not line.strip():
            cur = None
            continue
        if not line.startswith('\t') and not line.startswith(' '):
            cur = line.strip()
            devices[cur] = []
        else:
            if cur is None:
                continue
            devices[cur].append(line.strip())
    return devices

def list_formats(dev: str) -> List[Tuple[str, int, int]]:
    out = run_cmd(["v4l2-ctl", "--device", dev, "--list-formats-ext"])
    formats = []
    size_re = re.compile(r'\s*Size:\s*Discrete\s+(\d+)x(\d+)')
    fourcc = None
    for line in out.splitlines():
        m_fmt = re.match(r'\s*\[\d+\]:\s*\'([A-Z0-9]+)\'', line)
        if m_fmt:
            fourcc = m_fmt.group(1)
        m = size_re.match(line)
        if m and fourcc:
            w, h = int(m.group(1)), int(m.group(2))
            formats.append((fourcc, w, h))
    uniq = {}
    for fcc, w, h in formats:
        uniq[(w, h)] = fcc
    res = sorted([(fcc, w, h) for (w, h), fcc in uniq.items()], key=lambda x: x[1] * x[2])
    return res

def parse_ctrls(raw: str) -> Dict[str, Dict[str, Any]]:
    controls = {}
    if not raw:
        return controls
    line_re = re.compile(r'^\s*(?P<name>[^\(]+?)\s+[0-9xa-fA-F_]+\s*\((?P<type>[^)]+)\)\s*:\s*(?P<rest>.*)$')
    kv_re = re.compile(r'([a-zA-Z0-9_]+)=([^\s]+)')
    value_label_re = re.compile(r'value=(\d+)\s*\(([^)]+)\)')
    for line in raw.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        orig_name = m.group('name').strip()
        typ = m.group('type').strip()
        rest = m.group('rest').strip()
        norm = re.sub(r'[^0-9a-zA-Z]+', '_', orig_name).strip('_').lower()
        kv = {}
        for km in kv_re.finditer(rest):
            k = km.group(1)
            v = km.group(2)
            try:
                v2 = int(v)
            except ValueError:
                v2 = v
            kv[k] = v2
        label_match = value_label_re.search(rest)
        value_label = label_match.group(2) if label_match else None
        controls[norm] = {
            'name': orig_name,
            'type': typ,
            'raw': rest,
            'value_label': value_label,
            **kv
        }
    return controls

def set_ctrl(dev: str, ctrl_name: str, value: Any) -> bool:
    cmd = ["v4l2-ctl", "--device", dev, f"--set-ctrl={ctrl_name}={value}"]
    out = run_cmd(cmd)
    return out is not None

def set_framerate_v4l2(dev: str, fps: int) -> bool:
    cmd = ["v4l2-ctl", "--device", dev, f"--set-parm={fps}"]
    out = run_cmd(cmd)
    return out is not None

# ---------- Camera worker ----------
class CameraWorker(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(str)

    def __init__(self, dev_index: int, width: int, height: int, backend_flag=None, parent=None):
        super().__init__(parent)
        self.dev_index = dev_index
        self.width = width
        self.height = height
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_flag = backend_flag or (cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else 0)

    def start(self):
        if self.running:
            return
        try:
            if self.backend_flag:
                self.cap = cv2.VideoCapture(self.dev_index, self.backend_flag)
            else:
                self.cap = cv2.VideoCapture(self.dev_index)
            if not self.cap or not self.cap.isOpened():
                self.error.emit(f"Failed to open VideoCapture index {self.dev_index}")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
            self.running = True
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.running = False
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def tick(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.error.emit("Failed to read frame from camera.")
            return
        self.frame_ready.emit(frame)

# ---------- MainWindow ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMX335 Camera GUI (pattern filenames)")
        self.resize(1250, 820)
        self._build_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.worker: Optional[CameraWorker] = None
        self.current_frame: Optional[np.ndarray] = None
        self._ctrl_widgets = {}  # norm_key -> meta dict

        # burst state
        self._burst_timer: Optional[QtCore.QTimer] = None
        self._burst_remaining = 0
        self._burst_prefix = ""

        # default save folder
        self.save_folder = os.getcwd()
        self.le_save_folder.setText(self.save_folder)

        # default pattern
        self.le_pattern.setText("capture_{timestamp:%Y%m%d_%H%M%S}.jpg")

        self._populate_devices()

    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        main_layout = QtWidgets.QVBoxLayout(w)

        # top row: device/resolution/preview
        top = QtWidgets.QHBoxLayout()
        self.device_combo = QtWidgets.QComboBox(); self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        top.addWidget(QtWidgets.QLabel("Device:")); top.addWidget(self.device_combo)
        self.res_combo = QtWidgets.QComboBox(); top.addWidget(QtWidgets.QLabel("Resolution:")); top.addWidget(self.res_combo)
        self.btn_refresh = QtWidgets.QPushButton("Refresh"); self.btn_refresh.clicked.connect(self._populate_devices); top.addWidget(self.btn_refresh)
        self.btn_start = QtWidgets.QPushButton("Start Preview"); self.btn_start.clicked.connect(self._on_start_clicked); top.addWidget(self.btn_start)
        self.btn_stop = QtWidgets.QPushButton("Stop Preview"); self.btn_stop.clicked.connect(self._on_stop_clicked); self.btn_stop.setEnabled(False); top.addWidget(self.btn_stop)
        main_layout.addLayout(top)

        # middle
        mid = QtWidgets.QHBoxLayout()
        # preview
        self.video_label = QtWidgets.QLabel(); self.video_label.setMinimumSize(640,480); self.video_label.setStyleSheet("background-color: black;"); self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        mid.addWidget(self.video_label, stretch=1)

        # right column controls
        right = QtWidgets.QVBoxLayout()

        # frame rate group
        fr_group = QtWidgets.QGroupBox("Frame Rate Control")
        fr_layout = QtWidgets.QHBoxLayout(fr_group)
        fr_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spin = QtWidgets.QSpinBox(); self.fps_spin.setRange(1,240); self.fps_spin.setValue(30); fr_layout.addWidget(self.fps_spin)
        self.btn_set_fps = QtWidgets.QPushButton("Set FPS"); self.btn_set_fps.clicked.connect(self._on_set_fps); fr_layout.addWidget(self.btn_set_fps)
        self.lbl_fps_status = QtWidgets.QLabel(""); fr_layout.addWidget(self.lbl_fps_status, stretch=1)
        right.addWidget(fr_group)

        # burst group
        burst_group = QtWidgets.QGroupBox("Burst Capture")
        burst_layout = QtWidgets.QGridLayout(burst_group)
        burst_layout.addWidget(QtWidgets.QLabel("Count:"),0,0); self.burst_count_spin = QtWidgets.QSpinBox(); self.burst_count_spin.setRange(1,1000); self.burst_count_spin.setValue(5); burst_layout.addWidget(self.burst_count_spin,0,1)
        burst_layout.addWidget(QtWidgets.QLabel("Interval (ms):"),1,0); self.burst_interval_spin = QtWidgets.QSpinBox(); self.burst_interval_spin.setRange(10,60000); self.burst_interval_spin.setValue(200); burst_layout.addWidget(self.burst_interval_spin,1,1)
        self.btn_start_burst = QtWidgets.QPushButton("Start Burst Capture"); self.btn_start_burst.clicked.connect(self._on_start_burst); burst_layout.addWidget(self.btn_start_burst,2,0,1,2)
        self.lbl_burst_status = QtWidgets.QLabel(""); burst_layout.addWidget(self.lbl_burst_status,3,0,1,2)
        right.addWidget(burst_group)

        # save folder group
        save_group = QtWidgets.QGroupBox("Save Folder & Filename Pattern")
        save_layout = QtWidgets.QVBoxLayout(save_group)
        row = QtWidgets.QHBoxLayout()
        self.le_save_folder = QtWidgets.QLineEdit(); self.le_save_folder.setPlaceholderText("Select folder to save images"); row.addWidget(self.le_save_folder)
        self.btn_browse = QtWidgets.QPushButton("Browse"); self.btn_browse.clicked.connect(self._on_browse_folder); row.addWidget(self.btn_browse)
        self.btn_open_folder = QtWidgets.QPushButton("Open"); self.btn_open_folder.clicked.connect(self._on_open_folder); row.addWidget(self.btn_open_folder)
        save_layout.addLayout(row)
        # pattern input
        save_layout.addWidget(QtWidgets.QLabel("Filename pattern:"))
        self.le_pattern = QtWidgets.QLineEdit()
        self.le_pattern.setToolTip("Use tokens like {timestamp}, {timestamp:%Y%m%d_%H%M%S}, {index}, {index:03d}, {count}, {exposure}, {gain}")
        save_layout.addWidget(self.le_pattern)
        # example label
        self.lbl_pattern_example = QtWidgets.QLabel("Example: capture_{timestamp}_{index:03d}.jpg")
        save_layout.addWidget(self.lbl_pattern_example)
        right.addWidget(save_group)

        # dynamic controls area (scroll)
        self.ctrl_scroll = QtWidgets.QScrollArea(); self.ctrl_scroll.setWidgetResizable(True)
        self.ctrl_container = QtWidgets.QWidget(); self.ctrl_layout = QtWidgets.QVBoxLayout(self.ctrl_container); self.ctrl_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.ctrl_scroll.setWidget(self.ctrl_container)
        right.addWidget(self.ctrl_scroll, stretch=1)

        # save single frame
        self.btn_save = QtWidgets.QPushButton("Save Frame"); self.btn_save.clicked.connect(self._on_save_frame); self.btn_save.setEnabled(False)
        right.addWidget(self.btn_save)

        right.addStretch(1)
        mid.addLayout(right, stretch=0)
        main_layout.addLayout(mid)

        # bottom status
        self.status = QtWidgets.QLabel("")
        main_layout.addWidget(self.status)

    # ---- devices/formats ----
    def _populate_devices(self):
        self.device_combo.clear()
        if not v4l2_available():
            self.device_combo.addItem("/dev/video0 (v4l2-ctl not found)","/dev/video0")
            self.status.setText("v4l2-ctl not found; install v4l-utils for full functionality.")
            return
        devs = list_v4l2_devices()
        if not devs:
            self.device_combo.addItem("/dev/video0 (no devices found)","/dev/video0")
            self.status.setText("No v4l2 devices found.")
            return
        for name, paths in devs.items():
            for p in paths:
                self.device_combo.addItem(f"{name} — {p}", p)
        self.status.setText("Select a device and resolution, then Start Preview.")

    def _on_device_selected(self, idx: int):
        dev = self.device_combo.currentData()
        if not dev:
            return
        self.res_combo.clear()
        if v4l2_available():
            fmts = list_formats(dev)
            if fmts:
                for fcc,w,h in fmts:
                    self.res_combo.addItem(f"{w}x{h} ({fcc})", (w,h))
            else:
                self.res_combo.addItem("1280x720",(1280,720))
        else:
            self.res_combo.addItem("1280x720",(1280,720))
        self._build_controls_for_device(dev)

    # ---- dynamic controls builder ----
    def _clear_controls_ui(self):
        while self.ctrl_layout.count():
            item = self.ctrl_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._ctrl_widgets = {}

    def _build_controls_for_device(self, dev: str):
        self._clear_controls_ui()
        if not v4l2_available():
            self.status.setText("v4l2-ctl missing; cannot enumerate controls.")
            return
        raw = run_cmd(["v4l2-ctl","--device", dev, "--list-ctrls"])
        if not raw:
            self.status.setText("No controls returned by v4l2-ctl.")
            return
        ctrls = parse_ctrls(raw)
        if not ctrls:
            self.status.setText("Failed to parse controls.")
            return

        preferred_order = [
            'brightness','contrast','saturation','hue',
            'white_balance_automatic','white_balance_temperature','gamma','power_line_frequency',
            'sharpness','backlight_compensation',
            'auto_exposure','exposure_time_absolute','exposure_dynamic_framerate',
            'pan_absolute','tilt_absolute','zoom_absolute'
        ]

        def add_control(norm_key, info):
            disp_name = info.get('name', norm_key)
            typ = info.get('type','').lower()
            row_w = QtWidgets.QWidget(); row = QtWidgets.QHBoxLayout(row_w); row.setContentsMargins(6,4,6,4)
            lbl = QtWidgets.QLabel(disp_name); lbl.setMinimumWidth(200); row.addWidget(lbl)

            meta = {'orig_name': info.get('name', norm_key), 'type': typ}
            inactive = False
            if 'flags' in info and str(info.get('flags')).lower() == 'inactive':
                inactive = True
            if 'inactive' in str(info.get('raw','')).lower():
                inactive = True

            if 'int' in typ:
                minv = int(info.get('min',0)); maxv = int(info.get('max', minv+1000)); cur = info.get('value', minv)
                try: cur_i = int(cur)
                except: cur_i = minv
                slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); slider.setMinimum(0); slider.setMaximum(maxv-minv); slider.setValue(cur_i-minv); slider.setEnabled(not inactive)
                val_label = QtWidgets.QLabel(str(cur_i)); val_label.setMinimumWidth(80)
                row.addWidget(slider, stretch=1); row.addWidget(val_label)
                self.ctrl_layout.addWidget(row_w)
                meta.update({'min':minv,'max':maxv,'widget':slider,'val_label':val_label,'inactive':inactive})

                def on_slider(v, nk=norm_key):
                    m = self._ctrl_widgets.get(nk)
                    if not m: return
                    actual = v + m['min']
                    ok = set_ctrl(dev, m['orig_name'], actual)
                    if ok:
                        m['val_label'].setText(str(actual))
                    else:
                        self.status.setText(f"Failed to set {m['orig_name']} -> {actual}")
                slider.valueChanged.connect(on_slider)
                self._ctrl_widgets[norm_key] = meta

            elif 'bool' in typ:
                cur = info.get('value',0)
                try: cur_b = bool(int(cur))
                except: cur_b = bool(cur)
                chk = QtWidgets.QCheckBox(); chk.setChecked(cur_b); chk.setEnabled(not inactive)
                row.addWidget(chk); row.addStretch(1); self.ctrl_layout.addWidget(row_w)
                meta.update({'widget':chk,'inactive':inactive})
                def on_chk(state, nk=norm_key):
                    m = self._ctrl_widgets.get(nk); 
                    if not m: return
                    val = 1 if state else 0
                    ok = set_ctrl(dev, m['orig_name'], val)
                    if not ok:
                        self.status.setText(f"Failed to set {m['orig_name']} -> {val}")
                    else:
                        if nk in ('white_balance_automatic','auto_exposure'):
                            QtCore.QTimer.singleShot(200, lambda: self._build_controls_for_device(dev))
                chk.stateChanged.connect(on_chk)
                self._ctrl_widgets[norm_key] = meta

            elif 'menu' in typ:
                minv = int(info.get('min',0)); maxv = int(info.get('max',minv)); cur = int(info.get('value',minv)); value_label = info.get('value_label')
                combo = QtWidgets.QComboBox()
                for v in range(minv, maxv+1):
                    entry = str(v)
                    if v==cur and value_label: entry = f"{v} ({value_label})"
                    combo.addItem(entry, v)
                combo.setCurrentIndex(cur-minv); combo.setEnabled(not inactive)
                row.addWidget(combo, stretch=1); row.addStretch(0); self.ctrl_layout.addWidget(row_w)
                meta.update({'min':minv,'max':maxv,'widget':combo,'inactive':inactive})
                def on_combo(idx, nk=norm_key):
                    m = self._ctrl_widgets.get(nk); 
                    if not m: return
                    val = m['widget'].currentData()
                    ok = set_ctrl(dev, m['orig_name'], val)
                    if not ok:
                        self.status.setText(f"Failed to set {m['orig_name']} -> {val}")
                    else:
                        if nk == 'auto_exposure': QtCore.QTimer.singleShot(200, lambda: self._build_controls_for_device(dev))
                combo.currentIndexChanged.connect(on_combo)
                self._ctrl_widgets[norm_key] = meta

            else:
                row.addWidget(QtWidgets.QLabel(str(info.get('value','')))); self.ctrl_layout.addWidget(row_w)
                meta.update({'widget': None, 'inactive': True})
                self._ctrl_widgets[norm_key] = meta

        added = set()
        for k in preferred_order:
            if k in ctrls:
                add_control(k, ctrls[k]); added.add(k)
        for k in sorted(ctrls.keys()):
            if k in added: continue
            add_control(k, ctrls[k])

        self.ctrl_layout.addStretch(1)
        self.status.setText("Controls loaded.")

    # ---- preview start/stop ----
    def _on_start_clicked(self):
        dev = self.device_combo.currentData(); res = self.res_combo.currentData()
        if not dev or not res:
            self.status.setText("Select a device and resolution first."); return
        w,h = res
        idx = self._devpath_to_index(dev)
        if idx is None:
            self.status.setText(f"Cannot parse device index from {dev}"); return
        self.worker = CameraWorker(dev_index=idx, width=w, height=h)
        self.worker.frame_ready.connect(self._on_frame); self.worker.error.connect(self._on_worker_error)
        self.worker.start(); self.timer.start(30)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_save.setEnabled(True)
        self.status.setText("Preview started.")

    def _on_stop_clicked(self):
        if self.timer.isActive(): self.timer.stop()
        if self.worker: self.worker.stop(); self.worker=None
        self.video_label.clear(); self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_save.setEnabled(False)
        self.status.setText("Preview stopped.")

    def _on_timer(self):
        if self.worker: self.worker.tick()

    def _on_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape; bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix); self.current_frame = frame

    def _on_worker_error(self, text: str):
        self.status.setText("Camera error: "+str(text)); self._on_stop_clicked()

    # ---- FPS ----
    def _on_set_fps(self):
        dev = self.device_combo.currentData()
        if not dev: self.status.setText("Select device first."); return
        fps = int(self.fps_spin.value())
        success_v4l2 = False
        if v4l2_available():
            try: success_v4l2 = set_framerate_v4l2(dev, fps)
            except: success_v4l2 = False
        success_cv = False
        if self.worker and self.worker.cap:
            try: success_cv = bool(self.worker.cap.set(cv2.CAP_PROP_FPS, float(fps)))
            except: success_cv = False
        parts = []
        parts.append("set-parm OK" if success_v4l2 else "set-parm failed")
        parts.append("OpenCV FPS set OK" if success_cv else "OpenCV FPS set failed")
        self.lbl_fps_status.setText(", ".join(parts)); self.status.setText(f"Requested FPS={fps}. {', '.join(parts)}")

    # ---- burst capture ----
    def _on_start_burst(self):
        if self.worker is None or self.worker.cap is None:
            self.status.setText("Start preview before burst capture."); return
        if self.current_frame is None:
            self.status.setText("No frame available yet."); return
        count = int(self.burst_count_spin.value()); interval = int(self.burst_interval_spin.value())
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._burst_prefix = f"burst_{ts}"
        self._burst_remaining = count
        # disable conflicting UI
        self.btn_start_burst.setEnabled(False); self.btn_start.setEnabled(False); self.btn_stop.setEnabled(False); self.btn_save.setEnabled(False); self.btn_browse.setEnabled(False)
        self.status.setText(f"Starting burst: {count} frames every {interval} ms"); self.lbl_burst_status.setText(f"{self._burst_remaining} remaining")
        # timer for burst
        self._burst_timer = QtCore.QTimer(); self._burst_timer.setInterval(interval); self._burst_timer.timeout.connect(self._on_burst_tick)
        QtCore.QTimer.singleShot(0, self._on_burst_tick)
        self._burst_timer.start()

    def _on_burst_tick(self):
        if self._burst_remaining <= 0:
            self._finish_burst(); return
        if self.current_frame is None:
            self._burst_remaining -= 1
            self.lbl_burst_status.setText(f"{self._burst_remaining} remaining (frame missing)"); return
        total = int(self.burst_count_spin.value())
        index = total - self._burst_remaining
        folder = self._ensure_save_folder()
        pattern = self.le_pattern.text().strip() or "burst_{timestamp}_{index:03d}.jpg"
        fname = os.path.join(folder, self._fill_pattern(pattern, index=index, count=total))
        cv2.imwrite(fname, self.current_frame)
        self._burst_remaining -= 1
        self.lbl_burst_status.setText(f"{self._burst_remaining} remaining"); self.status.setText(f"Saved {fname}")
        if self._burst_remaining <= 0:
            self._finish_burst()

    def _finish_burst(self):
        if self._burst_timer:
            self._burst_timer.stop(); self._burst_timer.deleteLater(); self._burst_timer = None
        self.btn_start_burst.setEnabled(True); self.btn_start.setEnabled(True); self.btn_stop.setEnabled(True); self.btn_save.setEnabled(True if self.current_frame is not None else False); self.btn_browse.setEnabled(True)
        self.lbl_burst_status.setText("Burst complete"); self.status.setText("Burst capture finished.")

    # ---- single save ----
    def _on_save_frame(self):
        if self.current_frame is None:
            self.status.setText("No frame to save."); return
        folder = self._ensure_save_folder()
        pattern = self.le_pattern.text().strip() or "capture_{timestamp}.jpg"
        fname = os.path.join(folder, self._fill_pattern(pattern, index=0, count=1))
        cv2.imwrite(fname, self.current_frame)
        self.status.setText(f"Saved {fname}")

    # ---- save folder UI ----
    def _ensure_save_folder(self) -> str:
        folder = (self.le_save_folder.text().strip() or self.save_folder or os.getcwd())
        folder = os.path.expanduser(folder)
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            self.status.setText(f"Failed to create folder '{folder}': {e}")
            folder = os.getcwd()
        self.save_folder = os.path.abspath(folder); self.le_save_folder.setText(self.save_folder)
        return self.save_folder

    def _on_browse_folder(self):
        start = self.le_save_folder.text().strip() or self.save_folder or os.getcwd(); start = os.path.expanduser(start)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder to save images", start)
        if folder:
            self.save_folder = os.path.abspath(folder); self.le_save_folder.setText(self.save_folder); self.status.setText(f"Save folder set to: {self.save_folder}")

    def _on_open_folder(self):
        folder = self.le_save_folder.text().strip() or self.save_folder or os.getcwd(); folder = os.path.expanduser(folder)
        if not os.path.isdir(folder):
            self.status.setText("Folder does not exist."); return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            elif sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore
            else:
                self.status.setText("Open folder not supported on this platform.")
        except Exception as e:
            self.status.setText(f"Failed to open folder: {e}")

    # ---- pattern filling helper ----
    _token_re = re.compile(r'\{([^:{}]+)(?::([^{}]+))?\}')

    def _fill_pattern(self, pattern: str, index: int = 0, count: int = 1) -> str:
        """
        Replace tokens in `pattern`.
        Supports:
          {timestamp} or {timestamp:%Y%m%d_%H%M%S}
          {index} or {index:03d}
          {count} or {count:03d}
          {exposure} optionally with format
          {gain}
        Unknown tokens are left as-is (but braces are removed).
        """
        def repl(m):
            name = m.group(1)
            fmt = m.group(2)
            name_l = name.lower()
            try:
                if name_l == 'timestamp':
                    fmt_str = fmt if fmt else "%Y%m%d_%H%M%S"
                    return datetime.datetime.now().strftime(fmt_str)
                if name_l == 'index':
                    if fmt:
                        # e.g. fmt "03d" -> format(index, "03d")
                        try:
                            return format(index, fmt)
                        except Exception:
                            return str(index)
                    return str(index)
                if name_l == 'count':
                    if fmt:
                        try:
                            return format(count, fmt)
                        except Exception:
                            return str(count)
                    return str(count)
                if name_l in ('exposure', 'exposure_time_absolute'):
                    v = self._get_ctrl_value_by_names(['exposure_time_absolute', 'exposure_absolute', 'exposure'])
                    if v is None: return "NA"
                    if fmt:
                        try: return format(int(v), fmt)
                        except: return str(v)
                    return str(v)
                if name_l == 'gain':
                    v = self._get_ctrl_value_by_names(['gain','analogue_gain'])
                    if v is None: return "NA"
                    if fmt:
                        try: return format(int(v), fmt)
                        except: return str(v)
                    return str(v)
                # fallback: if there's a control with this name directly, try to get it
                v = self._get_ctrl_value_by_names([name_l])
                if v is not None:
                    if fmt:
                        try: return format(int(v), fmt)
                        except: return str(v)
                    return str(v)
                # unknown token -> return empty or keep raw? keep a safe placeholder:
                return ""
            except Exception:
                return ""

        result = self._token_re.sub(repl, pattern)
        # sanitize filename (optional): remove characters disallowed in filenames on many platforms
        # keep it simple: replace trailing / or backslash
        result = result.replace('/', '_').replace('\\','_')
        return result

    def _get_ctrl_value_by_names(self, candidate_names: List[str]) -> Optional[Any]:
        """
        Attempt to read value from controls UI (self._ctrl_widgets).
        candidate_names: list of normalized keys to try in order.
        Returns int/str or None.
        """
        for n in candidate_names:
            if n in self._ctrl_widgets:
                meta = self._ctrl_widgets[n]
                widget = meta.get('widget')
                # if integer slider with val_label
                if 'val_label' in meta:
                    try:
                        return int(meta['val_label'].text())
                    except Exception:
                        return meta['val_label'].text()
                # slider only
                if isinstance(widget, QtWidgets.QSlider):
                    try: return widget.value() + meta.get('min',0)
                    except: return widget.value()
                if isinstance(widget, QtWidgets.QCheckBox):
                    return 1 if widget.isChecked() else 0
                if isinstance(widget, QtWidgets.QComboBox):
                    return widget.currentData()
                # else no widget readable
        return None

    # ---- utility ----
    @staticmethod
    def _devpath_to_index(devpath: str) -> Optional[int]:
        m = re.search(r'video(\d+)', devpath)
        if m:
            return int(m.group(1))
        try:
            return int(devpath)
        except Exception:
            return None

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

