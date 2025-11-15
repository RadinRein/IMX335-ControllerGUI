#This code is the property of RadinRein
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

# optional dependency for GIF writing
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False

# ----------------- helpers for subprocess / v4l2-ctl -----------------
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
    kv_re = re.compile(r'([a-zA-Z0-9_]+)=([^\s,]+)')
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

# ----------------- camera worker -----------------
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

# ----------------- GIF saver worker (QThread) -----------------
class GifSaverWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)  # current, total
    finished = QtCore.pyqtSignal(bool, str)  # success, message

    def __init__(self, frames: List[np.ndarray], path: str, fps: int = 15, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.path = path
        self.fps = fps

    def run(self):
        try:
            total = len(self.frames)
            with imageio.get_writer(self.path, fps=self.fps) as writer:
                for i, frame in enumerate(self.frames):
                    writer.append_data(frame)
                    self.progress.emit(i + 1, total)
            self.finished.emit(True, f"GIF saved: {self.path}")
        except Exception as e:
            self.finished.emit(False, f"Failed to save GIF: {e}")

# ----------------- MainWindow -----------------
class MainWindow(QtWidgets.QMainWindow):
    SETTINGS_ORG = "usbcam_app"
    SETTINGS_APP = "usbcam_gui"

    def __init__(self):
        super().__init__()
        self._init_size_from_screen()
        self.setWindowTitle("IMX335 Camera GUI (resizable)")
        self._build_ui()

        # state
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self._on_timer)
        self.worker: Optional[CameraWorker] = None
        self.current_frame: Optional[np.ndarray] = None
        self._ctrl_widgets: Dict[str, Dict[str, Any]] = {}
        self._last_ctrls: Dict[str, Dict[str, Any]] = {}
        self._burst_timer: Optional[QtCore.QTimer] = None
        self._burst_remaining = 0
        self._burst_prefix = ""

        # recording state
        self.recording = False
        self.rec_format = "mp4"
        self.rec_writer = None
        self.rec_frames: List[np.ndarray] = []
        self.rec_path = ""
        self.rec_start_time = None
        self.rec_timer = QtCore.QTimer(); self.rec_timer.setInterval(500); self.rec_timer.timeout.connect(self._on_record_tick)
        self.gif_worker: Optional[GifSaverWorker] = None

        # QSettings
        self.settings = QtCore.QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._load_settings()

        # defaults & restore
        self.save_folder = os.path.expanduser(self.settings.value("save_folder", os.getcwd()))
        self.le_save_folder.setText(self.save_folder)
        self.le_pattern.setText(self.settings.value("file_pattern", "capture_{timestamp:%Y%m%d_%H%M%S}.jpg"))
        self.le_record_pattern.setText(self.settings.value("record_pattern", "recording_{timestamp:%Y%m%d_%H%M%S}.mp4"))
        self.rec_format_combo.setCurrentText(self.settings.value("record_format", "mp4"))
        self.fps_spin.setValue(int(self.settings.value("fps", 30)))

        self._populate_devices()

    def _init_size_from_screen(self):
        """
        Set an initial window size based on available screen geometry.
        Default fraction is 0.85 (85%) of available width and height.
        """
        app = QtWidgets.QApplication.instance()
        # In some contexts (tests), QApplication may not exist yet; guard.
        if app is None:
            return
        screen = app.primaryScreen()
        if not screen:
            return
        avail = screen.availableGeometry()
        frac = 0.85
        w = max(800, int(avail.width() * frac))
        h = max(600, int(avail.height() * frac))
        self.resize(w, h)
        # Set a reasonable minimum so layout doesn't collapse
        self.setMinimumSize(700, 500)

    # ----------------- UI -----------------
    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        main_layout = QtWidgets.QVBoxLayout(w)

        # device/resolution controls
        top = QtWidgets.QHBoxLayout()
        self.device_combo = QtWidgets.QComboBox(); self.device_combo.currentIndexChanged.connect(self._on_device_selected)
        top.addWidget(QtWidgets.QLabel("Device:")); top.addWidget(self.device_combo)
        self.res_combo = QtWidgets.QComboBox(); top.addWidget(QtWidgets.QLabel("Resolution:")); top.addWidget(self.res_combo)
        self.btn_refresh = QtWidgets.QPushButton("Refresh"); self.btn_refresh.clicked.connect(self._populate_devices); top.addWidget(self.btn_refresh)
        self.btn_start = QtWidgets.QPushButton("Start Preview"); self.btn_start.clicked.connect(self._on_start_clicked); top.addWidget(self.btn_start)
        self.btn_stop = QtWidgets.QPushButton("Stop Preview"); self.btn_stop.clicked.connect(self._on_stop_clicked); self.btn_stop.setEnabled(False); top.addWidget(self.btn_stop)
        main_layout.addLayout(top)

        # Instead of simple HBox, use a QSplitter so preview and control side can be resized by user
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # preview area (left)
        preview_container = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_container)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # Make video_label expand/contract with the splitter/window
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self.video_label)
        splitter.addWidget(preview_container)

        # right-side controls (put all right column widgets inside this widget)
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        # Frame Rate group
        fr_group = QtWidgets.QGroupBox("Frame Rate Control")
        fr_layout = QtWidgets.QHBoxLayout(fr_group)
        fr_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spin = QtWidgets.QSpinBox(); self.fps_spin.setRange(1, 240); self.fps_spin.setValue(30); fr_layout.addWidget(self.fps_spin)
        self.btn_set_fps = QtWidgets.QPushButton("Set FPS"); self.btn_set_fps.clicked.connect(self._on_set_fps); fr_layout.addWidget(self.btn_set_fps)
        self.lbl_fps_status = QtWidgets.QLabel(""); fr_layout.addWidget(self.lbl_fps_status, stretch=1)
        right_layout.addWidget(fr_group)

        # Burst group
        burst_group = QtWidgets.QGroupBox("Burst Capture")
        burst_layout = QtWidgets.QGridLayout(burst_group)
        burst_layout.addWidget(QtWidgets.QLabel("Count:"), 0, 0); self.burst_count_spin = QtWidgets.QSpinBox(); self.burst_count_spin.setRange(1, 1000); self.burst_count_spin.setValue(5); burst_layout.addWidget(self.burst_count_spin, 0, 1)
        burst_layout.addWidget(QtWidgets.QLabel("Interval (ms):"), 1, 0); self.burst_interval_spin = QtWidgets.QSpinBox(); self.burst_interval_spin.setRange(10, 60000); self.burst_interval_spin.setValue(200); burst_layout.addWidget(self.burst_interval_spin, 1, 1)
        self.btn_start_burst = QtWidgets.QPushButton("Start Burst Capture"); self.btn_start_burst.clicked.connect(self._on_start_burst); burst_layout.addWidget(self.btn_start_burst, 2, 0, 1, 2)
        self.lbl_burst_status = QtWidgets.QLabel(""); burst_layout.addWidget(self.lbl_burst_status, 3, 0, 1, 2)
        right_layout.addWidget(burst_group)

        # Save folder & pattern
        save_group = QtWidgets.QGroupBox("Save Folder & Filename Pattern")
        save_layout = QtWidgets.QVBoxLayout(save_group)
        row = QtWidgets.QHBoxLayout()
        self.le_save_folder = QtWidgets.QLineEdit(); self.le_save_folder.setPlaceholderText("Select folder to save images"); row.addWidget(self.le_save_folder)
        self.btn_browse = QtWidgets.QPushButton("Browse"); self.btn_browse.clicked.connect(self._on_browse_folder); row.addWidget(self.btn_browse)
        self.btn_open_folder = QtWidgets.QPushButton("Open"); self.btn_open_folder.clicked.connect(self._on_open_folder); row.addWidget(self.btn_open_folder)
        save_layout.addLayout(row)
        save_layout.addWidget(QtWidgets.QLabel("Filename pattern:"))
        self.le_pattern = QtWidgets.QLineEdit()
        self.le_pattern.setToolTip("Use tokens like {timestamp}, {timestamp:%Y%m%d_%H%M%S}, {index}, {index:03d}, {count}, {exposure}, {gain}")
        save_layout.addWidget(self.le_pattern)
        self.lbl_pattern_example = QtWidgets.QLabel("Example: capture_{timestamp}_{index:03d}.jpg")
        save_layout.addWidget(self.lbl_pattern_example)
        right_layout.addWidget(save_group)

        # dynamic controls area (scroll)
        self.ctrl_scroll = QtWidgets.QScrollArea(); self.ctrl_scroll.setWidgetResizable(True)
        self.ctrl_container = QtWidgets.QWidget(); self.ctrl_layout = QtWidgets.QVBoxLayout(self.ctrl_container); self.ctrl_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.ctrl_scroll.setWidget(self.ctrl_container)
        # let the scroll area expand vertically as the window grows
        self.ctrl_scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.ctrl_scroll, stretch=1)

        # actions: save frame + reset to hardware defaults
        actions_row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Frame"); self.btn_save.clicked.connect(self._on_save_frame); self.btn_save.setEnabled(False)
        actions_row.addWidget(self.btn_save)
        self.btn_reset_defaults = QtWidgets.QPushButton("Reset to hardware defaults"); self.btn_reset_defaults.clicked.connect(self._on_reset_to_defaults)
        actions_row.addWidget(self.btn_reset_defaults)
        right_layout.addLayout(actions_row)

        # Recording group (enhanced)
        rec_group = QtWidgets.QGroupBox("Recording")
        rec_layout = QtWidgets.QGridLayout(rec_group)
        rec_layout.addWidget(QtWidgets.QLabel("Format:"), 0, 0)
        self.rec_format_combo = QtWidgets.QComboBox()
        self.rec_format_combo.addItem("mp4"); self.rec_format_combo.addItem("avi"); self.rec_format_combo.addItem("gif")
        rec_layout.addWidget(self.rec_format_combo, 0, 1)
        rec_layout.addWidget(QtWidgets.QLabel("Filename pattern:"), 1, 0)
        self.le_record_pattern = QtWidgets.QLineEdit()
        rec_layout.addWidget(self.le_record_pattern, 1, 1)
        self.btn_record = QtWidgets.QPushButton("Start Recording")
        self.btn_record.clicked.connect(self._on_record_toggle)
        rec_layout.addWidget(self.btn_record, 2, 0, 1, 2)
        self.lbl_record_status = QtWidgets.QLabel("")
        rec_layout.addWidget(self.lbl_record_status, 3, 0, 1, 2)
        hrow = QtWidgets.QHBoxLayout()
        self.lbl_rec_duration = QtWidgets.QLabel("Duration: 00:00:00")
        self.lbl_rec_filesize = QtWidgets.QLabel("Size: 0 B")
        self.lbl_rec_est = QtWidgets.QLabel("Est: 0 B")
        hrow.addWidget(self.lbl_rec_duration); hrow.addWidget(self.lbl_rec_filesize); hrow.addWidget(self.lbl_rec_est)
        rec_layout.addLayout(hrow, 4, 0, 1, 2)
        right_layout.addWidget(rec_group)

        right_layout.addStretch(1)
        splitter.addWidget(right_widget)

        # set initial splitter sizes (left larger than right)
        splitter.setSizes([int(self.width() * 0.65), int(self.width() * 0.35)])

        main_layout.addWidget(splitter, stretch=1)

        # bottom status
        self.status = QtWidgets.QLabel("")
        main_layout.addWidget(self.status)

    # ----------------- settings -----------------
    def _load_settings(self):
        s = QtCore.QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        return s

    def _save_settings(self):
        s = self.settings
        s.setValue("save_folder", self.le_save_folder.text().strip() or "")
        s.setValue("file_pattern", self.le_pattern.text().strip() or "")
        s.setValue("record_pattern", self.le_record_pattern.text().strip() or "")
        s.setValue("record_format", self.rec_format_combo.currentText())
        s.setValue("fps", int(self.fps_spin.value()))
        s.sync()

    # ----------------- device enumeration -----------------
    def _populate_devices(self):
        self.device_combo.clear()
        if not v4l2_available():
            self.device_combo.addItem("/dev/video0 (v4l2-ctl not found)", "/dev/video0")
            self.status.setText("v4l2-ctl not found; install v4l-utils for full functionality.")
            return
        devs = list_v4l2_devices()
        if not devs:
            self.device_combo.addItem("/dev/video0 (no devices found)", "/dev/video0")
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
                for fcc, w, h in fmts:
                    self.res_combo.addItem(f"{w}x{h} ({fcc})", (w, h))
            else:
                self.res_combo.addItem("1280x720", (1280, 720))
        else:
            self.res_combo.addItem("1280x720", (1280, 720))
        self._build_controls_for_device(dev)

    # ----------------- dynamic controls -----------------
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
        raw = run_cmd(["v4l2-ctl", "--device", dev, "--list-ctrls"])
        if not raw:
            self.status.setText("No controls returned by v4l2-ctl.")
            return
        ctrls = parse_ctrls(raw)
        if not ctrls:
            self.status.setText("Failed to parse controls.")
            return
        self._last_ctrls = ctrls

        preferred_order = [
            'brightness', 'contrast', 'saturation', 'hue',
            'white_balance_automatic', 'white_balance_temperature', 'gamma', 'power_line_frequency',
            'sharpness', 'backlight_compensation',
            'auto_exposure', 'exposure_time_absolute', 'exposure_dynamic_framerate',
            'pan_absolute', 'tilt_absolute', 'zoom_absolute'
        ]

        def add_control(norm_key, info):
            disp_name = info.get('name', norm_key)
            typ = info.get('type', '').lower()
            row_w = QtWidgets.QWidget(); row = QtWidgets.QHBoxLayout(row_w); row.setContentsMargins(6, 4, 6, 4)
            lbl = QtWidgets.QLabel(disp_name); lbl.setMinimumWidth(200); row.addWidget(lbl)

            meta = {'orig_name': info.get('name', norm_key), 'type': typ}
            inactive = False
            if 'flags' in info and str(info.get('flags')).lower() == 'inactive':
                inactive = True
            if 'inactive' in str(info.get('raw', '')).lower():
                inactive = True

            if 'int' in typ:
                minv = int(info.get('min', 0)); maxv = int(info.get('max', minv + 1000)); cur = info.get('value', minv)
                try:
                    cur_i = int(cur)
                except:
                    cur_i = minv
                slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); slider.setMinimum(0); slider.setMaximum(maxv - minv); slider.setValue(cur_i - minv); slider.setEnabled(not inactive)
                val_spin = QtWidgets.QSpinBox()
                val_spin.setRange(minv, maxv)
                val_spin.setValue(cur_i)
                val_spin.setMinimumWidth(90)
                val_spin.setEnabled(not inactive)

                # make spin expand vertically less; layout handles width
                row.addWidget(slider, stretch=1); row.addWidget(val_spin)
                self.ctrl_layout.addWidget(row_w)
                meta.update({'min': minv, 'max': maxv, 'widget': slider, 'val_label': val_spin, 'inactive': inactive})

                def on_slider(v, nk=norm_key):
                    m = self._ctrl_widgets.get(nk)
                    if not m: return
                    actual = int(v + m['min'])
                    spin: QtWidgets.QSpinBox = m['val_label']
                    if spin.value() != actual:
                        spin.blockSignals(True)
                        spin.setValue(actual)
                        spin.blockSignals(False)
                    ok = set_ctrl(dev, m['orig_name'], actual)
                    if not ok:
                        self.status.setText(f"Failed to set {m['orig_name']} -> {actual}")
                slider.valueChanged.connect(on_slider)

                def on_spin(v, nk=norm_key):
                    m = self._ctrl_widgets.get(nk)
                    if not m: return
                    try:
                        val = int(v)
                    except Exception:
                        return
                    slider_widget: QtWidgets.QSlider = m['widget']
                    desired_slider = val - m['min']
                    if slider_widget.value() != desired_slider:
                        slider_widget.blockSignals(True)
                        slider_widget.setValue(desired_slider)
                        slider_widget.blockSignals(False)
                    ok = set_ctrl(dev, m['orig_name'], val)
                    if not ok:
                        self.status.setText(f"Failed to set {m['orig_name']} -> {val}")
                val_spin.valueChanged.connect(on_spin)

                self._ctrl_widgets[norm_key] = meta

            elif 'bool' in typ:
                cur = info.get('value', 0)
                try:
                    cur_b = bool(int(cur))
                except:
                    cur_b = bool(cur)
                chk = QtWidgets.QCheckBox(); chk.setChecked(cur_b); chk.setEnabled(not inactive)
                row.addWidget(chk); row.addStretch(1); self.ctrl_layout.addWidget(row_w)
                meta.update({'widget': chk, 'inactive': inactive})
                def on_chk(state, nk=norm_key):
                    m = self._ctrl_widgets.get(nk)
                    if not m: return
                    val = 1 if state else 0
                    ok = set_ctrl(dev, m['orig_name'], val)
                    if not ok:
                        self.status.setText(f"Failed to set {m['orig_name']} -> {val}")
                    else:
                        if nk in ('white_balance_automatic', 'auto_exposure'):
                            QtCore.QTimer.singleShot(200, lambda: self._build_controls_for_device(dev))
                chk.stateChanged.connect(on_chk)
                self._ctrl_widgets[norm_key] = meta

            elif 'menu' in typ:
                minv = int(info.get('min', 0)); maxv = int(info.get('max', minv)); cur = int(info.get('value', minv)); value_label = info.get('value_label')
                combo = QtWidgets.QComboBox()
                for v in range(minv, maxv + 1):
                    entry = str(v)
                    if v == cur and value_label:
                        entry = f"{v} ({value_label})"
                    combo.addItem(entry, v)
                combo.setCurrentIndex(cur - minv); combo.setEnabled(not inactive)
                row.addWidget(combo, stretch=1); row.addStretch(0); self.ctrl_layout.addWidget(row_w)
                meta.update({'min': minv, 'max': maxv, 'widget': combo, 'inactive': inactive})
                def on_combo(idx, nk=norm_key):
                    m = self._ctrl_widgets.get(nk)
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
                row.addWidget(QtWidgets.QLabel(str(info.get('value', '')))); self.ctrl_layout.addWidget(row_w)
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

    # ----------------- preview start/stop -----------------
    def _on_start_clicked(self):
        dev = self.device_combo.currentData(); res = self.res_combo.currentData()
        if not dev or not res:
            self.status.setText("Select a device and resolution first."); return
        w, h = res
        idx = self._devpath_to_index(dev)
        if idx is None:
            self.status.setText(f"Cannot parse device index from {dev}"); return
        self.worker = CameraWorker(dev_index=idx, width=w, height=h)
        self.worker.frame_ready.connect(self._on_frame); self.worker.error.connect(self._on_worker_error)
        self.worker.start(); self.timer.start(30)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_save.setEnabled(True)
        self.status.setText("Preview started.")

    def _on_stop_clicked(self):
        if self.recording:
            self._stop_recording()

        if self.timer.isActive(): self.timer.stop()
        if self.worker: self.worker.stop(); self.worker = None
        self.video_label.clear(); self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_save.setEnabled(False)
        self.status.setText("Preview stopped.")

    def _on_timer(self):
        if self.worker: self.worker.tick()

    def _on_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape; bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix); self.current_frame = frame

        # handle recording
        if self.recording:
            fmt = self.rec_format.lower()
            if fmt in ('mp4', 'avi'):
                if self.rec_writer is not None:
                    try:
                        self.rec_writer.write(frame)
                    except Exception as e:
                        self.status.setText(f"Recording write error: {e}")
            elif fmt == 'gif':
                if _HAS_IMAGEIO:
                    self.rec_frames.append(rgb.copy())
                else:
                    self.status.setText("imageio not available: cannot record GIF. Stop recording.")
                    self._stop_recording()

    def _on_worker_error(self, text: str):
        self.status.setText("Camera error: " + str(text)); self._on_stop_clicked()

    # ----------------- FPS -----------------
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
        self._save_settings()

    # ----------------- burst capture -----------------
    def _on_start_burst(self):
        if self.worker is None or self.worker.cap is None:
            self.status.setText("Start preview before burst capture."); return
        if self.current_frame is None:
            self.status.setText("No frame available yet."); return
        count = int(self.burst_count_spin.value()); interval = int(self.burst_interval_spin.value())
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._burst_prefix = f"burst_{ts}"
        self._burst_remaining = count
        self.btn_start_burst.setEnabled(False); self.btn_start.setEnabled(False); self.btn_stop.setEnabled(False); self.btn_save.setEnabled(False); self.btn_browse.setEnabled(False)
        self.status.setText(f"Starting burst: {count} frames every {interval} ms"); self.lbl_burst_status.setText(f"{self._burst_remaining} remaining")
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

    # ----------------- single save -----------------
    def _on_save_frame(self):
        if self.current_frame is None:
            self.status.setText("No frame to save."); return
        folder = self._ensure_save_folder()
        pattern = self.le_pattern.text().strip() or "capture_{timestamp}.jpg"
        fname = os.path.join(folder, self._fill_pattern(pattern, index=0, count=1))
        cv2.imwrite(fname, self.current_frame)
        self.status.setText(f"Saved {fname}")

    # ----------------- recording -----------------
    def _on_record_toggle(self):
        if not self.recording:
            if self.worker is None or self.worker.cap is None:
                self.status.setText("Start preview before recording."); return
            fmt = self.rec_format_combo.currentText().lower()
            pat = self.le_record_pattern.text().strip() or f"recording_{{timestamp:%Y%m%d_%H%M%S}}.{fmt}"
            folder = self._ensure_save_folder()
            fname = os.path.join(folder, self._fill_pattern(pat, index=0, count=1))
            if not fname.lower().endswith(f".{fmt}"):
                fname = fname + f".{fmt}"
            w, h = int(self.res_combo.currentData()[0]), int(self.res_combo.currentData()[1])
            fps = int(self.fps_spin.value())
            self._save_settings()
            if fmt in ('mp4', 'avi'):
                if fmt == 'mp4':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                try:
                    writer = cv2.VideoWriter(fname, fourcc, float(fps), (w, h))
                    if not writer or not writer.isOpened():
                        self.status.setText(f"Failed to open writer for {fname}")
                        return
                    self.rec_writer = writer
                    self.rec_format = fmt
                    self.rec_path = fname
                    self.recording = True
                    self.rec_start_time = time.time()
                    self.rec_frames = []
                    self.btn_record.setText("Stop Recording")
                    self.lbl_record_status.setText(f"Recording to {fname}")
                    self.lbl_rec_duration.setText("Duration: 00:00:00")
                    self.lbl_rec_filesize.setText("Size: 0 B")
                    self.lbl_rec_est.setText("Est: calculating...")
                    self.rec_timer.start()
                    self.status.setText(f"Recording started: {fname}")
                except Exception as e:
                    self.status.setText(f"Failed to start recording: {e}")
            elif fmt == 'gif':
                if not _HAS_IMAGEIO:
                    self.status.setText("imageio required for GIF recording but not available.")
                    return
                self.rec_format = fmt
                self.rec_frames = []
                self.rec_path = fname
                self.recording = True
                self.rec_start_time = time.time()
                self.btn_record.setText("Stop Recording")
                self.lbl_record_status.setText(f"Recording GIF to {fname}")
                self.lbl_rec_duration.setText("Duration: 00:00:00")
                self.lbl_rec_filesize.setText("Size: 0 B")
                self.lbl_rec_est.setText("Est: calculating...")
                self.rec_timer.start()
                self.status.setText(f"Recording (GIF) started: {fname}")
            else:
                self.status.setText(f"Unsupported format: {fmt}")
        else:
            self._stop_recording()

    def _on_record_tick(self):
        if not self.recording:
            return
        elapsed = int(time.time() - (self.rec_start_time or time.time()))
        h = elapsed // 3600; m = (elapsed % 3600) // 60; s = elapsed % 60
        self.lbl_rec_duration.setText(f"Duration: {h:02d}:{m:02d}:{s:02d}")
        if self.rec_path and os.path.exists(self.rec_path):
            try:
                size = os.path.getsize(self.rec_path)
                self.lbl_rec_filesize.setText(f"Size: {self._sizeof(size)}")
            except Exception:
                self.lbl_rec_filesize.setText("Size: unknown")
        else:
            self.lbl_rec_filesize.setText("Size: 0 B")
        if self.rec_format in ('mp4', 'avi'):
            fps = int(self.fps_spin.value())
            elapsed_f = max(1, elapsed)
            w, h = int(self.res_combo.currentData()[0]), int(self.res_combo.currentData()[1])
            frames_est = elapsed_f * fps
            raw_bytes = frames_est * w * h * 3
            self.lbl_rec_est.setText(f"Est raw: {self._sizeof(raw_bytes)}")
        elif self.rec_format == 'gif':
            self.lbl_rec_est.setText(f"Frames: {len(self.rec_frames)}")

    def _stop_recording(self):
        if not self.recording:
            return
        fmt = self.rec_format.lower()
        self.rec_timer.stop()
        if fmt in ('mp4', 'avi'):
            try:
                if self.rec_writer is not None:
                    self.rec_writer.release()
                    self.rec_writer = None
                    self.status.setText(f"Recording saved: {self.rec_path}")
                    self.lbl_record_status.setText(f"Saved: {self.rec_path}")
            except Exception as e:
                self.status.setText(f"Error closing writer: {e}")
        elif fmt == 'gif':
            if _HAS_IMAGEIO and self.rec_frames:
                self.lbl_record_status.setText("Finalizing GIF...")
                self.gif_worker = GifSaverWorker(self.rec_frames.copy(), self.rec_path, fps=int(self.fps_spin.value()))
                self._gif_progress = QtWidgets.QProgressDialog("Saving GIF...", "Cancel", 0, max(1, len(self.rec_frames)), self)
                self._gif_progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
                self._gif_progress.setAutoClose(False)
                self._gif_progress.setValue(0)
                self.gif_worker.progress.connect(self._on_gif_progress)
                self.gif_worker.finished.connect(self._on_gif_finished)
                self._gif_progress.canceled.connect(self._on_gif_cancel)
                self.gif_worker.start()
            else:
                if not _HAS_IMAGEIO:
                    self.status.setText("imageio not available; GIF not saved.")
                else:
                    self.status.setText("No frames recorded; GIF not saved.")
        if fmt in ('mp4', 'avi'):
            self.recording = False
            self.rec_format = "mp4"
            self.rec_writer = None
            self.rec_frames = []
            self.rec_path = ""
            self.rec_start_time = None
            self.btn_record.setText("Start Recording")
            self.lbl_record_status.setText("")
            self.lbl_rec_duration.setText("Duration: 00:00:00")
            self.lbl_rec_filesize.setText("Size: 0 B")
            self.lbl_rec_est.setText("Est: 0 B")
            self._save_settings()

    def _on_gif_progress(self, current: int, total: int):
        if hasattr(self, "_gif_progress"):
            self._gif_progress.setMaximum(total)
            self._gif_progress.setValue(current)
            self._gif_progress.setLabelText(f"Saving GIF... ({current}/{total})")

    def _on_gif_finished(self, success: bool, message: str):
        if hasattr(self, "_gif_progress"):
            self._gif_progress.close()
        self.status.setText(message)
        self.lbl_record_status.setText(message if success else "GIF save failed")
        self.recording = False
        self.rec_format = "mp4"
        self.rec_writer = None
        self.rec_frames = []
        self.rec_path = ""
        self.rec_start_time = None
        self.btn_record.setText("Start Recording")
        self.lbl_rec_duration.setText("Duration: 00:00:00")
        self.lbl_rec_filesize.setText("Size: 0 B")
        self.lbl_rec_est.setText("Est: 0 B")
        self._save_settings()

    def _on_gif_cancel(self):
        if self.gif_worker and self.gif_worker.isRunning():
            try:
                self.gif_worker.terminate()
            except Exception:
                pass
        if hasattr(self, "_gif_progress"):
            self._gif_progress.close()
        self.status.setText("GIF saving canceled.")
        self.recording = False
        self.rec_frames = []
        self.rec_path = ""
        self.btn_record.setText("Start Recording")
        self.lbl_record_status.setText("GIF canceled")
        self._save_settings()

    # ----------------- save folder -----------------
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
            self._save_settings()

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

    # ----------------- pattern filling -----------------
    _token_re = re.compile(r'\{([^:{}]+)(?::([^{}]+))?\}')

    def _fill_pattern(self, pattern: str, index: int = 0, count: int = 1) -> str:
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
                    v = self._get_ctrl_value_by_names(['gain', 'analogue_gain'])
                    if v is None: return "NA"
                    if fmt:
                        try: return format(int(v), fmt)
                        except: return str(v)
                    return str(v)
                v = self._get_ctrl_value_by_names([name_l])
                if v is not None:
                    if fmt:
                        try: return format(int(v), fmt)
                        except: return str(v)
                    return str(v)
                return ""
            except Exception:
                return ""
        result = self._token_re.sub(repl, pattern)
        result = result.replace('/', '_').replace('\\', '_')
        return result

    def _get_ctrl_value_by_names(self, candidate_names: List[str]) -> Optional[Any]:
        for n in candidate_names:
            if n in self._ctrl_widgets:
                meta = self._ctrl_widgets[n]
                widget = meta.get('widget')
                if 'val_label' in meta:
                    try:
                        spin = meta['val_label']
                        try:
                            return int(spin.value())
                        except Exception:
                            return int(spin.text())
                    except Exception:
                        return meta['val_label'].text() if hasattr(meta['val_label'], 'text') else None
                if isinstance(widget, QtWidgets.QSlider):
                    try: return widget.value() + meta.get('min', 0)
                    except: return widget.value()
                if isinstance(widget, QtWidgets.QCheckBox):
                    return 1 if widget.isChecked() else 0
                if isinstance(widget, QtWidgets.QComboBox):
                    return widget.currentData()
        return None

    # ----------------- reset to defaults -----------------
    def _on_reset_to_defaults(self):
        dev = self.device_combo.currentData()
        if not dev:
            self.status.setText("Select device first.")
            return
        if not self._last_ctrls:
            self.status.setText("No control metadata available to reset.")
            return
        missing_defaults = []
        changed = 0
        for norm_key, info in self._last_ctrls.items():
            if 'default' in info:
                default_val = info['default']
                ctrl_name = info['name']
                ok = set_ctrl(dev, ctrl_name, default_val)
                if ok:
                    if norm_key in self._ctrl_widgets:
                        meta = self._ctrl_widgets[norm_key]
                        if 'val_label' in meta:
                            spin = meta['val_label']
                            spin.blockSignals(True)
                            try: spin.setValue(int(default_val))
                            except: spin.setValue(spin.minimum())
                            spin.blockSignals(False)
                            slider = meta['widget']
                            slider.blockSignals(True)
                            try: slider.setValue(int(default_val) - meta.get('min', 0))
                            except: slider.setValue(0)
                            slider.blockSignals(False)
                        elif isinstance(meta.get('widget'), QtWidgets.QComboBox):
                            combo: QtWidgets.QComboBox = meta['widget']
                            idx_found = 0
                            for i in range(combo.count()):
                                if combo.itemData(i) == default_val:
                                    idx_found = i; break
                            combo.blockSignals(True)
                            combo.setCurrentIndex(idx_found)
                            combo.blockSignals(False)
                        elif isinstance(meta.get('widget'), QtWidgets.QCheckBox):
                            chk: QtWidgets.QCheckBox = meta['widget']
                            chk.blockSignals(True)
                            try: chk.setChecked(bool(int(default_val)))
                            except: chk.setChecked(bool(default_val))
                            chk.blockSignals(False)
                    changed += 1
                else:
                    missing_defaults.append(ctrl_name)
            else:
                missing_defaults.append(info.get('name', norm_key))
        self.status.setText(f"Reset {changed} controls to defaults. Skipped {len(missing_defaults)} (no default info).")
        QtCore.QTimer.singleShot(300, lambda: self._build_controls_for_device(dev))

    # ----------------- utility -----------------
    @staticmethod
    def _devpath_to_index(devpath: str) -> Optional[int]:
        m = re.search(r'video(\d+)', devpath)
        if m:
            return int(m.group(1))
        try:
            return int(devpath)
        except Exception:
            return None

    @staticmethod
    def _sizeof(num: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num < 1024.0:
                return f"{num:3.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} PB"

# ----------------- main -----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
