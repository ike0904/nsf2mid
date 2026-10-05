"""
nsf2mid GUI (Phase 4)

Open an NSF file (button or drag & drop), pick the tracks to convert, set the options and
convert. Track names and lengths are read from an .m3u playlist (nsfe2m3u / GME format)
in the same folder when there is one. Each track is converted with the same code as the CLI
(nsf2mid.convert) in a worker thread; its console output is shown in the log pane.

Usage:
    python nsf2mid_gui.py [FILE.nsf]
    python nsf2mid.py            (no arguments also starts the GUI)
"""

import importlib.util
import subprocess
import sys


def ensure_package(pkg, import_name=None):
    if importlib.util.find_spec(import_name or pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


ensure_package("PyQt6")

import contextlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402

from PyQt6.QtCore import Qt, QThread, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QFont, QColor  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QSplitter, QAbstractItemView,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nsf2mid  # noqa: E402
from nsf import NSFHeader  # noqa: E402

SETTINGS_PATH = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "nsf2mid", "settings.json")
ROWS_PER_BEAT = ("auto", "2", "3", "4", "6", "8", "9", "12", "16", "24")

COL_CHECK, COL_NO, COL_TITLE, COL_LEN, COL_RESULT = range(5)


# ---------------------------------------------------------------- m3u playlist

def _split_m3u_fields(s):
    """Split on commas that are not escaped with a backslash."""
    out, cur, esc = [], "", False
    for ch in s:
        if esc:
            cur += ch
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == ",":
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _read_text(path):
    data = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp932", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def _format_len(s):
    """m3u length ("0:01:11.498", "1:23" or seconds) -> "m:ss"."""
    s = s.strip()
    if not s:
        return ""
    try:
        sec = 0.0
        for part in s.split(":"):
            sec = sec * 60 + float(part)
    except ValueError:
        return ""
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def read_m3u(nsf_path):
    """Return [(track 1-based, title, length)] in playlist order from .m3u files next to the NSF."""
    folder = os.path.dirname(os.path.abspath(nsf_path))
    base = os.path.basename(nsf_path).lower()
    stem = os.path.splitext(base)[0]
    entries, seen = [], set()
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return entries
    # the playlist with the same name first
    names.sort(key=lambda n: os.path.splitext(n.lower())[0] != stem)
    for name in names:
        if not name.lower().endswith(".m3u"):
            continue
        same_stem = os.path.splitext(name.lower())[0] == stem
        for line in _read_text(os.path.join(folder, name)).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "::" not in line:
                continue
            fname, rest = line.split("::", 1)
            if not same_stem and os.path.basename(fname.strip()).lower() != base:
                continue
            f = _split_m3u_fields(rest)
            if len(f) < 2 or f[0].strip().upper() != "NSF":
                continue
            num = f[1].strip()
            try:
                no = int(num[1:], 16) if num.startswith("$") else int(num)
            except ValueError:
                continue
            if no in seen:
                continue
            seen.add(no)
            title = f[2].strip() if len(f) > 2 else ""
            length = _format_len(f[3]) if len(f) > 3 else ""
            entries.append((no, title, length))
    return entries


# ---------------------------------------------------------------- worker

class _LineEmitter:
    """File-like object that forwards complete lines to a callback."""

    def __init__(self, cb):
        self.cb = cb
        self.buf = ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.cb(line)
        return len(s)

    def flush(self):
        if self.buf:
            self.cb(self.buf)
            self.buf = ""


class ConvertWorker(QThread):
    log = pyqtSignal(str)
    track_started = pyqtSignal(int)
    track_done = pyqtSignal(int, str, str, list)   # track, status ("ok"/"error"/"cancel"), summary, outputs

    def __init__(self, nsf_path, tracks, base_argv):
        super().__init__()
        self.nsf_path = nsf_path
        self.tracks = tracks
        self.base_argv = base_argv
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        for t in self.tracks:
            if self._stop:
                self.track_done.emit(t, "cancel", "", [])
                continue
            self.track_started.emit(t)
            argv = [self.nsf_path, "-t", str(t)] + self.base_argv
            self.log.emit(f"=== track {t}: nsf2mid {' '.join(argv[1:])}")
            lines = []

            def out(line):
                lines.append(line)
                self.log.emit(line)

            em = _LineEmitter(out)
            try:
                args = nsf2mid.build_parser().parse_args(argv)
                with contextlib.redirect_stdout(em), contextlib.redirect_stderr(em):
                    code, outputs = nsf2mid.convert(args, lambda: self._stop)
                em.flush()
                if code:
                    self.track_done.emit(t, "error", "", [])
                else:
                    self.track_done.emit(t, "ok", summarize(lines), outputs)
            except nsf2mid.Cancelled:
                em.flush()
                self.log.emit("Cancelled.")
                self.track_done.emit(t, "cancel", "", [])
            except SystemExit:
                em.flush()
                self.log.emit("Error: invalid option")
                self.track_done.emit(t, "error", "", [])
            except Exception:
                em.flush()
                self.log.emit(traceback.format_exc())
                self.track_done.emit(t, "error", "", [])


def summarize(lines):
    """One-line result for the track table, from the CLI output."""
    parts = []
    segs = [l.strip() for l in lines if l.strip().startswith("from ")]
    if segs:
        bpms = []
        for l in segs:
            m = re.search(r"([\d.]+) BPM", l)
            if m:
                bpms.append(float(m.group(1)))
        meters = []
        for l in segs:
            m = re.search(r"\)\s+(\d+/\d+)", l)
            if m and m.group(1) not in meters:
                meters.append(m.group(1))
        if bpms:
            lo, hi = min(bpms), max(bpms)
            parts.append(f"{lo:.0f} BPM" if round(lo) == round(hi) else f"{lo:.0f}-{hi:.0f} BPM")
        if meters:
            parts.append("→".join(meters))
    else:
        parts.append("テンポ検出不可")
    for l in lines:
        if l.startswith("Loop:"):
            m = re.search(r"starts at ([\d.]+) s, length ([\d.]+) s", l)
            if m:
                parts.append(f"ループ {float(m.group(1)):.1f}+{float(m.group(2)):.1f} 秒")
            elif "song ended" in l:
                m = re.search(r"song length ([\d.]+)", l)
                parts.append(f"終了 {float(m.group(1)):.1f} 秒" if m else "終了")
            else:
                parts.append("ループ判定不能")
    return " / ".join(parts)


# ---------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self, initial=None):
        super().__init__()
        self.setWindowTitle(f"nsf2mid {nsf2mid.APP_VERSION}")
        self.setAcceptDrops(True)
        self.resize(980, 760)
        self.nsf_path = None
        self.header = None
        self.worker = None
        self.outputs = {}          # track -> output files
        self.last_outdir = None
        self.m3u_tracks = set()
        self.settings = self._load_settings()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # file row
        row = QHBoxLayout()
        row.addWidget(QLabel("NSF ファイル:"))
        self.ed_file = QLineEdit()
        self.ed_file.setReadOnly(True)
        self.ed_file.setPlaceholderText("「開く」またはウィンドウへドラッグ＆ドロップ")
        row.addWidget(self.ed_file, 1)
        b = QPushButton("開く...")
        b.clicked.connect(self.on_open)
        row.addWidget(b)
        root.addLayout(row)

        self.lb_info = QLabel("")
        self.lb_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lb_info.setWordWrap(True)
        root.addWidget(self.lb_info)

        split = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(split, 1)

        # tracks + options
        top = QWidget()
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)

        tr_box = QGroupBox("曲（チェックした曲を変換。変換後にダブルクリックで MIDI を開く）")
        tr_l = QVBoxLayout(tr_box)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "No.", "曲名（m3u）", "長さ", "結果"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_CHECK, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_NO, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_TITLE, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(COL_LEN, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_RESULT, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(COL_TITLE, 220)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.on_row_double_clicked)
        tr_l.addWidget(self.table)
        brow = QHBoxLayout()
        for text, fn in (("すべて選択", lambda: self.set_checks("all")),
                         ("m3u の曲を選択", lambda: self.set_checks("m3u")),
                         ("すべて解除", lambda: self.set_checks("none"))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            brow.addWidget(b)
        brow.addStretch(1)
        tr_l.addLayout(brow)
        top_l.addWidget(tr_box, 3)

        top_l.addWidget(self._build_options(), 2)
        split.addWidget(top)

        # log
        log_box = QWidget()
        log_l = QVBoxLayout(log_box)
        log_l.setContentsMargins(0, 0, 0, 0)
        crow = QHBoxLayout()
        self.bt_convert = QPushButton("変換")
        self.bt_convert.setMinimumHeight(32)
        self.bt_convert.clicked.connect(self.on_convert)
        self.bt_stop = QPushButton("中止")
        self.bt_stop.setEnabled(False)
        self.bt_stop.clicked.connect(self.on_stop)
        self.bt_folder = QPushButton("出力フォルダを開く")
        self.bt_folder.clicked.connect(self.on_open_folder)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m")
        self.progress.setValue(0)
        crow.addWidget(self.bt_convert)
        crow.addWidget(self.bt_stop)
        crow.addWidget(self.bt_folder)
        crow.addWidget(self.progress, 1)
        log_l.addLayout(crow)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(f)
        log_l.addWidget(self.log, 1)
        split.addWidget(log_box)
        split.setSizes([420, 300])

        self._apply_settings()
        if initial:
            self.load_nsf(initial)

    # ------------------------------------------------------------ options

    def _build_options(self):
        box = QGroupBox("オプション")
        g = QGridLayout(box)
        r = 0

        def add(label, w, tip):
            nonlocal r
            lb = QLabel(label)
            lb.setToolTip(tip)
            w.setToolTip(tip)
            g.addWidget(lb, r, 0)
            g.addWidget(w, r, 1)
            r += 1
            return w

        def check(label, tip):
            nonlocal r
            c = QCheckBox(label)
            c.setToolTip(tip)
            g.addWidget(c, r, 0, 1, 2)
            r += 1
            return c

        self.sp_loops = add("ループ回数", QSpinBox(), "ループ部を何回書き出すか（--loops）")
        self.sp_loops.setRange(1, 16)
        self.sp_seconds = add("最大秒数", QSpinBox(), "エミュレートする最大秒数（-s）。ループ確定・曲終了で自動停止")
        self.sp_seconds.setRange(10, 3600)
        self.sp_seconds.setSingleStep(60)
        self.cb_rows = add("1 拍の行数", QComboBox(), "1 拍あたりの行数（--rows-per-beat）。auto = 自動検出。テンポが倍/半分になる場合に指定")
        self.cb_rows.addItems(ROWS_PER_BEAT)
        self.ed_timesig = add("拍子", QLineEdit(), '区間ごとの拍子（--time-sig）。例 "6/8,4/4"、"-" = 自動。空欄 = 自動')
        self.sp_tri_oct = add("三角波オクターブ", QSpinBox(), "三角波の出力オクターブ移動（--tri-octave）。0 = APU の物理的な音高")
        self.sp_tri_oct.setRange(-3, 3)
        self.ed_programs = add("音色", QLineEdit(), 'P1,P2,TRI の GM 音色番号（--programs）。例 "80,80,38"。空欄 = 音色を出力しない')
        self.ed_drummap = add("ドラム割当", QLineEdit(), 'ドラム割当の上書き（--drum-map）。例 "3:0=42,12:0=36,DMC:E000:129:15=38"')
        self.ck_noloop = check("ループ検出をしない", "最大秒数をそのまま変換（--no-loop）")
        self.ck_tail = check("リバーブ余韻を残す", "余韻もノート長に含める（--keep-tail）")
        self.ck_noquant = check("クオンタイズしない", "グリッドへのスナップをしない（--no-quantize）")
        self.ck_notridrums = check("三角波ドラムを音符のまま残す", "--no-tri-drums")
        self.ck_dump = check("ダンプも出力（writes / frames / tracker）", "--dump")

        orow = QHBoxLayout()
        self.ed_outdir = QLineEdit()
        self.ed_outdir.setPlaceholderText("空欄 = NSF と同じフォルダの nsf2mid_out")
        orow.addWidget(self.ed_outdir, 1)
        b = QPushButton("...")
        b.setFixedWidth(32)
        b.clicked.connect(self.on_pick_outdir)
        orow.addWidget(b)
        g.addWidget(QLabel("出力フォルダ"), r, 0)
        g.addLayout(orow, r, 1)
        r += 1
        b = QPushButton("既定値に戻す")
        b.clicked.connect(self.on_reset_options)
        g.addWidget(b, r, 1, alignment=Qt.AlignmentFlag.AlignRight)
        g.setRowStretch(r + 1, 1)
        return box

    def options_argv(self):
        a = ["-s", str(self.sp_seconds.value()), "--loops", str(self.sp_loops.value())]
        if self.cb_rows.currentText() != "auto":
            a += ["--rows-per-beat", self.cb_rows.currentText()]
        if self.ed_timesig.text().strip():
            a += ["--time-sig", self.ed_timesig.text().strip()]
        if self.sp_tri_oct.value():
            a += ["--tri-octave", str(self.sp_tri_oct.value())]
        if self.ed_programs.text().strip():
            a += ["--programs", self.ed_programs.text().strip()]
        if self.ed_drummap.text().strip():
            a += ["--drum-map", self.ed_drummap.text().strip()]
        for c, flag in ((self.ck_noloop, "--no-loop"), (self.ck_tail, "--keep-tail"),
                        (self.ck_noquant, "--no-quantize"), (self.ck_notridrums, "--no-tri-drums"),
                        (self.ck_dump, "--dump")):
            if c.isChecked():
                a.append(flag)
        if self.ed_outdir.text().strip():
            a += ["-o", self.ed_outdir.text().strip()]
        return a

    # ------------------------------------------------------------ settings

    DEFAULTS = {"loops": 1, "seconds": 600, "rows": "auto", "time_sig": "", "tri_octave": 0,
                "programs": "", "drum_map": "", "no_loop": False, "keep_tail": False,
                "no_quantize": False, "no_tri_drums": False, "dump": False, "outdir": "",
                "last_dir": "", "geometry": None}

    def _load_settings(self):
        s = dict(self.DEFAULTS)
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                s.update(json.load(f))
        except (OSError, ValueError):
            pass
        return s

    def _apply_settings(self, s=None):
        s = s or self.settings
        self.sp_loops.setValue(int(s["loops"]))
        self.sp_seconds.setValue(int(s["seconds"]))
        i = self.cb_rows.findText(str(s["rows"]))
        self.cb_rows.setCurrentIndex(max(0, i))
        self.ed_timesig.setText(s["time_sig"])
        self.sp_tri_oct.setValue(int(s["tri_octave"]))
        self.ed_programs.setText(s["programs"])
        self.ed_drummap.setText(s["drum_map"])
        self.ck_noloop.setChecked(bool(s["no_loop"]))
        self.ck_tail.setChecked(bool(s["keep_tail"]))
        self.ck_noquant.setChecked(bool(s["no_quantize"]))
        self.ck_notridrums.setChecked(bool(s["no_tri_drums"]))
        self.ck_dump.setChecked(bool(s["dump"]))
        self.ed_outdir.setText(s["outdir"])
        if s is self.settings and s.get("geometry"):
            try:
                self.setGeometry(*s["geometry"])
            except TypeError:
                pass

    def _save_settings(self):
        g = self.geometry()
        s = dict(self.settings)
        s.update({"loops": self.sp_loops.value(), "seconds": self.sp_seconds.value(),
                  "rows": self.cb_rows.currentText(), "time_sig": self.ed_timesig.text(),
                  "tri_octave": self.sp_tri_oct.value(), "programs": self.ed_programs.text(),
                  "drum_map": self.ed_drummap.text(), "no_loop": self.ck_noloop.isChecked(),
                  "keep_tail": self.ck_tail.isChecked(), "no_quantize": self.ck_noquant.isChecked(),
                  "no_tri_drums": self.ck_notridrums.isChecked(), "dump": self.ck_dump.isChecked(),
                  "outdir": self.ed_outdir.text(), "geometry": [g.x(), g.y(), g.width(), g.height()]})
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def on_reset_options(self):
        d = dict(self.DEFAULTS)
        d["geometry"] = None
        self._apply_settings(d)

    # ------------------------------------------------------------ file loading

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(".nsf"):
                self.load_nsf(p)
                return
        QMessageBox.warning(self, "nsf2mid", "NSF ファイル（.nsf）をドロップしてください。")

    def on_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "NSF ファイルを開く", self.settings.get("last_dir", ""),
                                           "NSF (*.nsf);;All files (*)")
        if p:
            self.load_nsf(p)

    def load_nsf(self, path):
        if self.worker:
            return
        try:
            with open(path, "rb") as f:
                hdr = NSFHeader(f.read(0x80))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "nsf2mid", f"読み込めませんでした:\n{path}\n{e}")
            return
        self.nsf_path = os.path.abspath(path)
        self.header = hdr
        self.outputs = {}
        self.settings["last_dir"] = os.path.dirname(self.nsf_path)
        self.ed_file.setText(self.nsf_path)
        info = f"<b>{_esc(hdr.title)}</b>　{_esc(hdr.artist)}　{_esc(hdr.copyright)}　" \
               f"曲数 {hdr.total_songs}（開始 {hdr.start_song}）　{'PAL' if hdr.is_pal else 'NTSC'}"
        if hdr.expansion_names:
            info += f"　<span style='color:#c05000'>拡張音源 {', '.join(hdr.expansion_names)}" \
                    f"（未対応: 2A03 の音のみ変換）</span>"
        self.lb_info.setText(info)

        playlist = [e for e in read_m3u(self.nsf_path) if 1 <= e[0] <= hdr.total_songs]
        self.m3u_tracks = {e[0] for e in playlist}
        rows = list(playlist)
        rows += [(t, "", "") for t in range(1, hdr.total_songs + 1) if t not in self.m3u_tracks]
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))
        for i, (no, title, length) in enumerate(rows):
            ck = QTableWidgetItem()
            ck.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            ck.setCheckState(Qt.CheckState.Checked if no == hdr.start_song else Qt.CheckState.Unchecked)
            ck.setData(Qt.ItemDataRole.UserRole, no)
            self.table.setItem(i, COL_CHECK, ck)
            it = QTableWidgetItem(f"{no:3d}")
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, COL_NO, it)
            self.table.setItem(i, COL_TITLE, QTableWidgetItem(title))
            self.table.setItem(i, COL_LEN, QTableWidgetItem(length))
            self.table.setItem(i, COL_RESULT, QTableWidgetItem(""))
        self.progress.setValue(0)
        self.append_log(f"Loaded: {self.nsf_path}" + (f" (playlist: {len(playlist)} tracks)" if playlist else ""))

    def set_checks(self, mode):
        for i in range(self.table.rowCount()):
            it = self.table.item(i, COL_CHECK)
            no = it.data(Qt.ItemDataRole.UserRole)
            on = mode == "all" or (mode == "m3u" and no in self.m3u_tracks)
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def _row_of(self, track):
        for i in range(self.table.rowCount()):
            if self.table.item(i, COL_CHECK).data(Qt.ItemDataRole.UserRole) == track:
                return i
        return -1

    # ------------------------------------------------------------ conversion

    def on_pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "出力フォルダ", self.ed_outdir.text() or self.settings.get("last_dir", ""))
        if d:
            self.ed_outdir.setText(d)

    def output_dir(self):
        if self.ed_outdir.text().strip():
            return self.ed_outdir.text().strip()
        if self.nsf_path:
            return os.path.join(os.path.dirname(self.nsf_path), "nsf2mid_out")
        return None

    def on_convert(self):
        if not self.nsf_path:
            QMessageBox.information(self, "nsf2mid", "NSF ファイルを開いてください。")
            return
        tracks = []
        for i in range(self.table.rowCount()):
            it = self.table.item(i, COL_CHECK)
            if it.checkState() == Qt.CheckState.Checked:
                tracks.append(it.data(Qt.ItemDataRole.UserRole))
        if not tracks:
            QMessageBox.information(self, "nsf2mid", "変換する曲にチェックを入れてください。")
            return
        argv = self.options_argv()
        try:
            nsf2mid.build_parser().parse_args([self.nsf_path] + argv)
        except SystemExit:
            QMessageBox.warning(self, "nsf2mid", "オプションの指定が正しくありません。")
            return
        self._save_settings()
        for t in tracks:
            self._set_result(t, "待機中", None)
        self.progress.setRange(0, len(tracks))
        self.progress.setValue(0)
        self.worker = ConvertWorker(self.nsf_path, tracks, argv)
        self.worker.log.connect(self.append_log)
        self.worker.track_started.connect(lambda t: self._set_result(t, "変換中...", "#0060c0"))
        self.worker.track_done.connect(self.on_track_done)
        self.worker.finished.connect(self.on_worker_finished)
        self._set_running(True)
        self.worker.start()

    def on_stop(self):
        if self.worker:
            self.worker.stop()
            self.bt_stop.setEnabled(False)
            self.append_log("Stopping...")

    def on_track_done(self, track, status, summary, outputs):
        if status == "ok":
            self.outputs[track] = outputs
            self._set_result(track, summary or "完了", None)
            if outputs:
                self.last_outdir = os.path.dirname(outputs[0])
        elif status == "cancel":
            self._set_result(track, "中止", "#808080")
        else:
            self._set_result(track, "エラー（ログ参照）", "#c00000")
        self.progress.setValue(self.progress.value() + 1)

    def on_worker_finished(self):
        self.worker = None
        self._set_running(False)
        self.append_log("Done.\n")

    def _set_running(self, on):
        self.bt_convert.setEnabled(not on)
        self.bt_stop.setEnabled(on)
        self.table.setEnabled(True)

    def _set_result(self, track, text, color):
        i = self._row_of(track)
        if i < 0:
            return
        it = self.table.item(i, COL_RESULT)
        it.setText(text)
        it.setForeground(QColor(color) if color else self.palette().text().color())

    def append_log(self, line):
        self.log.appendPlainText(line)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_row_double_clicked(self, row, col):
        track = self.table.item(row, COL_CHECK).data(Qt.ItemDataRole.UserRole)
        mids = [o for o in self.outputs.get(track, []) if o.lower().endswith(".mid")]
        if mids and os.path.exists(mids[0]):
            _open_path(mids[0])

    def on_open_folder(self):
        d = self.last_outdir or self.output_dir()
        if d and os.path.isdir(d):
            _open_path(d)
        else:
            QMessageBox.information(self, "nsf2mid", "出力フォルダはまだありません。")

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
            self.worker.wait(15000)
        self._save_settings()
        super().closeEvent(e)


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _open_path(p):
    if sys.platform == "win32":
        os.startfile(p)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow(argv[0] if argv else None)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
