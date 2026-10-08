"""
Simple NSF player for the GUI (play/pause and back to start only).

Uses libgme (src/dll/libgme.dll, the same build as morokoshi) through ctypes and streams
its output to sounddevice. The emulator is run directly in the audio callback, so there is
no pre-rendering: playback starts at once and pausing keeps the emulator state.
"""

import ctypes as _ct
import os
import sys
import threading

import numpy as np
import sounddevice as sd

SR = 44100

_lib = None


def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def gme_load():
    """Load libgme.dll (dll/ next to the script or exe first). Returns None when not found."""
    global _lib
    if _lib is not None:
        return _lib
    base = _app_dir()
    dirs = [os.path.join(base, "dll"), base]
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
    for path in [os.path.join(d, "libgme.dll") for d in dirs] + ["libgme.dll"]:
        try:
            lib = _ct.CDLL(path)
        except OSError:
            continue
        lib.gme_open_data.restype = _ct.c_char_p
        lib.gme_open_data.argtypes = [_ct.c_void_p, _ct.c_long, _ct.POINTER(_ct.c_void_p), _ct.c_int]
        lib.gme_delete.restype = None
        lib.gme_delete.argtypes = [_ct.c_void_p]
        lib.gme_start_track.restype = _ct.c_char_p
        lib.gme_start_track.argtypes = [_ct.c_void_p, _ct.c_int]
        lib.gme_play.restype = _ct.c_char_p
        lib.gme_play.argtypes = [_ct.c_void_p, _ct.c_int, _ct.c_void_p]
        lib.gme_track_ended.restype = _ct.c_int
        lib.gme_track_ended.argtypes = [_ct.c_void_p]
        lib.gme_clear_blip_buffer.restype = None
        lib.gme_clear_blip_buffer.argtypes = [_ct.c_void_p]
        _lib = lib
        return lib
    return None


class NsfPlayer:
    def __init__(self):
        self.lib = gme_load()
        self.emu = None
        self.path = None
        self.track = None          # 1-based
        self.playing = False
        self.samples = 0           # samples played since the start of the track
        self.error = ""
        self._stream = None
        self._lock = threading.Lock()

    @property
    def available(self):
        return self.lib is not None

    def position_sec(self):
        return self.samples / SR

    def load(self, path, track):
        """Open the NSF and start the track (paused). Returns False on error (see .error)."""
        self.close()
        if not self.lib:
            self.error = "libgme.dll not found"
            return False
        with open(path, "rb") as f:
            raw = f.read()
        self._raw = _ct.create_string_buffer(raw, len(raw))   # kept alive while the emulator lives
        emu = _ct.c_void_p()
        err = self.lib.gme_open_data(self._raw, len(raw), _ct.byref(emu), SR)
        if err:
            self.error = err.decode(errors="replace")
            return False
        with self._lock:
            self.emu = emu
            self.path = path
            self.track = track
        return self._start_track()

    def _start_track(self):
        with self._lock:
            err = self.lib.gme_start_track(self.emu, self.track - 1)
            if err:
                self.error = err.decode(errors="replace")
                return False
            self.lib.gme_clear_blip_buffer(self.emu)   # drop the INIT click at the start
            self.samples = 0
        return True

    def play(self):
        if not self.emu or self.playing:
            return
        self.playing = True
        self._stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32",
                                       blocksize=2048, callback=self._callback)
        self._stream.start()

    def pause(self):
        self.playing = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            if self.emu and self.lib.gme_track_ended(self.emu):
                self._start_track()
            self.play()

    def poll(self):
        """Call periodically from the GUI: closes the stream after the track has ended."""
        if not self.playing and self._stream:
            self.pause()

    def rewind(self):
        if self.emu:
            self._start_track()

    def close(self):
        self.pause()
        with self._lock:
            if self.emu:
                self.lib.gme_delete(self.emu)
            self.emu = None
            self.path = None
            self.track = None
            self.samples = 0

    def _callback(self, outdata, frames, t, status):
        with self._lock:
            if not self.playing or not self.emu:
                outdata[:] = 0
                return
            buf = (_ct.c_short * (frames * 2))()
            err = self.lib.gme_play(self.emu, frames * 2, buf)
            ended = err is not None or self.lib.gme_track_ended(self.emu)
            self.samples += frames
        outdata[:] = np.frombuffer(buf, dtype=np.int16).reshape(frames, 2) / 32768.0
        if ended:
            # stopping the stream from its own callback is not allowed; the GUI timer closes it
            self.playing = False
