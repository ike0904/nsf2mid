"""
2A03 APU state tracker (no audio synthesis).

Replays a timestamped register-write log and models the parts of the APU that
decide *whether* and *how loud* a channel sounds:
  - length counters, envelopes, sweep units (with muting), triangle linear
    counter, frame sequencer (4-step / 5-step), DMC sample playback span.
At each frame boundary a snapshot of every channel is taken.
"""

import math

LENGTH_TABLE = [
    10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
    12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30,
]
NOISE_PERIOD_NTSC = [4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068]
DMC_RATE_NTSC = [428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54]

# Frame sequencer step times in CPU cycles (NTSC)
SEQ4 = [7457, 14913, 22371, 29829]
SEQ4_PERIOD = 29830
SEQ5 = [7457, 14913, 22371, 29829, 37281]
SEQ5_PERIOD = 37282

NOTE_NAMES = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]


def freq_to_note(freq):
    """Return (midi_note_float) for a frequency, or None."""
    if freq <= 0:
        return None
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def note_name(n):
    if n is None or not (0 <= n <= 127.5):
        return "---"
    i = int(round(n))
    return f"{NOTE_NAMES[i % 12]}{i // 12 - 1}"


class Envelope:
    def __init__(self):
        self.start = False
        self.divider = 0
        self.decay = 0
        self.loop = False
        self.const = False
        self.vol = 0

    def clock(self):
        if self.start:
            self.start = False
            self.decay = 15
            self.divider = self.vol
        else:
            if self.divider == 0:
                self.divider = self.vol
                if self.decay > 0:
                    self.decay -= 1
                elif self.loop:
                    self.decay = 15
            else:
                self.divider -= 1

    def output(self):
        return self.vol if self.const else self.decay


class Pulse:
    def __init__(self, ones_complement):
        self.ones = ones_complement
        self.env = Envelope()
        self.duty = 0
        self.halt = False
        self.length = 0
        self.timer = 0
        self.sw_en = False
        self.sw_period = 0
        self.sw_neg = False
        self.sw_shift = 0
        self.sw_reload = False
        self.sw_div = 0
        self.enabled = False

    def write(self, reg, v):
        if reg == 0:
            self.duty = v >> 6
            self.halt = self.env.loop = bool(v & 0x20)
            self.env.const = bool(v & 0x10)
            self.env.vol = v & 0x0F
        elif reg == 1:
            self.sw_en = bool(v & 0x80)
            self.sw_period = (v >> 4) & 7
            self.sw_neg = bool(v & 0x08)
            self.sw_shift = v & 7
            self.sw_reload = True
        elif reg == 2:
            self.timer = (self.timer & 0x700) | v
        elif reg == 3:
            self.timer = (self.timer & 0xFF) | ((v & 7) << 8)
            if self.enabled:
                self.length = LENGTH_TABLE[v >> 3]
            self.env.start = True

    def _target(self):
        delta = self.timer >> self.sw_shift
        if self.sw_neg:
            delta = -delta - (1 if self.ones else 0)
        return self.timer + delta

    def muted(self):
        return self.timer < 8 or self._target() > 0x7FF

    def clock_quarter(self):
        self.env.clock()

    def clock_half(self):
        if self.length and not self.halt:
            self.length -= 1
        if self.sw_div == 0 and self.sw_en and self.sw_shift and not self.muted():
            t = self._target()
            self.timer = max(0, t)
        if self.sw_div == 0 or self.sw_reload:
            self.sw_div = self.sw_period
            self.sw_reload = False
        else:
            self.sw_div -= 1

    def volume(self):
        if not self.enabled or self.length == 0 or self.muted():
            return 0
        return self.env.output()


class Triangle:
    def __init__(self):
        self.control = False
        self.lin_reload_val = 0
        self.lin = 0
        self.lin_reload = False
        self.length = 0
        self.timer = 0
        self.enabled = False

    def write(self, reg, v):
        if reg == 0:
            self.control = bool(v & 0x80)
            self.lin_reload_val = v & 0x7F
        elif reg == 2:
            self.timer = (self.timer & 0x700) | v
        elif reg == 3:
            self.timer = (self.timer & 0xFF) | ((v & 7) << 8)
            if self.enabled:
                self.length = LENGTH_TABLE[v >> 3]
            self.lin_reload = True

    def clock_quarter(self):
        if self.lin_reload:
            self.lin = self.lin_reload_val
        elif self.lin:
            self.lin -= 1
        if not self.control:
            self.lin_reload = False

    def clock_half(self):
        if self.length and not self.control:
            self.length -= 1

    def active(self):
        # timer < 2 is ultrasonic; many drivers use it as a silence trick
        return self.enabled and self.length > 0 and self.lin > 0 and self.timer >= 2


class Noise:
    def __init__(self):
        self.env = Envelope()
        self.halt = False
        self.mode = 0
        self.period_idx = 0
        self.length = 0
        self.enabled = False

    def write(self, reg, v):
        if reg == 0:
            self.halt = self.env.loop = bool(v & 0x20)
            self.env.const = bool(v & 0x10)
            self.env.vol = v & 0x0F
        elif reg == 2:
            self.mode = (v >> 7) & 1
            self.period_idx = v & 0x0F
        elif reg == 3:
            if self.enabled:
                self.length = LENGTH_TABLE[v >> 3]
            self.env.start = True

    def clock_quarter(self):
        self.env.clock()

    def clock_half(self):
        if self.length and not self.halt:
            self.length -= 1

    def volume(self):
        if not self.enabled or self.length == 0:
            return 0
        return self.env.output()


class DMC:
    def __init__(self):
        self.irq = False
        self.loop = False
        self.rate_idx = 0
        self.dac = 0
        self.addr_reg = 0
        self.len_reg = 0
        self.end_cycle = None     # None = idle, float('inf') = looping
        self.start_count = 0      # increments each time a sample (re)starts

    def write(self, reg, v):
        if reg == 0:
            self.irq = bool(v & 0x80)
            self.loop = bool(v & 0x40)
            self.rate_idx = v & 0x0F
        elif reg == 1:
            self.dac = v & 0x7F
        elif reg == 2:
            self.addr_reg = v
        elif reg == 3:
            self.len_reg = v

    @property
    def sample_addr(self):
        return 0xC000 + self.addr_reg * 64

    @property
    def sample_len(self):
        return self.len_reg * 16 + 1

    def playing(self, cyc):
        return self.end_cycle is not None and cyc < self.end_cycle

    def enable(self, on, cyc):
        if not on:
            self.end_cycle = None
        elif not self.playing(cyc):
            self.start_count += 1
            if self.loop:
                self.end_cycle = float("inf")
            else:
                self.end_cycle = cyc + self.sample_len * 8 * DMC_RATE_NTSC[self.rate_idx]


class APUState:
    def __init__(self, cpu_hz=1789773.0):
        self.cpu_hz = cpu_hz
        self.p1 = Pulse(True)
        self.p2 = Pulse(False)
        self.tri = Triangle()
        self.noise = Noise()
        self.dmc = DMC()
        self.mode5 = False
        self.seq_origin = 0       # cycle at which the sequencer was last reset
        self.seq_step = 0         # index of next step
        self.now = None

    # ------------------------------------------------------------ sequencer
    def _clock_quarter(self):
        self.p1.clock_quarter()
        self.p2.clock_quarter()
        self.tri.clock_quarter()
        self.noise.clock_quarter()

    def _clock_half(self):
        self.p1.clock_half()
        self.p2.clock_half()
        self.tri.clock_half()
        self.noise.clock_half()

    def advance(self, cyc):
        """Run the frame sequencer up to (not including) CPU cycle `cyc`."""
        if self.now is None:
            self.now = cyc
            self.seq_origin = cyc
            self.seq_step = 0
            return
        steps = SEQ5 if self.mode5 else SEQ4
        period = SEQ5_PERIOD if self.mode5 else SEQ4_PERIOD
        while True:
            t = self.seq_origin + steps[self.seq_step]
            if t >= cyc:
                break
            i = self.seq_step
            if self.mode5:
                if i in (0, 2):
                    self._clock_quarter()
                elif i in (1, 4):
                    self._clock_quarter()
                    self._clock_half()
                # step 3 does nothing in 5-step mode
            else:
                self._clock_quarter()
                if i in (1, 3):
                    self._clock_half()
            self.seq_step += 1
            if self.seq_step >= len(steps):
                self.seq_step = 0
                self.seq_origin += period
        self.now = cyc

    # --------------------------------------------------------------- writes
    def write(self, cyc, addr, v):
        self.advance(cyc)
        if addr <= 0x4003:
            self.p1.write(addr - 0x4000, v)
        elif addr <= 0x4007:
            self.p2.write(addr - 0x4004, v)
        elif addr <= 0x400B:
            self.tri.write(addr - 0x4008, v)
        elif addr <= 0x400F:
            self.noise.write(addr - 0x400C, v)
        elif addr <= 0x4013:
            self.dmc.write(addr - 0x4010, v)
        elif addr == 0x4015:
            for ch, bit in ((self.p1, 1), (self.p2, 2), (self.tri, 4), (self.noise, 8)):
                ch.enabled = bool(v & bit)
                if not ch.enabled:
                    ch.length = 0
            self.dmc.enable(bool(v & 0x10), cyc)
        elif addr == 0x4017:
            self.mode5 = bool(v & 0x80)
            self.seq_origin = cyc
            self.seq_step = 0
            if self.mode5:
                self._clock_quarter()
                self._clock_half()

    def read_status(self, cyc):
        self.advance(cyc)
        v = 0
        for ch, bit in ((self.p1, 1), (self.p2, 2), (self.tri, 4), (self.noise, 8)):
            if ch.length > 0:
                v |= bit
        if self.dmc.playing(cyc):
            v |= 0x10
        return v

    # ------------------------------------------------------------- snapshot
    def snapshot(self):
        hz = self.cpu_hz
        out = {}
        for name, ch in (("P1", self.p1), ("P2", self.p2)):
            f = hz / (16.0 * (ch.timer + 1))
            out[name] = {
                "period": ch.timer,
                "freq": f,
                "note": freq_to_note(f),
                "vol": ch.volume(),
                "reg_vol": ch.env.vol,
                "const": ch.env.const,
                "duty": ch.duty,
                "length": ch.length,
                "sweep": ch.sw_en and ch.sw_shift > 0,
                "muted": ch.muted(),
            }
        t = self.tri
        f = hz / (32.0 * (t.timer + 1))
        out["TRI"] = {
            "period": t.timer,
            "freq": f,
            "note": freq_to_note(f),
            "vol": 15 if t.active() else 0,
            "linear": t.lin,
            "length": t.length,
        }
        n = self.noise
        out["NOI"] = {
            "period_idx": n.period_idx,
            "mode": n.mode,
            "vol": n.volume(),
            "reg_vol": n.env.vol,
            "const": n.env.const,
            "length": n.length,
        }
        d = self.dmc
        out["DMC"] = {
            "playing": d.playing(self.now or 0),
            "addr": d.sample_addr,
            "len": d.sample_len,
            "rate": d.rate_idx,
            "loop": d.loop,
            "dac": d.dac,
            "starts": d.start_count,
        }
        return out
