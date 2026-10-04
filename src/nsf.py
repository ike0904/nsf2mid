"""
NSF file parser and player (register-logging, no audio output).

The player runs INIT once and then calls PLAY at the rate given in the header,
recording every APU register write together with its CPU-cycle timestamp.
"""

import struct

from cpu6502 import CPU6502
from apu_state import APUState

NTSC_CPU_HZ = 1789773.0
PAL_CPU_HZ = 1662607.0

EXPANSION_NAMES = [
    (0x01, "VRC6"),
    (0x02, "VRC7"),
    (0x04, "FDS"),
    (0x08, "MMC5"),
    (0x10, "N163"),
    (0x20, "5B"),
]

# Sentinel address used as the return address for INIT / PLAY calls.
# Placed in the unmapped $5xxx area (not used by 2A03-only NSFs).
SENTINEL = 0x5FF0


class NSFHeader:
    def __init__(self, data):
        if data[:5] != b"NESM\x1a":
            raise ValueError("Not an NSF file (bad magic)")
        self.version = data[5]
        self.total_songs = data[6]
        self.start_song = data[7]
        self.load_addr, self.init_addr, self.play_addr = struct.unpack_from("<HHH", data, 8)
        self.title = _cstr(data[0x0E:0x2E])
        self.artist = _cstr(data[0x2E:0x4E])
        self.copyright = _cstr(data[0x4E:0x6E])
        self.speed_ntsc = struct.unpack_from("<H", data, 0x6E)[0]
        self.banks = list(data[0x70:0x78])
        self.speed_pal = struct.unpack_from("<H", data, 0x78)[0]
        self.pal_flags = data[0x7A]
        self.expansion = data[0x7B]
        self.bankswitched = any(self.banks)

    @property
    def is_pal(self):
        # bit0: PAL, bit1: dual. Dual-compatible tunes are played as NTSC.
        return (self.pal_flags & 0x03) == 0x01

    @property
    def expansion_names(self):
        return [n for bit, n in EXPANSION_NAMES if self.expansion & bit]

    def summary(self):
        lines = [
            f"Title     : {self.title}",
            f"Artist    : {self.artist}",
            f"Copyright : {self.copyright}",
            f"Songs     : {self.total_songs} (start {self.start_song})",
            f"Load/Init/Play: ${self.load_addr:04X} / ${self.init_addr:04X} / ${self.play_addr:04X}",
            f"Speed     : NTSC {self.speed_ntsc} us, PAL {self.speed_pal} us, "
            f"region {'PAL' if self.is_pal else 'NTSC'}",
            f"Bankswitch: {'yes ' + ' '.join(f'{b:02X}' for b in self.banks) if self.bankswitched else 'no'}",
            f"Expansion : {', '.join(self.expansion_names) or 'none'}",
        ]
        return "\n".join(lines)


def _cstr(b):
    b = b.split(b"\x00", 1)[0]
    for enc in ("shift_jis", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


class NSFBus:
    """Memory map for an NSF tune (2A03 only for now)."""

    def __init__(self, header, rom_data, on_apu_write, on_status_read=None):
        self.h = header
        self.ram = bytearray(0x800)
        self.sram = bytearray(0x2000)       # $6000-$7FFF
        self.on_apu_write = on_apu_write
        self.on_status_read = on_status_read
        self.cpu = None

        if header.bankswitched:
            pad = header.load_addr & 0x0FFF
            img = bytes(pad) + rom_data
        else:
            pad = header.load_addr - 0x8000
            if pad < 0:
                raise ValueError(f"Load address ${header.load_addr:04X} below $8000 is not supported")
            img = bytes(pad) + rom_data
        n_banks = max(1, (len(img) + 0xFFF) // 0x1000)
        img = img + bytes(n_banks * 0x1000 - len(img))
        self.image = img
        self.n_banks = n_banks
        # 8 slots of 4KB for $8000-$FFFF; each holds the bank's start offset
        self.slot = [0] * 8
        if header.bankswitched:
            for i, b in enumerate(header.banks):
                self._set_bank(i, b)
        else:
            for i in range(8):
                self.slot[i] = i * 0x1000 if i < n_banks else None

    def _set_bank(self, slot, bank):
        self.slot[slot] = (bank % self.n_banks) * 0x1000

    def read(self, addr):
        if addr < 0x2000:
            return self.ram[addr & 0x7FF]
        if addr >= 0x8000:
            base = self.slot[(addr - 0x8000) >> 12]
            if base is None:
                return 0
            return self.image[base + (addr & 0x0FFF)]
        if addr >= 0x6000:
            return self.sram[addr - 0x6000]
        if addr == 0x4015 and self.on_status_read:
            return self.on_status_read()
        if addr == SENTINEL:
            return 0x4C  # JMP to self, never actually executed (player stops at sentinel)
        return 0

    def write(self, addr, val):
        if addr < 0x2000:
            self.ram[addr & 0x7FF] = val
        elif 0x4000 <= addr <= 0x4017:
            self.on_apu_write(addr, val)
        elif 0x5FF8 <= addr <= 0x5FFF:
            if self.h.bankswitched:
                self._set_bank(addr - 0x5FF8, val)
        elif 0x6000 <= addr < 0x8000:
            self.sram[addr - 0x6000] = val


class NSFPlayer:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        self.header = NSFHeader(data)
        self.rom = data[0x80:]
        self.writes = []          # (cpu_cycle, addr, value)
        self.bus = None
        self.cpu = None
        self.warnings = []

    @property
    def cpu_hz(self):
        return PAL_CPU_HZ if self.header.is_pal else NTSC_CPU_HZ

    @property
    def frame_cycles(self):
        speed = self.header.speed_pal if self.header.is_pal else self.header.speed_ntsc
        if speed == 0:
            speed = 20000 if self.header.is_pal else 16639
        return speed * self.cpu_hz / 1_000_000.0

    def _log_write(self, addr, val):
        cyc = self.cpu.cycles
        self.writes.append((cyc - self.t0 if self.t0 is not None else cyc, addr, val))
        dmc = self.apu.dmc
        before = dmc.start_count
        self.apu.write(cyc, addr, val)
        if dmc.start_count != before:
            self._capture_dmc(dmc.sample_addr, dmc.sample_len)

    def _capture_dmc(self, addr, length):
        """Keep the bytes of each DPCM sample (for drum classification)."""
        key = (addr, length)
        if key in self.dmc_samples:
            return
        rd = self.bus.read
        self.dmc_samples[key] = bytes(rd(0x8000 | ((addr + i) & 0x7FFF)) for i in range(length))

    def _status_read(self):
        # Drivers that read $4015 (length counter status) need a live APU model
        return self.apu.read_status(self.cpu.cycles)

    def _call(self, addr, max_cycles):
        cpu = self.cpu
        ret = SENTINEL - 1
        cpu.push(ret >> 8)
        cpu.push(ret & 0xFF)
        cpu.pc = addr
        limit = cpu.cycles + max_cycles
        step = cpu.step
        while cpu.pc != SENTINEL:
            step()
            if cpu.cycles >= limit or cpu.jammed:
                return False
        return True

    def start(self, track):
        """Load the tune and run INIT for `track` (1-based)."""
        h = self.header
        self.warnings = []
        if h.expansion:
            self.warnings.append(
                f"Expansion audio ({', '.join(h.expansion_names)}) is not emulated; "
                f"only 2A03 channels are logged.")
        self.writes = []
        self.dmc_samples = {}
        self.t0 = None
        self.frames_done = 0
        self.stuck = 0
        self.aborted = False
        self.apu = APUState(self.cpu_hz)
        self.bus = NSFBus(h, self.rom, self._log_write, self._status_read)
        self.cpu = CPU6502(self.bus)
        cpu = self.cpu

        # Power-up APU state as recommended by the NSF spec
        cpu.cycles = 0
        for a in range(0x4000, 0x4014):
            self.bus.write(a, 0x00)
        self.bus.write(0x4015, 0x00)
        self.bus.write(0x4015, 0x0F)
        self.bus.write(0x4017, 0x40)

        # INIT
        cpu.a = (track - 1) & 0xFF
        cpu.x = 1 if h.is_pal else 0
        cpu.y = 0
        cpu.s = 0xFD
        cpu.p = 0x24
        if not self._call(h.init_addr, 3_000_000):
            self.warnings.append("INIT did not return within cycle limit")
        # Frame 0 starts when INIT returns; INIT writes get negative cycle values
        self.t0 = cpu.cycles
        self.writes = [(c - self.t0, a, v) for c, a, v in self.writes]

    def advance(self, frames):
        """Call PLAY `frames` more times. Returns the total number of frames emulated."""
        cpu = self.cpu
        fc = self.frame_cycles
        play = self.header.play_addr
        for _ in range(frames):
            if self.aborted:
                break
            f = self.frames_done
            start = self.t0 + int(round(f * fc))
            if cpu.cycles < start:
                cpu.cycles = start
            cpu.s = 0xFD
            if not self._call(play, int(fc * 2)):
                self.stuck += 1
                cpu.s = 0xFD
                cpu.jammed = False
                if self.stuck >= 10 and self.stuck == f + 1:
                    self.warnings.append("PLAY never returns; emulation aborted")
                    self.aborted = True
            self.frames_done += 1
        return self.frames_done

    def finish_warnings(self):
        if self.stuck and not self.aborted:
            self.warnings.append(f"PLAY did not return within limit on {self.stuck} frame(s)")

    def run(self, track, frames):
        """Run `track` (1-based) for `frames` play calls; returns the register writes
        as (cpu_cycle relative to frame 0, addr, value). INIT writes have cycle < 0."""
        self.start(track)
        self.advance(frames)
        self.finish_warnings()
        return self.writes
