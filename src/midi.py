"""Minimal Standard MIDI File (format 1) writer."""

import struct


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


class Track:
    def __init__(self, name=None):
        self.events = []    # (tick, order, bytes)
        if name:
            self.meta(0, 0x03, name.encode("utf-8", "replace"))

    def meta(self, tick, kind, data):
        self.events.append((tick, 0, b"\xFF" + bytes([kind]) + _vlq(len(data)) + data))

    def tempo(self, tick, bpm):
        us = int(round(60_000_000 / bpm))
        self.meta(tick, 0x51, us.to_bytes(3, "big"))

    def time_signature(self, tick, num=4, den_pow=2, clocks=24):
        self.meta(tick, 0x58, bytes([num, den_pow, clocks, 8]))

    def program(self, tick, ch, prog):
        self.events.append((tick, 1, bytes([0xC0 | ch, prog & 0x7F])))

    def control(self, tick, ch, cc, val):
        self.events.append((tick, 1, bytes([0xB0 | ch, cc & 0x7F, val & 0x7F])))

    def note(self, start, end, ch, pitch, vel):
        # note-offs sort before note-ons at the same tick (order 2 < 3)
        self.events.append((start, 3, bytes([0x90 | ch, pitch & 0x7F, max(1, min(127, vel))])))
        self.events.append((end, 2, bytes([0x80 | ch, pitch & 0x7F, 0])))

    def encode(self):
        evs = sorted(self.events, key=lambda e: (e[0], e[1]))
        body = bytearray()
        last = 0
        for tick, _, data in evs:
            body += _vlq(tick - last) + data
            last = tick
        body += b"\x00\xFF\x2F\x00"
        return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def write_smf(path, tracks, ppq=480):
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ppq))
        for t in tracks:
            f.write(t.encode())
