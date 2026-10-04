"""
Drum extraction (Phase 3): noise channel, DPCM samples and triangle "drums".

All hits are mapped to General MIDI percussion (MIDI channel 10).

Noise
  Hit start: hi reg ($400F) write while audible, silence -> sound, or a volume
  jump of >= NOISE_REATTACK (drivers that re-hit by rewriting the volume only).
  A hit lasts until silence or the next hit. Many drivers sweep the period
  during a hit (e.g. Mega Man 2 snare 11,0,5,10,15), so the classification uses
  the volume-weighted average period index, the hit length and the peak volume.
  The kind of a hit is keyed by its first-frame (period index, mode) so it can be
  remapped from the command line (--drum-map "3:0=42,12:0=36").

DPCM
  Each sample start is a hit. The sample bytes are decoded (1-bit delta) and the
  zero-crossing rate after a 40 Hz high-pass gives a rough brightness:
    < 1200 Hz : kick (short) / low tom (long)
    < 4500 Hz : snare
    otherwise : hi-hat (short) / open hi-hat
  Looping samples are treated as tonal samples and skipped.

Triangle drums
  Short triangle notes (<= TRI_DRUM_MAX frames) whose pitch falls by
  >= TRI_DRUM_DROP semitones (the classic "triangle kick") become kick drums.
"""

from apu_state import DMC_RATE_NTSC
from notes import find_gate_end

GM_NAMES = {
    35: "Acoustic Bass Drum", 36: "Bass Drum", 37: "Side Stick", 38: "Snare", 39: "Hand Clap",
    40: "Electric Snare", 41: "Low Floor Tom", 42: "Closed Hi-Hat", 43: "High Floor Tom",
    44: "Pedal Hi-Hat", 45: "Low Tom", 46: "Open Hi-Hat", 47: "Low-Mid Tom", 48: "Hi-Mid Tom",
    49: "Crash Cymbal", 50: "High Tom", 51: "Ride Cymbal", 56: "Cowbell", 57: "Crash Cymbal 2",
}

NOISE_REATTACK = 3
TRI_DRUM_MAX = 12
TRI_DRUM_DROP = 5.0


class DrumHit:
    __slots__ = ("src", "start", "end", "gm", "vel", "kind", "info")

    def __init__(self, src, start, end, gm, vel, kind, info=""):
        self.src = src          # "NOI" / "DMC" / "TRI"
        self.start = start
        self.end = end
        self.gm = gm
        self.vel = vel
        self.kind = kind        # key used for remapping / reporting
        self.info = info


def _vel(v):
    return max(1, min(127, int(round(v * 127 / 15))))


# ------------------------------------------------------------------ noise
def classify_noise(w_idx, dur, mode):
    if w_idx <= 4.5:
        if dur <= 5:
            return 42
        return 46 if dur <= 12 else 49
    if w_idx < 9.5:
        return 38 if dur <= 20 else 49
    return 36 if dur <= 8 else 41


def noise_hits(frames, drum_map=None):
    hits = []
    cur = None
    prev_vol = 0

    def close(at):
        nonlocal cur
        if cur is None:
            return
        f0, key, data = cur
        vols = [v for _, v in data]
        # ignore a low reverb tail (DQ2 holds the noise at volume 1) when measuring the hit
        g = find_gate_end(vols)
        data = data[:g]
        vols = vols[:g]
        tot = sum(vols) or 1
        w_idx = sum(p * v for p, v in data) / tot
        dur = min(at - f0, g)
        gm = classify_noise(w_idx, dur, key[1])
        if drum_map and key in drum_map:
            gm = drum_map[key]
        hits.append(DrumHit("NOI", f0, f0 + dur, gm, _vel(max(vols)), key,
                            f"idx {key[0]} mode {key[1]} avg {w_idx:.1f} {dur}f"))
        cur = None

    for fr in frames:
        n = fr["state"]["NOI"]
        vol = n["vol"]
        trig = 3 in fr["written"]["NOI"]
        f = fr["frame"]
        if vol == 0:
            close(f)
            prev_vol = 0
            continue
        if cur is None or trig or vol - prev_vol >= NOISE_REATTACK:
            close(f)
            cur = (f, (n["period_idx"], n["mode"]), [])
        if f - cur[0] < 60:
            cur[2].append((n["period_idx"], vol))
        prev_vol = vol
    close(len(frames))
    return hits


# ------------------------------------------------------------------- DPCM
def dpcm_decode(data, dac=64):
    out = []
    for byte in data:
        for i in range(8):
            if (byte >> i) & 1:
                if dac <= 125:
                    dac += 2
            elif dac >= 2:
                dac -= 2
            out.append(dac)
    return out


def dpcm_brightness(data, rate_hz):
    sig = dpcm_decode(data)
    if len(sig) < 16:
        return 0.0, 0.0
    win = max(2, int(rate_hz / 40))
    s = 0.0
    prev = None
    z = 0
    for i, v in enumerate(sig):
        s += v
        if i >= win:
            s -= sig[i - win]
        hp = v - s / min(i + 1, win)
        if prev is not None and (hp < 0) != (prev < 0):
            z += 1
        prev = hp
    dur = len(sig) / rate_hz
    return z / 2 / dur, dur


def classify_dpcm(freq, dur):
    if freq < 1200:
        return 36 if dur < 0.12 else 45
    if freq < 4500:
        return 38
    return 42 if dur < 0.08 else 46


def dmc_hits(frames, samples, cpu_hz, drum_map=None):
    hits = []
    prev = None
    cache = {}
    for fr in frames:
        d = fr["state"]["DMC"]
        if prev is not None and d["starts"] != prev:
            key = (d["addr"], d["len"], d["rate"])
            if d["loop"]:
                prev = d["starts"]
                continue
            if key not in cache:
                rate_hz = cpu_hz / DMC_RATE_NTSC[d["rate"]]
                data = samples.get((d["addr"], d["len"]), b"")
                freq, dur = dpcm_brightness(data, rate_hz)
                cache[key] = (classify_dpcm(freq, dur), freq, dur)
            gm, freq, dur = cache[key]
            dkey = ("DMC", d["addr"], d["len"], d["rate"])
            if drum_map and dkey in drum_map:
                gm = drum_map[dkey]
            frames_len = max(1, int(round(dur * 60)))
            hits.append(DrumHit("DMC", fr["frame"], fr["frame"] + frames_len, gm, 110, dkey,
                                f"${d['addr']:04X} len {d['len']} rate {d['rate']} ~{freq:.0f}Hz {dur * 1000:.0f}ms"))
        prev = d["starts"]
    return hits


# --------------------------------------------------------------- triangle
def split_triangle_drums(tri_notes, enabled=True):
    """Return (tonal_notes, drum_hits)."""
    if not enabled:
        return tri_notes, []
    tonal, hits = [], []
    for n in tri_notes:
        length = n.end - n.start
        ps = [p for p in n.pitches if p is not None]
        if ps and length <= TRI_DRUM_MAX and ps[0] - min(ps) >= TRI_DRUM_DROP:
            gm = 36 if min(ps) < 60 else 45
            hits.append(DrumHit("TRI", n.start, n.end, gm, 110, ("TRI",),
                                f"glide {ps[0]:.0f}->{min(ps):.0f}"))
        else:
            tonal.append(n)
    return tonal, hits


def parse_drum_map(text):
    """'3:0=42,12:0=36,DMC:E000:129:15=38' -> {key: gm}."""
    out = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=")
        parts = k.split(":")
        if parts[0].upper() == "DMC":
            key = ("DMC", int(parts[1], 16), int(parts[2]), int(parts[3]))
        else:
            key = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        out[key] = int(v)
    return out


def summarize(hits):
    """Per kind: (count, gm, example info) for the console report."""
    table = {}
    for h in hits:
        t = table.setdefault((h.src, h.kind), [0, {}, h.info])
        t[0] += 1
        t[1][h.gm] = t[1].get(h.gm, 0) + 1
    out = []
    for (src, kind), (cnt, gms, info) in sorted(table.items(), key=lambda x: -x[1][0]):
        gm = max(gms, key=gms.get)
        out.append((src, kind, cnt, gm, info))
    return out
