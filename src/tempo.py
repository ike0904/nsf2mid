"""
Tempo detection from frame-exact note onsets.

NES music drivers advance on whole frames, so onsets fall on a regular grid of
`unit` frames (the finest row). Drivers with a tempo accumulator (e.g. Dragon
Quest) produce fractional units such as 14.4 frames (onsets on round(k * 14.4));
grooves such as 6,7,6,7 give 6.5.

1. Grid search (detect_grid): unit candidates 2..48 frames on a geometric scale
   (0.2 % steps) plus all integer / half units. Phase by circular mean. Score is
   the share of onsets within tolerance, corrected for chance:
       norm = (hit - chance) / (1 - chance),  chance = 2 * tol / unit
   The largest unit whose norm is within NORM_MARGIN of the best is chosen.
2. Rows per beat (choose_rows): if most inter-onset intervals are a multiple of
   3 units the music is ternary (triplets / shuffle) -> 12, 6, 3 or 24 rows per
   beat, otherwise 4, 2, 8 or 16; first candidate giving BPM in [BPM_LO, BPM_HI).
3. Tempo map (detect_tempo_map): the global grid is checked in windows of
   WINDOW_SEC; windows it does not fit get their own grid; consecutive windows
   with the same unit are merged into segments; boundaries are refined per onset.
"""

import math
from collections import Counter

BPM_LO = 70.0
BPM_HI = 170.0
NORM_MARGIN = 0.08
WINDOW_SEC = 8.0
FIT_OK = 0.85
LOCAL_MIN = 0.6     # a window gets its own grid if it fits at least this well
LOCAL_GAIN = 0.25   # ... and beats the global grid by this margin
MERGE_RATIO = 0.03  # neighbouring segments closer than this are merged


def _tol(unit):
    # rounding of accumulator drivers is <= 0.5 frame; a little slack for phase error
    return min(0.75, unit * 0.2)


def _hits(onsets, unit, phase):
    tol = _tol(unit)
    out = []
    for o in onsets:
        d = (o - phase) % unit
        if d > unit / 2:
            d -= unit
        out.append(abs(d) <= tol)
    return out


def _phase(onsets, unit):
    two_pi = 2.0 * math.pi
    sx = sy = 0.0
    for o in onsets:
        a = two_pi * (o / unit)
        sx += math.cos(a)
        sy += math.sin(a)
    return (math.atan2(sy, sx) / two_pi) * unit % unit


def _fit(onsets, unit):
    phase = _phase(onsets, unit)
    score = sum(_hits(onsets, unit, phase)) / len(onsets)
    chance = min(0.999, 2 * _tol(unit) / unit)
    norm = (score - chance) / (1 - chance)
    return score, norm, phase


def _candidates(min_unit, max_unit):
    out = set(i / 2 for i in range(int(min_unit * 2), int(max_unit * 2) + 1))
    u = float(min_unit)
    while u <= max_unit + 1e-9:
        out.add(u)
        u *= 1.002
    return sorted(out, reverse=True)


def detect_grid(onsets, min_unit=2.0, max_unit=48.0):
    onsets = sorted(set(onsets))
    if len(onsets) < 4:
        return None
    results = [(u,) + _fit(onsets, u) for u in _candidates(min_unit, max_unit)]
    best = max(r[2] for r in results)
    for u, score, norm, phase in results:          # descending unit order
        if norm >= best - NORM_MARGIN:
            # local refinement absorbs accumulator drift; keep u unless clearly better
            fine = [(v,) + _fit(onsets, v) for v in (u * (1 + k * 0.0004) for k in range(-6, 7))]
            v = max(fine, key=lambda r: r[2])
            if v[2] > norm + 0.01:
                u, score, norm, phase = v
            # snap to an exact integer / half unit when it fits as well
            snap = round(u * 2) / 2
            if snap != u and abs(snap - u) / u < 0.002:
                s2 = _fit(onsets, snap)
                if s2[0] >= score - 0.005:
                    u, (score, norm, phase) = snap, s2
            return {"unit": u, "score": score, "norm": norm, "phase": phase}
    return None


def is_ternary(onsets, unit):
    onsets = sorted(set(onsets))
    iois = [round((b - a) / unit) for a, b in zip(onsets, onsets[1:])]
    iois = [i for i in iois if i > 0]
    if not iois:
        return False
    div3 = sum(1 for i in iois if i % 3 == 0)
    return div3 / len(iois) > 0.5


SHUFFLE_PAIRS = 0.45   # share of consecutive interval pairs that are long-short (2:1) or short-long


def shuffle_step(onsets, unit):
    """Shuffle / swing written on a straight grid: consecutive intervals alternate
    long-short in a 2:1 ratio (2s, s). Returns s (in grid units) or None.

    Examples: Super Mario Bros. 3 - grid 6 frames, intervals 12,6,12,6 (s = 1 unit, 86 % of
    the pairs); Super Mario USA - grid 2 frames, intervals 12,6 (s = 3 units, 54 %).
    Plain runs of equal notes, dotted rhythms (3:1) or 8th+16th figures do not qualify.
    """
    onsets = sorted(set(onsets))
    if len(onsets) < 16:
        return None
    iois = [round((b - a) / unit) for a, b in zip(onsets, onsets[1:])]
    pairs = list(zip(iois, iois[1:]))
    if not pairs:
        return None
    best, best_s = 0.0, None
    for step in range(1, 13):
        hit = sum(1 for a, b in pairs if (a, b) in ((2 * step, step), (step, 2 * step)))
        share = hit / len(pairs)
        if share > best:
            best, best_s = share, step
    return best_s if best >= SHUFFLE_PAIRS else None


def choose_rows(unit, frame_rate, ternary, prefer_bpm=None):
    family = (12, 6, 3, 24) if ternary else (4, 2, 8, 16)
    if prefer_bpm:
        # keep continuity with the previous segment
        return min(family, key=lambda m: abs(math.log((60.0 * frame_rate / (unit * m)) / prefer_bpm)))
    for m in family:
        bpm = 60.0 * frame_rate / (unit * m)
        if BPM_LO <= bpm < BPM_HI:
            return m
    return family[0]


def beat_fit(onsets, period, tol=0.75):
    """Best-phase grid fit (exhaustive over onset residues, unlike the circular mean this
    works for multi-modal patterns such as shuffles). Returns (norm, score, phase)."""
    if not onsets:
        return 0.0, 0.0, 0.0
    res = [o % period for o in onsets]
    best = (0, 0.0)
    for ph in set(round(r * 4) / 4 for r in res):
        hit = 0
        for r in res:
            d = abs(r - ph)
            if min(d, period - d) <= tol:
                hit += 1
        if hit > best[0]:
            best = (hit, ph)
    score = best[0] / len(onsets)
    chance = min(0.999, 2 * tol / period)
    return (score - chance) / (1 - chance), score, best[1]


class Segment:
    def __init__(self, start, unit, phase, score):
        self.start = start      # first onset frame belonging to the segment
        self.unit = unit
        self.phase = phase
        self.score = score
        self.rows = 4
        self.bpm = 0.0

    def __repr__(self):
        return f"Segment(start={self.start}, unit={self.unit:.4g}, bpm={self.bpm:.2f}, fit={self.score:.2f})"


def detect_tempo_map(onsets, frame_rate, rows_per_beat=None, unit=None):
    onsets = sorted(set(onsets))
    if len(onsets) < 4:
        return None
    if unit:
        sc, nm, ph = _fit(onsets, unit)
        g = {"unit": unit, "score": sc, "norm": nm, "phase": ph}
        segs = [Segment(onsets[0], unit, ph, sc)]
    else:
        g = detect_grid(onsets)
        segs = _segment(onsets, frame_rate, g)
    ternary = is_ternary(onsets, g["unit"])
    step = None if rows_per_beat else shuffle_step(onsets, g["unit"])
    shuffle = step is not None
    if shuffle:
        ternary = True
    prev = None
    for s in segs:
        if rows_per_beat:
            s.rows = rows_per_beat
        elif shuffle and prev is None:
            # 1 beat = one long-short pair (2s + s); double it if that would exceed 240 BPM
            s.rows = 3 * step if 60.0 * frame_rate / (s.unit * 3 * step) < 240.0 else 6 * step
        else:
            s.rows = choose_rows(s.unit, frame_rate, ternary, prev)
        s.bpm = 60.0 * frame_rate / (s.unit * s.rows)
        prev = s.bpm
    return {"global": g, "segments": segs, "ternary": ternary, "shuffle": shuffle}


def _segment(onsets, frame_rate, g):
    win = frame_rate * WINDOW_SEC
    t0 = onsets[0]
    windows = []
    i = 0
    while i < len(onsets):
        w0 = t0 + len(windows) * win
        sub = []
        while i < len(onsets) and onsets[i] < w0 + win:
            sub.append(onsets[i])
            i += 1
        windows.append(sub)

    # label each window: None = global grid fits, else its own grid
    labels = []
    for sub in windows:
        if len(sub) < 6:
            labels.append(None)
            continue
        ph = _phase(sub, g["unit"])
        gfit = sum(_hits(sub, g["unit"], ph)) / len(sub)
        if gfit >= FIT_OK:
            labels.append(None)
            continue
        lg = detect_grid(sub, max(2.0, g["unit"] / 2.2), min(48.0, g["unit"] * 2.2))
        if (lg and abs(lg["unit"] - g["unit"]) / g["unit"] >= 0.015
                and lg["score"] >= LOCAL_MIN and lg["score"] - gfit >= LOCAL_GAIN):
            labels.append(lg["unit"])
        else:
            labels.append(None)

    # merge runs of equal labels (units within 1.5 %)
    runs = []   # [label, [onsets]]
    for lab, sub in zip(labels, windows):
        if runs and _same(runs[-1][0], lab):
            runs[-1][1].extend(sub)
        else:
            runs.append([lab, list(sub)])
    runs = [r for r in runs if r[1]]

    # merge neighbouring runs whose units differ by < MERGE_RATIO (accumulator drift, not a tempo change)
    merged = []
    for lab, sub in runs:
        u = g["unit"] if lab is None else lab
        if merged:
            pu = g["unit"] if merged[-1][0] is None else merged[-1][0]
            if abs(u - pu) / pu < MERGE_RATIO:
                merged[-1][1].extend(sub)
                continue
        merged.append([lab, list(sub)])
    runs = merged

    segs = []
    for lab, sub in runs:
        u = g["unit"] if lab is None else lab
        if lab is not None:
            lg = detect_grid(sub, max(2.0, u * 0.95), min(48.0, u * 1.05))
            if lg:
                u = lg["unit"]
        ph = _phase(sub, u)
        sc = sum(_hits(sub, u, ph)) / len(sub)
        segs.append(Segment(sub[0], u, ph, sc))

    # refine boundaries: move each split to the onset that maximizes total hits
    for k in range(1, len(segs)):
        a, b = segs[k - 1], segs[k]
        lo = b.start - win
        hi = b.start + win
        region = [o for o in onsets if lo <= o < hi and o >= a.start]
        if len(region) < 2:
            continue
        ha = _hits(region, a.unit, a.phase)
        hb = _hits(region, b.unit, b.phase)
        best_j, best_v = None, -1
        for j in range(1, len(region)):
            v = sum(ha[:j]) + sum(hb[j:])
            if v > best_v:
                best_j, best_v = j, v
        b.start = region[best_j]
    # drop segments that became empty or out of order
    clean = []
    for s in segs:
        if clean and s.start <= clean[-1].start:
            continue
        clean.append(s)
    return clean


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) / a < 0.015


class TickMap:
    """frame -> MIDI tick through a piecewise tempo map."""

    def __init__(self, tmap, ppq, quantize=True):
        self.ppq = ppq
        self.quantize = quantize
        self.parts = []   # (start_frame, origin_frame, unit, tick_at_origin, ticks_per_unit, bpm)
        tick = 0
        prev = None
        for s in tmap["segments"]:
            tpu = ppq / s.rows
            if prev is None:
                # keep the leading silence: the song starts at frame 0 (drivers start the sequence
                # on the first PLAY call), so the first grid point at/near frame 0 is tick 0
                origin = s.phase + s.unit * round((0 - s.phase) / s.unit)
            else:
                origin = s.phase + s.unit * round((s.start - s.phase) / s.unit)
            if prev is not None:
                tick = prev[3] + round((origin - prev[1]) / prev[2]) * prev[4]
            part = (s.start, origin, s.unit, tick, tpu, s.bpm)
            self.parts.append(part)
            prev = part

    def tempo_events(self):
        return [(max(0, int(round(p[3]))), p[5]) for p in self.parts]

    @property
    def min_ticks(self):
        return min(p[4] for p in self.parts)

    def __call__(self, frame):
        part = self.parts[0]
        for p in self.parts:
            if frame >= p[0] - p[2] / 2:
                part = p
        _, origin, unit, t0, tpu, _ = part
        rows = (frame - origin) / unit
        if self.quantize:
            rows = round(rows)
        return max(0, int(round(t0 + rows * tpu)))
