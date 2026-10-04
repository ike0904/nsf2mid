"""
Loop / end detection (Phase 3).

Game music normally loops forever. After the loop point, every channel repeats
its event sequence with a fixed period L. For each channel's event list
(start_frame, token) we look for a lag k (in events) such that, scanning back
from the end, token[i] == token[i+k] and the time difference stays within
TOL frames of L. The scan stops at the first mismatch -> s (earliest matching
event). A lag is valid when the matched part spans (REPEATS - 1) periods, i.e. the
loop body was heard REPEATS times in a row (a section played twice and then followed
by a chorus must not be taken for the loop).
Among valid lags the one with the earliest s wins (ties: smallest L), because
the true loop repeats back to the loop point while shorter internal repeats
(A A B ...) break earlier.

Channels are combined: L = the largest per-channel period (others divide it),
each channel is re-scanned with that period and the loop start is the latest
per-channel start (the point after which *all* channels repeat).

Accumulator-tempo drivers (Dragon Quest) may shift onsets by a frame between
repetitions, hence the tolerance.
"""

TOL = 2                 # frames
MIN_EVENTS = 8          # a channel needs this many events to take part
MIN_LOOP_FRAMES = 120   # ignore periods shorter than 2 s
REPEATS = 3             # the loop body must be heard this many times in a row


def _scan(t, tok, k, period=None):
    n = len(t)
    if k <= 0 or k >= n:
        return None
    last = n - 1 - k
    if tok[n - 1] != tok[last]:
        return None
    L = t[n - 1] - t[last] if period is None else period
    i = last
    while i >= 0 and tok[i] == tok[i + k] and abs((t[i + k] - t[i]) - L) <= TOL:
        i -= 1
    s = i + 1
    if s > last:
        return None
    span = t[last] - t[s]
    return s, L, span


def channel_period(events):
    """events: sorted [(frame, token)]. Returns (loop_start_frame, period) or None."""
    if len(events) < MIN_EVENTS:
        return None
    t = [e[0] for e in events]
    tok = [e[1] for e in events]
    n = len(t)
    best = None
    for k in range(1, n // 2 + 1):
        r = _scan(t, tok, k)
        if r is None:
            continue
        s, L, span = r
        if L < MIN_LOOP_FRAMES or span < (REPEATS - 1) * L - TOL or (n - s) < MIN_EVENTS:
            continue
        cand = (t[s], L)
        if best is None or cand[0] < best[0] - TOL or (abs(cand[0] - best[0]) <= TOL and L < best[1]):
            best = cand
    return best


def channel_start_for_period(events, period):
    """Earliest frame from which `events` repeat with `period` until the end."""
    t = [e[0] for e in events]
    tok = [e[1] for e in events]
    n = len(t)
    # find k whose time difference at the end matches the period
    best = None
    for k in range(1, n):
        last = n - 1 - k
        if last < 0:
            break
        d = t[n - 1] - t[last]
        if d > period + TOL:
            break
        if abs(d - period) <= TOL:
            r = _scan(t, tok, k, period)
            if r and (best is None or r[0] < best):
                best = r[0]
    return None if best is None else t[best]


def detect_loop(channel_events):
    """channel_events: {name: sorted [(frame, token)]}.

    Returns dict(start=frame, period=frames, channels={name: (start, period)}) or None.
    """
    per = {}
    for name, ev in channel_events.items():
        r = channel_period(ev)
        if r:
            per[name] = r
    if not per:
        return None
    period = max(L for _, L in per.values())
    starts = {}
    for name, ev in channel_events.items():
        if len(ev) < MIN_EVENTS:
            continue
        s = channel_start_for_period(ev, period)
        if s is None:
            # a channel that does not repeat with the common period -> no reliable loop
            if name in per:
                return None
            continue
        starts[name] = s
    if not starts:
        return None
    return {"start": max(starts.values()), "period": period, "channels": per}


def song_end(channel_events, emulated_frames, silence_frames):
    """Frame of the last event if the song stops (no events in the final
    `silence_frames` of the emulation), else None."""
    last = max((ev[-1][0] for ev in channel_events.values() if ev), default=None)
    if last is None:
        return None
    if emulated_frames - last >= silence_frames:
        return last
    return None
