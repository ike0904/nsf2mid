"""
Note extraction from per-frame APU snapshots (tonal channels: P1, P2, TRI).

Design (see docs/spec.md "Phase 2 ノート判定"):
  Note-on cues
    1. hi-period register write ($4003/$4007/$400B) = hardware trigger.
       If the trigger frame is silent (SMB style "trigger, vol 0, sound next frame")
       the note starts at the trigger frame provided the pitch matches.
    2. pitch moves to another semitone and stays there (>= 2 frames).
    3. silence -> sound.
  NOT note-on
    - duty change (DQ2 duty envelope), volume rises / tremolo (DQ3, Okhotsk ni Kiyu).
  Reverb tail ("echo")
    Many drivers (DQ2-4, SMB USA) keep the pitch sounding at a low volume after the
    musical note has ended. The tail start (gate-off) is the first frame g where:
      - every later volume in the note is <= vol[g], and
      - vol[g] <= peak/2, and
      - vol[g] dropped by >= 3 from the previous frame, or vol[g] <= max(2, peak/4)
      - at least 2 frames of tail remain
    By default the MIDI note ends at g; keep_tail=True keeps the full length.
"""

PITCH_SPLIT = 0.75        # semitones away from the note's pitch to count as a new note
PITCH_HOLD = 2            # frames the new pitch must persist
TRIGGER_SPAM_RATIO = 0.6  # trigger on more than this share of sounding frames -> ignore triggers


class Note:
    __slots__ = ("ch", "start", "end", "gate_end", "pitch", "vols", "duties", "pitches", "cue")

    def __init__(self, ch, start, pitch, cue):
        self.ch = ch
        self.start = start          # first frame
        self.end = None             # frame after last sounding frame
        self.gate_end = None        # frame where the reverb tail starts (== end if no tail)
        self.pitch = pitch          # MIDI note number
        self.vols = []
        self.duties = []
        self.pitches = []           # per-frame pitch (float semitones, None while silent)
        self.cue = cue              # what started the note: trig / pitch / sound

    @property
    def peak(self):
        g = self.gate_end - self.start if self.gate_end is not None else len(self.vols)
        return max(self.vols[:max(1, g)]) if self.vols else 0

    @property
    def has_tail(self):
        return self.gate_end is not None and self.gate_end < self.end


def _frame_info(state, written, ch):
    c = state[ch]
    trig = 3 in written[ch]
    if ch == "TRI":
        vol = 15 if c["vol"] else 0
        duty = None
    else:
        vol = c["vol"]
        duty = c["duty"]
    note = c["note"]
    if note is not None and not (0 <= note <= 127):
        note = None
    if note is None:
        vol = 0
    return vol, note, trig, duty


def find_gate_end(vols):
    """Return index (relative to note start) where the reverb tail starts, or len(vols)."""
    n = len(vols)
    if n < 3:
        return n
    peak = max(vols)
    # suffix maximum: max(vols[i:])
    suf = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = max(vols[i], suf[i + 1])
    for g in range(1, n - 1):
        v = vols[g]
        if suf[g] > v:          # volume rises again later -> not a tail
            continue
        if v * 2 > peak:
            continue
        drop = vols[g - 1] - v
        if drop >= 3 or v <= max(2, peak / 4.0):
            return g
    return n


def extract_channel(frames, ch, keep_tail=False):
    infos = [_frame_info(f["state"], f["written"], ch) for f in frames]

    sounding = sum(1 for v, _, _, _ in infos if v > 0)
    trigs = sum(1 for v, _, t, _ in infos if t and v > 0)
    use_trig = not (sounding and trigs / sounding > TRIGGER_SPAM_RATIO)

    notes = []
    cur = None
    pending_trig = None     # (frame, note) of a trigger on a silent frame
    off_count = 0           # consecutive frames away from current pitch

    def close(at):
        nonlocal cur
        if cur is None:
            return
        cur.end = at
        g = find_gate_end(cur.vols)
        cur.gate_end = cur.end if keep_tail else cur.start + g
        notes.append(cur)
        cur = None

    for f, (vol, note, trig, duty) in enumerate(infos):
        trig = trig and use_trig
        if vol == 0:
            if trig and note is not None:
                pending_trig = (f, note)
            elif pending_trig and f - pending_trig[0] > 2:
                pending_trig = None
            close(f)
            off_count = 0
            continue

        rp = int(round(note))
        start_new = None
        if cur is None:
            start_new = "sound"
            start_frame = f
            if trig:
                start_new = "trig"
            elif pending_trig and abs(pending_trig[1] - note) < 0.5 and f - pending_trig[0] <= 2:
                start_new = "trig"
                start_frame = pending_trig[0]
        elif trig:
            start_new = "trig"
            start_frame = f
        else:
            if abs(note - cur.pitch) >= PITCH_SPLIT:
                off_count += 1
                if off_count >= PITCH_HOLD:
                    # split: the new note began off_count-1 frames ago
                    start_new = "pitch"
                    start_frame = f - off_count + 1
            else:
                off_count = 0
        pending_trig = None

        if start_new:
            if cur is not None and start_frame < f:
                # move the already-collected frames of the new pitch out of the old note
                k = f - start_frame
                moved_v = cur.vols[-k:]
                moved_d = cur.duties[-k:]
                moved_p = cur.pitches[-k:]
                del cur.vols[-k:]
                del cur.duties[-k:]
                del cur.pitches[-k:]
                close(start_frame)
                cur = Note(ch, start_frame, rp, start_new)
                cur.vols.extend(moved_v)
                cur.duties.extend(moved_d)
                cur.pitches.extend(moved_p)
            else:
                close(f)
                cur = Note(ch, start_frame, rp, start_new)
                # frames between a silent trigger and the first sounding frame
                for _ in range(f - start_frame):
                    cur.vols.append(0)
                    cur.duties.append(duty)
                    cur.pitches.append(None)
            off_count = 0
        cur.vols.append(vol)
        cur.duties.append(duty)
        cur.pitches.append(note)

    close(len(infos))
    return notes, {"trigger_cue": use_trig, "trigger_ratio": (trigs / sounding) if sounding else 0.0}


def merge_slides(notes, keep_tail=False):
    """Merge fast pitch glides into one note.

    A glide shows up as a chain of short notes split by pitch change (e.g. the
    Mega Man 2 "triangle kick": B4 -> G4 -> D#4 -> C4 -> A#3 within 10 frames).
    A note whose length is <= SLIDE_MAX frames absorbs the following contiguous
    pitch-split note.
    """
    out = []
    for n in notes:
        if out:
            p = out[-1]
            # a long pitch-split note after a glide is the held target note: keep it separate
            if (n.cue == "pitch" and n.start == p.end and n.end - n.start <= SLIDE_MAX
                    and (p.end - p.start <= SLIDE_MAX or "slide" in p.cue)):
                p.vols.extend(n.vols)
                p.duties.extend(n.duties)
                p.pitches.extend(n.pitches)
                p.end = n.end
                g = find_gate_end(p.vols)
                p.gate_end = p.end if keep_tail else p.start + g
                if p.cue == "pitch":
                    p.cue = "slide"
                elif p.cue != "slide":
                    p.cue = p.cue + "+slide"
                continue
        out.append(n)
    return [n for n in out if n.end - n.start >= MIN_FRAMES]


SLIDE_MAX = 3      # frames
ONSET_MERGE = 2    # tempo onsets closer than this (frames) are one event
MIN_FRAMES = 2     # notes shorter than this are dropped


def extract_notes(frames, channels=("P1", "P2", "TRI"), keep_tail=False):
    result = {}
    stats = {}
    for ch in channels:
        lst, stats[ch] = extract_channel(frames, ch, keep_tail)
        result[ch] = merge_slides(lst, keep_tail)
    return result, stats


def noise_onsets(frames):
    """Frames where the noise channel is (re)triggered and audible. Used for tempo only (Phase 2)."""
    out = []
    for f in frames:
        if 3 in f["written"]["NOI"] and f["state"]["NOI"]["vol"] > 0:
            out.append(f["frame"])
    return out


def tempo_onsets(notes, frames):
    """Onsets used for tempo detection.

    Pitch-split notes are excluded (glides, legato runs are less reliable), and
    onsets within ONSET_MERGE frames of each other are merged (some drivers update the
    channels on different frames).
    """
    raw = [n.start for lst in notes.values() for n in lst if not n.cue.startswith("pitch")]
    raw += noise_onsets(frames)
    raw = sorted(set(raw))
    out = []
    for o in raw:
        if out and o - out[-1] <= ONSET_MERGE:
            continue
        out.append(o)
    return out
