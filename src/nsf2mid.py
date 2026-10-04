"""
nsf2mid - NSF to MIDI converter (derived from morokoshi)

Phase 1: 6502 emulation + APU register log / per-frame channel state dump.
Phase 2: note extraction (P1/P2/TRI), tempo detection, quantized MIDI output.

Usage:
    python nsf2mid.py FILE.nsf [-t TRACK] [-s SECONDS] [-o OUTDIR] [--info] [--dump]
                      [--keep-tail] [--no-quantize] [--unit FRAMES] [--rows-per-beat N]

バージョン履歴:
  v0.1.0 (2026-10-04) - 初版 (Phase 1: 6502 エミュレータ・APU レジスタログ・フレーム状態ダンプ)
  v0.1.1 (2026-10-04) - $4015 読み出し対応 (オホーツクに消ゆ等)・Phase 2: ノート抽出・テンポ検出・MIDI 出力
"""

APP_VERSION = "v0.1.1"

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsf import NSFPlayer  # noqa: E402
from apu_state import APUState, note_name  # noqa: E402
from notes import extract_notes, tempo_onsets  # noqa: E402
from tempo import detect_tempo_map, TickMap  # noqa: E402
from midi import Track, write_smf  # noqa: E402

PPQ = 480
MIDI_CH = {"P1": 0, "P2": 1, "TRI": 2}
MIDI_PROG = {"P1": 80, "P2": 80, "TRI": 38}   # GM: Lead 1 (square), Synth Bass 1

REG_NAMES = {
    0x4000: "P1_CTRL", 0x4001: "P1_SWEEP", 0x4002: "P1_LO", 0x4003: "P1_HI",
    0x4004: "P2_CTRL", 0x4005: "P2_SWEEP", 0x4006: "P2_LO", 0x4007: "P2_HI",
    0x4008: "TRI_LIN", 0x4009: "TRI_UNUSED", 0x400A: "TRI_LO", 0x400B: "TRI_HI",
    0x400C: "NOI_CTRL", 0x400D: "NOI_UNUSED", 0x400E: "NOI_PERIOD", 0x400F: "NOI_LEN",
    0x4010: "DMC_CTRL", 0x4011: "DMC_DAC", 0x4012: "DMC_ADDR", 0x4013: "DMC_LEN",
    0x4014: "OAM_DMA", 0x4015: "APU_STATUS", 0x4016: "JOY1", 0x4017: "FRAME_CNT",
}

CH_RANGES = (("P1", 0x4000), ("P2", 0x4004), ("TRI", 0x4008), ("NOI", 0x400C), ("DMC", 0x4010))


def ch_of(addr):
    for name, base in CH_RANGES:
        if base <= addr < base + 4:
            return name, addr - base
    return None, None


def build_frames(writes, n_frames, frame_cycles, cpu_hz):
    """Replay writes through the APU model; return per-frame snapshots.

    Each entry: dict(frame, time, state, written={ch: set(reg)}, status_writes)
    """
    apu = APUState(cpu_hz)
    frames = []
    wi = 0
    nw = len(writes)
    for f in range(n_frames):
        end = (f + 1) * frame_cycles
        written = {"P1": set(), "P2": set(), "TRI": set(), "NOI": set(), "DMC": set()}
        status = []
        while wi < nw and writes[wi][0] < end:
            cyc, addr, val = writes[wi]
            apu.write(cyc, addr, val)
            ch, reg = ch_of(addr)
            if ch:
                written[ch].add(reg)
            elif addr == 0x4015:
                status.append(val)
            wi += 1
        apu.advance(end)
        frames.append({
            "frame": f,
            "time": f * frame_cycles / cpu_hz,
            "state": apu.snapshot(),
            "written": written,
            "status": status,
        })
    return frames


def cents_of(note):
    if note is None:
        return 0
    return int(round((note - round(note)) * 100))


def write_writes_csv(path, writes, frame_cycles, cpu_hz):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cycle_in_frame", "time_s", "addr", "value", "value_bin", "reg"])
        for cyc, addr, val in writes:
            fr = int(cyc // frame_cycles) if cyc >= 0 else -1
            cin = cyc - fr * frame_cycles if fr >= 0 else cyc
            w.writerow([fr, int(cin), f"{max(cyc, 0) / cpu_hz:.5f}", f"${addr:04X}", f"${val:02X}",
                        f"{val:08b}", REG_NAMES.get(addr, "?")])


def write_frames_csv(path, frames):
    cols = ["frame", "time_s"]
    for ch in ("P1", "P2"):
        cols += [f"{ch}_note", f"{ch}_cents", f"{ch}_period", f"{ch}_vol", f"{ch}_duty",
                 f"{ch}_envvol", f"{ch}_const", f"{ch}_len", f"{ch}_sweep", f"{ch}_written"]
    cols += ["TRI_note", "TRI_cents", "TRI_period", "TRI_on", "TRI_linear", "TRI_len", "TRI_written"]
    cols += ["NOI_period", "NOI_mode", "NOI_vol", "NOI_envvol", "NOI_const", "NOI_len", "NOI_written"]
    cols += ["DMC_playing", "DMC_addr", "DMC_len", "DMC_rate", "DMC_loop", "DMC_starts", "DMC_written",
             "STATUS_writes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for fr in frames:
            s = fr["state"]
            wr = fr["written"]
            row = [fr["frame"], f"{fr['time']:.4f}"]
            for ch in ("P1", "P2"):
                c = s[ch]
                row += [note_name(c["note"]), cents_of(c["note"]), c["period"], c["vol"], c["duty"],
                        c["reg_vol"], int(c["const"]), c["length"], int(c["sweep"]),
                        "".join(str(r) for r in sorted(wr[ch]))]
            t = s["TRI"]
            row += [note_name(t["note"]), cents_of(t["note"]), t["period"], int(t["vol"] > 0),
                    t["linear"], t["length"], "".join(str(r) for r in sorted(wr["TRI"]))]
            n = s["NOI"]
            row += [f"{n['period_idx']:X}", n["mode"], n["vol"], n["reg_vol"], int(n["const"]),
                    n["length"], "".join(str(r) for r in sorted(wr["NOI"]))]
            d = s["DMC"]
            row += [int(d["playing"]), f"${d['addr']:04X}", d["len"], d["rate"], int(d["loop"]),
                    d["starts"], "".join(str(r) for r in sorted(wr["DMC"])),
                    " ".join(f"${v:02X}" for v in fr["status"])]
            w.writerow(row)


def _tone_cell(c, written, prev):
    """Tracker cell for a pulse channel.

    Markers: '*' = $4003/$4007 written (phase reset + envelope restart),
             'd' = duty changed, 'p' = period changed without hi-reg write,
             '~' = pitch more than 20 cents off the tempered note.
    """
    if c["vol"] == 0:
        if 3 in written:
            # triggered but silent this frame (software envelope starting at 0)
            return f"{note_name(c['note'])} v0 D{c['duty']} *  "
        return f"{'...':<14}"
    mark = ""
    mark += "*" if 3 in written else " "
    mark += "d" if prev is not None and prev["duty"] != c["duty"] else " "
    mark += "p" if prev is not None and prev["period"] != c["period"] and 3 not in written else " "
    off = "~" if abs(cents_of(c["note"])) > 20 else " "
    return f"{note_name(c['note'])}{off}v{c['vol']:X} D{c['duty']} {mark}"


def write_tracker_txt(path, frames, header, track, frame_rate):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# nsf2mid {APP_VERSION} tracker view\n")
        f.write(f"# {header.title} / {header.artist} / track {track} / {frame_rate:.3f} frames/s\n")
        f.write("# Pulse cell: NOTE[~] vVOL Dduty [*][d][p]\n")
        f.write("#   * = hi-period reg written (phase reset/env restart), d = duty change,\n")
        f.write("#   p = period change w/o hi-reg write, ~ = >20 cents off pitch\n")
        f.write("# TRI cell: NOTE[~] [*]   NOI cell: Pperiod Mmode vVOL [*]   DMC: sample addr when (re)started\n\n")
        f.write(f"{'frame':>6} {'time':>8} | {'P1':<14} | {'P2':<14} | {'TRI':<8} | {'NOI':<10} | DMC\n")
        prev = None
        for fr in frames:
            s = fr["state"]
            wr = fr["written"]
            p1 = _tone_cell(s["P1"], wr["P1"], prev and prev["P1"])
            p2 = _tone_cell(s["P2"], wr["P2"], prev and prev["P2"])
            t = s["TRI"]
            if t["vol"]:
                off = "~" if abs(cents_of(t["note"])) > 20 else " "
                tri = f"{note_name(t['note'])}{off} {'*' if 3 in wr['TRI'] else ' '}  "
            else:
                tri = f"{'...':<8}"
            n = s["NOI"]
            if n["vol"]:
                noi = f"P{n['period_idx']:X} M{n['mode']} v{n['vol']:X} {'*' if 3 in wr['NOI'] else ' '}"
            else:
                noi = f"{'...':<10}"
            d = s["DMC"]
            dmc = ""
            if prev is not None and d["starts"] != prev["DMC"]["starts"]:
                dmc = f"${d['addr']:04X}"
            elif d["playing"]:
                dmc = " |"
            f.write(f"{fr['frame']:>6} {fr['time']:>8.3f} | {p1} | {p2} | {tri} | {noi} | {dmc}\n")
            prev = s


def vol_to_velocity(v):
    return max(1, min(127, int(round(v * 127 / 15))))


def build_midi(notes, tmap, title, track_no):
    t0 = Track(f"{title} #{track_no}" if title else f"track {track_no}")
    for tick, bpm in tmap.tempo_events():
        t0.tempo(tick, bpm)
    t0.time_signature(0, 4, 2)
    tracks = [t0]
    for ch in ("P1", "P2", "TRI"):
        tr = Track({"P1": "Pulse 1", "P2": "Pulse 2", "TRI": "Triangle"}[ch])
        mch = MIDI_CH[ch]
        tr.program(0, mch, MIDI_PROG[ch])
        for n in notes[ch]:
            st = tmap(n.start)
            en = tmap(n.gate_end)
            if en <= st:
                en = st + (int(tmap.min_ticks) if tmap.quantize else max(1, int(tmap.min_ticks / 4)))
            vel = 100 if ch == "TRI" else vol_to_velocity(n.peak)
            tr.note(st, en, mch, n.pitch, vel)
        tracks.append(tr)
    return tracks


def write_notes_csv(path, notes, tmap):
    rows = []
    for ch, lst in notes.items():
        for n in lst:
            rows.append((n.start, ch, n))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ch", "start_frame", "gate_end_frame", "end_frame", "gate_frames", "tail_frames",
                    "note", "midi", "peak_vol", "cue", "start_tick", "end_tick", "vols"])
        for _, ch, n in rows:
            w.writerow([ch, n.start, n.gate_end, n.end, n.gate_end - n.start, n.end - n.gate_end,
                        note_name(n.pitch), n.pitch, n.peak, n.cue,
                        tmap(n.start) if tmap else "", tmap(n.gate_end) if tmap else "",
                        " ".join(f"{v:X}" for v in n.vols[:48])])


def main():
    ap = argparse.ArgumentParser(description=f"nsf2mid {APP_VERSION} - NSF to MIDI converter")
    ap.add_argument("nsf", help="input .nsf file")
    ap.add_argument("-t", "--track", type=int, default=0, help="track number (1-based, default: start song)")
    ap.add_argument("-s", "--seconds", type=float, default=120.0, help="length to emulate in seconds")
    ap.add_argument("-o", "--outdir", default=None, help="output folder (default: <nsf folder>/nsf2mid_out)")
    ap.add_argument("--info", action="store_true", help="print header info only")
    ap.add_argument("--dump", action="store_true", help="also write register/frame/tracker dumps (Phase 1)")
    ap.add_argument("--keep-tail", action="store_true", help="keep reverb tails in note length")
    ap.add_argument("--no-quantize", action="store_true", help="do not snap notes to the detected grid")
    ap.add_argument("--unit", type=float, default=None, help="override grid unit (frames per row)")
    ap.add_argument("--rows-per-beat", type=int, default=None, help="override rows per beat (e.g. 4 = 16th rows)")
    args = ap.parse_args()

    player = NSFPlayer(args.nsf)
    h = player.header
    print(f"nsf2mid {APP_VERSION}")
    print(h.summary())
    if args.info:
        return 0

    track = args.track or h.start_song
    if not 1 <= track <= h.total_songs:
        print(f"Error: track {track} out of range 1..{h.total_songs}")
        return 1

    frame_rate = player.cpu_hz / player.frame_cycles
    n_frames = int(args.seconds * frame_rate)
    print(f"\nEmulating track {track}: {n_frames} frames ({args.seconds:.1f} s @ {frame_rate:.3f} Hz)")
    t0 = time.time()
    writes = player.run(track, n_frames)
    t1 = time.time()
    print(f"  CPU emulation: {t1 - t0:.2f} s, {len(writes)} register writes")
    for w in player.warnings:
        print(f"  Warning: {w}")

    frames = build_frames(writes, n_frames, player.frame_cycles, player.cpu_hz)
    print(f"  APU state replay: {time.time() - t1:.2f} s")

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.nsf)), "nsf2mid_out")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.nsf))[0]
    if len(base) > 40:
        base = base[:40].rstrip()
    stem = os.path.join(outdir, f"{base}_t{track:02d}")
    outputs = []

    if args.dump:
        write_writes_csv(stem + "_writes.csv", writes, player.frame_cycles, player.cpu_hz)
        write_frames_csv(stem + "_frames.csv", frames)
        write_tracker_txt(stem + "_tracker.txt", frames, h, track, frame_rate)
        outputs += [stem + "_writes.csv", stem + "_frames.csv", stem + "_tracker.txt"]

    # ---- Phase 2: notes + tempo + MIDI
    notes, stats = extract_notes(frames, keep_tail=args.keep_tail)
    print("\nNotes:")
    for ch in ("P1", "P2", "TRI"):
        lst = notes[ch]
        tails = sum(1 for n in lst if n.has_tail)
        st = stats[ch]
        cue = "on" if st["trigger_cue"] else "IGNORED (driver re-triggers almost every frame)"
        print(f"  {ch:3}: {len(lst):4} notes, {tails:4} reverb tails cut, "
              f"trigger ratio {st['trigger_ratio']:.2f} -> trigger cue {cue}")

    onsets = tempo_onsets(notes, frames)
    tempo = detect_tempo_map(onsets, frame_rate, args.rows_per_beat, args.unit)
    tmap = None
    if tempo is None:
        print("\nTempo: not enough notes to detect")
    else:
        g = tempo["global"]
        print(f"\nTempo: grid unit {g['unit']:g} frames (fit {g['score'] * 100:.1f}%), "
              f"{'ternary (triplet/shuffle)' if tempo['ternary'] else 'binary'}")
        for sg in tempo["segments"]:
            print(f"  from {sg.start / frame_rate:7.2f} s: {sg.bpm:7.2f} BPM  "
                  f"(unit {sg.unit:.4g} frames x {sg.rows} rows/beat, fit {sg.score * 100:.1f}%)")
            if sg.score < 0.8:
                print("    Warning: poor grid fit (rubato or unsupported rhythm?)")
        tmap = TickMap(tempo, PPQ, not args.no_quantize)
        write_smf(stem + ".mid", build_midi(notes, tmap, h.title, track), PPQ)
        outputs.insert(0, stem + ".mid")

    write_notes_csv(stem + "_notes.csv", notes, tmap)
    outputs.append(stem + "_notes.csv")
    print("\nOutput:")
    for o in outputs:
        print(f"  {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
