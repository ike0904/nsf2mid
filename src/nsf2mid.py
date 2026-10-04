"""
nsf2mid - NSF to MIDI converter (derived from morokoshi)

Phase 1: 6502 emulation + APU register log / per-frame channel state dump.
Phase 2: note extraction (P1/P2/TRI), tempo detection, quantized MIDI output.
Phase 3: drums (noise / DPCM / triangle glides -> GM percussion), loop / end detection.

Usage:
    python nsf2mid.py FILE.nsf [-t TRACK] [-s SECONDS] [-o OUTDIR] [--info] [--dump]
                      [--keep-tail] [--no-quantize] [--unit FRAMES] [--rows-per-beat N]
                      [--loops N] [--no-loop] [--no-tri-drums] [--drum-map MAP]

バージョン履歴:
  v0.1.0 (2026-10-04) - 初版 (Phase 1: 6502 エミュレータ・APU レジスタログ・フレーム状態ダンプ)
  v0.1.1 (2026-10-04) - $4015 読み出し対応 (オホーツクに消ゆ等)・Phase 2: ノート抽出・テンポ検出・MIDI 出力
  v0.1.2 (2026-10-04) - Phase 3: ノイズ/DPCM/三角波ドラム・ループ/曲終端検出・分割エミュレーション
  v0.1.3 (2026-10-04) - ループ確定を 3 回連続一致に変更 (2 ループ後にサビが入る曲への対策)・三角波を 1 オクターブ下げて出力 (--tri-octave)
  v0.1.4 (2026-10-04) - 三角波のオクターブを元に戻す (--tri-octave 既定 0)
  v0.1.5 (2026-10-04) - 三角波の GM 音色を 38→80 (Sibelius での移調表示・トラック並べ替え対策)・--programs
  v0.1.6 (2026-10-04) - トラック名を楽器名と紛らわしくない名前に変更 (Sibelius が Triangle を打楽器と判定)
  v0.1.7 (2026-10-04) - 音色 (プログラムチェンジ) を既定で出力しない
"""

APP_VERSION = "v0.1.7"

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
from drums import (noise_hits, dmc_hits, split_triangle_drums, parse_drum_map,  # noqa: E402
                   summarize, GM_NAMES)
from loop import detect_loop, REPEATS as LOOP_REPEATS  # noqa: E402

PPQ = 480
MIDI_CH = {"P1": 0, "P2": 1, "TRI": 2}
# GM programs (0-based) for the tonal tracks; None = no program change (default).
# Notation software (Sibelius) picks instruments from the program number (e.g. 38 Synth Bass is
# notated an octave up) and re-sorts tracks by instrument family, so no program is written unless
# requested with --programs.
MIDI_PROG = {"P1": None, "P2": None, "TRI": None}
CHUNK_SEC = 60.0      # emulate in chunks; stop when a loop or the end of the song is found
SILENCE_SEC = 4.0     # this much silence after the last note = song ended
CONFIRM_SEC = 20.0    # at least this much repeat after the first loop (short loops)

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


# Track names avoid real instrument names: notation software (Sibelius) picks instruments from the
# track name, and "Triangle" became the percussion triangle (moved into the percussion section).
TRACK_NAMES = {"P1": "Ch1 Pulse", "P2": "Ch2 Pulse", "TRI": "Ch3 Wave"}
DRUM_TRACKS = (("NOI", "Ch4 Noise drums"), ("DMC", "Ch5 DPCM drums"), ("TRI", "Ch3 Wave drums"))


def build_midi(notes, drums, tmap, title, track_no, end_frame, loop, transpose, programs):
    t0 = Track(f"{title} #{track_no}" if title else f"track {track_no}")
    for tick, bpm in tmap.tempo_events():
        t0.tempo(tick, bpm)
    t0.time_signature(0, 4, 2)
    if loop:
        t0.meta(tmap(loop["start"]), 0x06, b"loopStart")
        t0.meta(tmap(loop["start"] + loop["period"]), 0x06, b"loopEnd")
    tracks = [t0]
    min_len = int(tmap.min_ticks) if tmap.quantize else max(1, int(tmap.min_ticks / 4))
    end_tick = tmap(end_frame)

    def span(start, stop):
        st = tmap(start)
        en = min(tmap(stop), end_tick)
        if en <= st:
            en = st + min_len
        return st, en

    for ch in ("P1", "P2", "TRI"):
        tr = Track(TRACK_NAMES[ch])
        mch = MIDI_CH[ch]
        if programs.get(ch) is not None:
            tr.program(0, mch, programs[ch])
        if loop and ch == "P1":
            tr.control(tmap(loop["start"]), mch, 111, 0)   # RPG Maker style loop marker
        for n in notes[ch]:
            if n.start >= end_frame:
                continue
            st, en = span(n.start, n.gate_end)
            vel = 100 if ch == "TRI" else vol_to_velocity(n.peak)
            tr.note(st, en, mch, max(0, min(127, n.pitch + transpose.get(ch, 0))), vel)
        tracks.append(tr)
    for src, name in DRUM_TRACKS:
        hits = [d for d in drums if d.src == src and d.start < end_frame]
        if not hits:
            continue
        tr = Track(name)
        for d in hits:
            st, en = span(d.start, d.end)
            tr.note(st, en, 9, d.gm, d.vel)
        tracks.append(tr)
    return tracks


def write_notes_csv(path, notes, drums, tmap, transpose):
    rows = []
    for ch, lst in notes.items():
        for n in lst:
            rows.append((n.start, ch, n))
    for d in drums:
        rows.append((d.start, "DRUM_" + d.src, d))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ch", "start_frame", "gate_end_frame", "end_frame", "gate_frames", "tail_frames",
                    "note", "midi", "peak_vol", "cue", "start_tick", "end_tick", "vols"])
        for _, ch, n in rows:
            if ch.startswith("DRUM_"):
                w.writerow([ch, n.start, n.end, n.end, n.end - n.start, 0,
                            GM_NAMES.get(n.gm, str(n.gm)), n.gm, n.vel, n.info,
                            tmap(n.start) if tmap else "", tmap(n.end) if tmap else "", ""])
                continue
            out_pitch = n.pitch + transpose.get(ch, 0)
            w.writerow([ch, n.start, n.gate_end, n.end, n.gate_end - n.start, n.end - n.gate_end,
                        note_name(out_pitch), out_pitch, n.peak, n.cue,
                        tmap(n.start) if tmap else "", tmap(n.gate_end) if tmap else "",
                        " ".join(f"{v:X}" for v in n.vols[:48])])


def analyze(player, n_frames, args, drum_map):
    """Replay the register log and extract notes / drums / loop."""
    frames = build_frames(player.writes, n_frames, player.frame_cycles, player.cpu_hz)
    notes, stats = extract_notes(frames, keep_tail=args.keep_tail)
    notes["TRI"], tri_drums = split_triangle_drums(notes["TRI"], not args.no_tri_drums)
    drums = noise_hits(frames, drum_map) + dmc_hits(frames, player.dmc_samples, player.cpu_hz, drum_map)
    drums += tri_drums
    drums.sort(key=lambda d: d.start)
    events = {ch: [(n.start, n.pitch) for n in lst] for ch, lst in notes.items()}
    events["DRUM"] = [(d.start, (d.src, d.kind)) for d in drums]
    loop = None if args.no_loop else detect_loop(events)
    return frames, notes, stats, drums, events, loop


def main():
    ap = argparse.ArgumentParser(description=f"nsf2mid {APP_VERSION} - NSF to MIDI converter")
    ap.add_argument("nsf", help="input .nsf file")
    ap.add_argument("-t", "--track", type=int, default=0, help="track number (1-based, default: start song)")
    ap.add_argument("-s", "--seconds", type=float, default=600.0,
                    help="maximum length to emulate in seconds (stops early when a loop or the end is found)")
    ap.add_argument("-o", "--outdir", default=None, help="output folder (default: <nsf folder>/nsf2mid_out)")
    ap.add_argument("--info", action="store_true", help="print header info only")
    ap.add_argument("--dump", action="store_true", help="also write register/frame/tracker dumps (Phase 1)")
    ap.add_argument("--keep-tail", action="store_true", help="keep reverb tails in note length")
    ap.add_argument("--no-quantize", action="store_true", help="do not snap notes to the detected grid")
    ap.add_argument("--unit", type=float, default=None, help="override grid unit (frames per row)")
    ap.add_argument("--rows-per-beat", type=int, default=None, help="override rows per beat (e.g. 4 = 16th rows)")
    ap.add_argument("--loops", type=int, default=1, help="how many times to write the loop body (default 1)")
    ap.add_argument("--no-loop", action="store_true", help="disable loop detection (emulate --seconds fully)")
    ap.add_argument("--tri-octave", type=int, default=0,
                    help="octave shift for triangle notes in the MIDI (default 0 = APU pitch)")
    ap.add_argument("--programs", default="",
                    help='GM programs (0-based) for P1,P2,TRI, e.g. "80,80,38"; "-" = no program change '
                         '(default: none)')
    ap.add_argument("--no-tri-drums", action="store_true", help="keep triangle glide drums as triangle notes")
    ap.add_argument("--drum-map", default="",
                    help='override drum notes, e.g. "3:0=42,12:0=36,DMC:E000:129:15=38" '
                         "(noise: periodIdx:mode, DPCM: DMC:addrHex:len:rate)")
    args = ap.parse_args()
    drum_map = parse_drum_map(args.drum_map)
    transpose = {"TRI": 12 * args.tri_octave}
    programs = dict(MIDI_PROG)
    if args.programs:
        for ch, v in zip(("P1", "P2", "TRI"), args.programs.split(",")):
            v = v.strip()
            programs[ch] = None if v == "-" else int(v)

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
    max_frames = int(args.seconds * frame_rate)
    chunk = int(CHUNK_SEC * frame_rate)
    print(f"\nEmulating track {track} (max {args.seconds:.0f} s @ {frame_rate:.3f} Hz)")
    t0 = time.time()
    player.start(track)
    n_frames = 0
    end_reason = "max length"
    prev_loop = None
    while n_frames < max_frames:
        n_frames = player.advance(min(chunk, max_frames - n_frames))
        if args.no_loop or player.aborted:
            if player.aborted:
                end_reason = "emulation aborted"
            if player.aborted:
                break
            continue
        frames, notes, stats, drums, events, loop = analyze(player, n_frames, args, drum_map)
        # accept a loop only when it is stable over two chunks and heard LOOP_REPEATS times in a row
        if (loop and prev_loop and abs(loop["start"] - prev_loop["start"]) <= 2
                and abs(loop["period"] - prev_loop["period"]) <= 2
                and n_frames - loop["start"] >= max(LOOP_REPEATS * loop["period"],
                                                    loop["period"] + CONFIRM_SEC * frame_rate)):
            end_reason = "loop found"
            break
        prev_loop = loop
        last_end = max([n.end for lst in notes.values() for n in lst] + [d.end for d in drums], default=0)
        if last_end and n_frames - last_end >= SILENCE_SEC * frame_rate:
            end_reason = "song ended"
            break
    player.finish_warnings()
    frames, notes, stats, drums, events, loop = analyze(player, n_frames, args, drum_map)
    print(f"  {n_frames} frames ({n_frames / frame_rate:.1f} s) in {time.time() - t0:.2f} s, "
          f"{len(player.writes)} register writes; stopped: {end_reason}")
    for w in player.warnings:
        print(f"  Warning: {w}")

    # ---- song range
    last_end = max([n.end for lst in notes.values() for n in lst] + [d.end for d in drums], default=n_frames)
    if loop and end_reason == "loop found":
        end_frame = loop["start"] + loop["period"] * max(1, args.loops)
        print(f"\nLoop: starts at {loop['start'] / frame_rate:.2f} s, length {loop['period'] / frame_rate:.2f} s "
              f"({loop['period']} frames); writing intro + {max(1, args.loops)} loop(s)")
    else:
        loop = None
        end_frame = last_end if end_reason == "song ended" else n_frames
        print(f"\nLoop: none ({end_reason}); song length {end_frame / frame_rate:.2f} s")

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.nsf)), "nsf2mid_out")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.nsf))[0]
    if len(base) > 40:
        base = base[:40].rstrip()
    stem = os.path.join(outdir, f"{base}_t{track:02d}")
    outputs = []

    if args.dump:
        write_writes_csv(stem + "_writes.csv", player.writes, player.frame_cycles, player.cpu_hz)
        write_frames_csv(stem + "_frames.csv", frames)
        write_tracker_txt(stem + "_tracker.txt", frames, h, track, frame_rate)
        outputs += [stem + "_writes.csv", stem + "_frames.csv", stem + "_tracker.txt"]

    print("\nNotes:")
    for ch in ("P1", "P2", "TRI"):
        lst = [n for n in notes[ch] if n.start < end_frame]
        tails = sum(1 for n in lst if n.has_tail)
        st = stats[ch]
        cue = "on" if st["trigger_cue"] else "IGNORED (driver re-triggers almost every frame)"
        print(f"  {ch:3}: {len(lst):4} notes, {tails:4} reverb tails cut, "
              f"trigger ratio {st['trigger_ratio']:.2f} -> trigger cue {cue}")
    in_range = [d for d in drums if d.start < end_frame]
    if in_range:
        print("Drums (source, kind, hits -> GM note):")
        for src, kind, cnt, gm, info in summarize(in_range):
            print(f"  {src:3} {str(kind):28} {cnt:4} -> {gm:3} {GM_NAMES.get(gm, ''):16} ({info})")

    onsets = [o for o in tempo_onsets(notes, frames) if o < end_frame]
    onsets += [d.start for d in in_range if d.src == "DMC"]
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
        write_smf(stem + ".mid", build_midi(notes, drums, tmap, h.title, track, end_frame, loop, transpose, programs), PPQ)
        outputs.insert(0, stem + ".mid")

    write_notes_csv(stem + "_notes.csv", notes, drums, tmap, transpose)
    outputs.append(stem + "_notes.csv")
    print("\nOutput:")
    for o in outputs:
        print(f"  {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
