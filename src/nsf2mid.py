"""
nsf2mid - NSF to MIDI converter (derived from morokoshi)

Phase 1: 6502 emulation + APU register log / per-frame channel state dump.

Usage:
    python nsf2mid.py FILE.nsf [-t TRACK] [-s SECONDS] [-o OUTDIR] [--info]

バージョン履歴:
  v0.1.0 (2026-10-04) - 初版 (Phase 1: 6502 エミュレータ・APU レジスタログ・フレーム状態ダンプ)
"""

APP_VERSION = "v0.1.0"

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsf import NSFPlayer  # noqa: E402
from apu_state import APUState, note_name  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser(description=f"nsf2mid {APP_VERSION} - NSF register logger (Phase 1)")
    ap.add_argument("nsf", help="input .nsf file")
    ap.add_argument("-t", "--track", type=int, default=0, help="track number (1-based, default: start song)")
    ap.add_argument("-s", "--seconds", type=float, default=120.0, help="length to emulate in seconds")
    ap.add_argument("-o", "--outdir", default=None, help="output folder (default: <nsf folder>/nsf2mid_out)")
    ap.add_argument("--info", action="store_true", help="print header info only")
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
    write_writes_csv(stem + "_writes.csv", writes, player.frame_cycles, player.cpu_hz)
    write_frames_csv(stem + "_frames.csv", frames)
    write_tracker_txt(stem + "_tracker.txt", frames, h, track, frame_rate)
    print(f"\nOutput:\n  {stem}_writes.csv\n  {stem}_frames.csv\n  {stem}_tracker.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
