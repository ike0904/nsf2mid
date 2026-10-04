"""
Regression test over a sample of NSF files.

Runs src/nsf2mid.py on 40 randomly chosen (seed 1) NSF archives plus a few fixed titles from
the morokoshi test collection, and prints per file: tempo segments, meters, number of
tempo-tracked segments, loop / end result and the difference between the MIDI length and
the detected song length (a timing check of the tempo map).

Usage (from the project root):
    python tools/regress.py [NSF_ZIP_FOLDER] > result.txt
Compare two result files with a diff to see what a change affected.
"""

import os
import random
import re
import subprocess
import sys
import tempfile
import zipfile

import importlib.util
if importlib.util.find_spec("mido") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mido"])
import mido  # noqa: E402

DEFAULT_DIR = r"E:\Users\takashi\Desktop\ClaudeCode\morokoshi\tmp\nsf_downloads"
FIXED = ("Mega Man 2 (", "Super Mario Bros. (", "Castlevania (", "Final Fantasy (", "Gradius (")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    names = sorted(os.listdir(src_dir))
    random.seed(1)
    pick = random.sample(names, 40) + [n for n in names if n.startswith(FIXED)]
    tmp = tempfile.mkdtemp()
    for n in pick:
        z = zipfile.ZipFile(os.path.join(src_dir, n))
        nsfs = [m for m in z.namelist() if m.lower().endswith(".nsf")]
        if not nsfs:
            continue
        p = os.path.join(tmp, "x.nsf")
        with open(p, "wb") as f:
            f.write(z.read(nsfs[0]))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "nsf2mid.py"), p, "-o", tmp, "-s", "600"],
                           capture_output=True, text=True)
        out = r.stdout.splitlines()
        segs = [l.strip() for l in out if l.strip().startswith("from ")]
        bpm = " ".join(l.split(":")[1].split("BPM")[0].strip() for l in segs)
        meters = " ".join(m.group(1) for l in segs for m in [re.search(r"(\d+/\d+)", l.split("fit")[-1])] if m)
        tracked = sum(1 for l in segs if "tracked" in l)
        stop = next((l.split("stopped: ")[1] for l in out if "stopped:" in l), "?")
        loop = next((l for l in out if l.startswith("Loop:")), "")
        m = re.search(r"starts at ([\d.]+) s, length ([\d.]+) s", loop)
        if m:
            expect = float(m.group(1)) + float(m.group(2))
        else:
            m2 = re.search(r"song length ([\d.]+)", loop)
            expect = float(m2.group(1)) if m2 else None
        mid = next((l.strip() for l in out if l.strip().endswith(".mid")), None)
        mlen = mido.MidiFile(mid).length if mid and os.path.exists(mid) else None
        dev = f"{mlen - expect:+.2f}s" if (mlen is not None and expect) else "-"
        err = r.stderr.strip().splitlines()[-1] if r.returncode else ""
        print(f"{n[:34]:34} bpm={bpm[:30]:30} meter={meters[:12]:12} tracked={tracked} "
              f"stop={stop:12} midi-vs-song={dev} {err}", flush=True)


if __name__ == "__main__":
    main()
