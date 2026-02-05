#!/usr/bin/env python3
""" Executes Phase 2, Phase 3, and Phase 4 consecutively
output csvs are written to "tBuOCl.tsv"


"""

import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))                        # src/dev_area/
WORK_DIR = os.path.normpath(os.path.join(_HERE, "..", "main", "tBuOCl"))  # src/main/tBuOCl/

# (label, path relative to _HERE, extra CLI args forwarded to the script)
STEPS = [
    ("Phase 2 – Orbach",  os.path.join("phase2", "tBuOCl", "tBuOCl_orbach.py"),  []),
    ("Phase 2 – QTM",     os.path.join("phase2", "tBuOCl", "tBuOCl_qtm.py"),     []),
    ("Phase 2 – Raman",   os.path.join("phase2", "tBuOCl", "tBuOCl_raman.py"),   []),
    ("Phase 3 – MC",      os.path.join("phase3", "tBuOCl", "tBuOCl_mc.py"),      ["--clear"]),
    ("Phase 4 – Global",  os.path.join("phase4", "tBuOCl", "tBuOCl_global.py"),  []),
]


def main():
    if not os.path.isdir(WORK_DIR):
        print(f"ERROR: working directory not found: {WORK_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Working directory : {WORK_DIR}")
    print(f"Python            : {sys.executable}\n")

    for idx, (label, rel_script, extra_args) in enumerate(STEPS):
        script_path = os.path.normpath(os.path.join(_HERE, rel_script))
        if not os.path.isfile(script_path):
            print(f"ERROR: script not found: {script_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print(f"  [{idx}] {label}")
        print(f"{'=' * 60}")

        cmd = [sys.executable, script_path] + extra_args
        result = subprocess.run(cmd, cwd=WORK_DIR)

        if result.returncode != 0:
            print(f"\nERROR: [{idx}] {label} exited with code {result.returncode}. "
                  "Aborting pipeline.", file=sys.stderr)
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
