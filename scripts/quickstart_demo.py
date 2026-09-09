"""Reduced scale end to end run: data, three training phases, one action.

Runs the entire pipeline in a few minutes on CPU. It writes its dataset to
data/quickstart and its checkpoints to runs/quickstart, so it never disturbs
a full run living in data/generated and runs/main.

Usage:
    python scripts/quickstart_demo.py
"""

import subprocess
import sys

DATA = ["--data-name", "quickstart"]
RUN = ["--run", "runs/quickstart"]

STEPS = [
    [sys.executable, "scripts/make_synthetic_data.py", *DATA,
     "--months", "300", "--chart-stride", "2", "--ports-per-month", "2"],
    [sys.executable, "scripts/train_phase1_clip.py", *DATA, *RUN,
     "--epochs", "4"],
    [sys.executable, "scripts/build_dkg.py", *DATA, *RUN],
    [sys.executable, "scripts/train_phase3_vla.py", *DATA, *RUN,
     "--epochs", "15"],
]


def main() -> None:
    for step in STEPS:
        print(f"\n=== {' '.join(step[1:])} ===", flush=True)
        subprocess.run(step, check=True)

    import _bootstrap
    from mmtg.config import Paths
    from mmtg.utils import load_json

    tilts = load_json(f"{Paths().root}/runs/quickstart/tilts_history.json")
    month = next((t["month"] for t in reversed(tilts) if t["split"] == "val"),
                 tilts[-1]["month"])
    print(f"\n=== generate_trades.py --month {month} ===", flush=True)
    subprocess.run([sys.executable, "scripts/generate_trades.py",
                    *DATA, *RUN, "--month", month], check=True)
    print("\nQuickstart complete. For a full run use the default "
          "--data-name generated and --run runs/main.", flush=True)


if __name__ == "__main__":
    main()
