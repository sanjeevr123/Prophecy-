from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ROOT / "pipeline" / "train_selector.py",
    ROOT / "pipeline" / "reconstruct_game4.py",
    ROOT / "pipeline" / "score_game4.py",
    ROOT / "pipeline" / "generate_game4.py",
    ROOT / "pipeline" / "validate_backend.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n=== {step.name} ===")
        subprocess.run([sys.executable, str(step)], cwd=ROOT, check=True)

    print("\nPipeline complete. The verified question set is in data/out/questions.json.")


if __name__ == "__main__":
    main()
