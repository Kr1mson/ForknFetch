import subprocess
import sys
from pathlib import Path

STAGES = [
    ("Preprocessing", "stages/preprocessor.py"),
    ("Embedding",     "stages/embedder.py"),
]

def run(script: str):
    result = subprocess.run(
        [sys.executable, script],
        check=False
    )
    if result.returncode != 0:
        print(f"\nFailed at: {script}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 40)
    print("Fork&Fetch — Build Pipeline")
    print("=" * 40)

    for name, script in STAGES:
        if not Path(script).exists():
            print(f"\nSkipping {name} — {script} not found")
            continue
        print(f"\n{name} ({script})")
        print("-" * 40)
        run(script)
        print(f"✓ {name} complete")

    print("\n" + "=" * 40)
    print("Pipeline complete — chroma_db ready")
    print("=" * 40)