#!/usr/bin/env python3
import json
import os
import re
import subprocess
import tempfile
import shutil
import sys


def main() -> int:
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(repo_dir, "test.json")
    script = os.path.join(repo_dir, "beautiful_mermaid.py")

    if not os.path.isfile(input_file):
        print(f"test.json not found at {input_file}", file=sys.stderr)
        return 1
    if not os.path.isfile(script):
        print(f"beautiful_mermaid.py not found at {script}", file=sys.stderr)
        return 1

    work_dir = None
    try:
        work_dir = tempfile.mkdtemp(prefix="beautiful-mermaid-tests.")
        with open(input_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not samples:
            print("No samples found.", file=sys.stderr)
            return 1

        for i, sample in enumerate(samples, start=1):
            title = sample.get("title") or f"Sample {i}"
            source = sample.get("source") or ""
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
            if not safe:
                safe = f"sample_{i}"
            path = os.path.join(work_dir, f"{i:03d}_{safe}.mmd")
            with open(path, "w", encoding="utf-8") as f:
                f.write(source.strip() + "\n")

            print("=" * 80)
            print(f"[{i:03d}] {title}")
            print("-" * 80)
            subprocess.run(["python3", script, path], check=False)

        print("=" * 80)
        print(f"Rendered {len(samples)} samples from {os.path.basename(input_file)}")
        print(f"Temporary files in: {work_dir}")
        return 0
    finally:
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
