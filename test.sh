#!/bin/zsh
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")" && pwd)
input_file="$repo_dir/samples-data.ts"
script="$repo_dir/beautiful-mermaid.py"

if [[ ! -f "$input_file" ]]; then
  echo "samples-data.ts not found at $input_file" >&2
  exit 1
fi
if [[ ! -f "$script" ]]; then
  echo "beautiful-mermaid.py not found at $script" >&2
  exit 1
fi

work_dir=$(mktemp -d "/tmp/beautiful-mermaid-tests.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

python3 - <<'PY' "$input_file" "$script" "$work_dir"
import os
import re
import sys
import subprocess

input_file, script, work_dir = sys.argv[1:4]

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Simple state machine to pair titles with sources
samples = []
current_title = None

for match in re.finditer(r"title:\s*'([^']+)'|source:\s*`(.*?)`", text, re.DOTALL):
    title = match.group(1)
    source = match.group(2)
    if title is not None:
        current_title = title.strip()
    elif source is not None:
        if current_title is None:
            continue
        samples.append((current_title, source))
        current_title = None

if not samples:
    print("No samples found.", file=sys.stderr)
    sys.exit(1)

for i, (title, source) in enumerate(samples, start=1):
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
PY
