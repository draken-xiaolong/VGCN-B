"""Verify every original input and inference resource against the migration manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'input_manifest.json').read_text(encoding='utf-8'))
    failures = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file():
            failures.append(f"MISSING {entry['path']}")
        else:
            with path.open('rb') as stream:
                actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if path.stat().st_size != entry['bytes'] or actual_hash != entry['sha256']:
                failures.append(f"CHANGED {entry['path']}")
    print(f"Checked {len(manifest['files'])} files; {len(failures)} failures.")
    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
