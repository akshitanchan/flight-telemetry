#!/usr/bin/env python3
"""
Manifest generation and verification for bootstrapped data files.

Creates a JSON manifest with SHA-256 checksums and file metadata.
Used to verify data integrity after bootstrap or before processing.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_FILENAME = "manifest.json"

# Files to skip in manifest (not data files)
SKIP_FILES = {MANIFEST_FILENAME, ".gitkeep"}


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(data_dir: Path) -> Path:
    """Generate a manifest.json with checksums for all files in data_dir.

    Returns path to the manifest file.
    """
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "files": {},
    }

    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.name not in SKIP_FILES:
            manifest["files"][path.name] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }

    manifest_path = data_dir / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def verify_manifest(data_dir: Path) -> bool:
    """Verify files against manifest checksums.

    Returns True if all files match, False otherwise.
    """
    manifest_path = data_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"  ✗ No manifest found at {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_ok = True
    for filename, info in manifest["files"].items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ✗ Missing: {filename}")
            all_ok = False
            continue

        actual_sha = _sha256(filepath)
        if actual_sha != info["sha256"]:
            print(f"  ✗ Checksum mismatch: {filename}")
            print(f"    expected: {info['sha256']}")
            print(f"    actual:   {actual_sha}")
            all_ok = False
        else:
            print(f"  ✓ {filename} ({info['size_bytes']:,} bytes)")

    if all_ok:
        print(f"\n  All {len(manifest['files'])} files verified.")
    else:
        print(f"\n  ⚠ Verification failed.")

    return all_ok
