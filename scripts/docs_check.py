#!/usr/bin/env python3
"""Documentation link checker (make docs-check).

Scans Markdown files for relative links and verifies the targets exist; skips
external (http/https/mailto/tel), anchor-only, and links into ignored dirs. Also
checks that a set of required docs exist. Exits non-zero on any broken link or
missing required doc — lightweight, dependency-free.

Usage:
    python scripts/docs_check.py      # or: make docs-check
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_DIRS = {".git", ".venv", "mlruns", "node_modules", "outputs", "artifacts",
             ".pytest_cache", ".claude", "__pycache__"}
REQUIRED = [
    "README.md",
    "docs/architecture.md", "docs/decisions.md", "docs/research-findings.md",
    "docs/plan.md", "docs/execution.md", "docs/checklist.md",
    "systems/README.md", "data/README.md", "ml/README.md", "ai/README.md",
]


def md_files():
    for p in ROOT.rglob("*.md"):
        if SKIP_DIRS & set(p.relative_to(ROOT).parts):
            continue
        yield p


def check_links():
    broken = []
    for md in md_files():
        text = md.read_text(encoding="utf-8", errors="replace")
        for target in LINK.findall(text):
            t = target.strip()
            # Skip external schemes and anchors; file:// are machine-absolute URIs.
            if t.startswith(("http://", "https://", "mailto:", "tel:", "file:", "#")):
                continue
            t = t.split("#", 1)[0]  # strip anchor
            if not t:
                continue
            if not (md.parent / t).resolve().exists():
                broken.append((md.relative_to(ROOT), target))
    return broken


def main():
    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    broken = check_links()
    n_md = sum(1 for _ in md_files())

    print(f"docs-check: scanned {n_md} Markdown files")
    for r in missing:
        print(f"  MISSING REQUIRED: {r}")
    for md, t in broken:
        print(f"  BROKEN LINK: {md} -> {t}")

    ok = not missing and not broken
    print("OK" if ok else f"FAIL ({len(missing)} missing, {len(broken)} broken)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
