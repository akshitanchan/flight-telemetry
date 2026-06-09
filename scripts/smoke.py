#!/usr/bin/env python3
"""Cross-layer smoke test (W3.4).

Runs one bounded, offline check per layer through the existing make targets and
records a pass/fail + timing summary. Proves the whole platform still runs
end-to-end from a single command after all layers have landed.

Deliberately excludes anything slow or external (no cloud, no large data, no
Ollama) — it chains the deterministic, sample-sized targets so it is safe for CI.

Usage:
    python scripts/smoke.py      # or: make smoke
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "smoke_report.json"

# (label, make target) — ordered so each layer is represented.
STEPS = [
    ("contracts: schema validation", "test-contracts"),
    ("systems: sample ingestion (replay)", "systems-replay-sample"),
    ("data: local transforms (silver + gold)", "data-local-gold"),
    ("systems: spatiotemporal benchmark", "systems-benchmark-small"),
    ("ml: serving smoke", "ml-serve-smoke"),
    ("ai: eval fixtures", "ai-eval-fixtures"),
    ("ai: answer-path eval", "ai-eval-small"),
]


def run_steps():
    results = []
    for label, target in STEPS:
        t0 = time.perf_counter()
        proc = subprocess.run(["make", target], cwd=ROOT,
                              capture_output=True, text=True)
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0
        results.append({
            "step": label,
            "target": target,
            "ok": ok,
            "returncode": proc.returncode,
            "seconds": round(dt, 2),
        })
        print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({dt:.2f}s)")
        if not ok:
            print("        ---- stdout (tail) ----")
            for line in proc.stdout.splitlines()[-15:]:
                print(f"        {line}")
            print("        ---- stderr (tail) ----")
            for line in proc.stderr.splitlines()[-15:]:
                print(f"        {line}")
    return results


def main():
    print("Cross-layer smoke test")
    print("=" * 56)
    results = run_steps()

    n = len(results)
    passed = sum(1 for r in results if r["ok"])
    total_s = round(sum(r["seconds"] for r in results), 2)

    print("-" * 56)
    print(f"{passed}/{n} steps passed in {total_s}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"passed": passed, "total": n, "seconds": total_s, "steps": results}, indent=2))
    print(f"wrote {OUT}")

    sys.exit(0 if passed == n else 1)


if __name__ == "__main__":
    main()
