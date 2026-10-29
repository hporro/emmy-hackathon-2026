#!/usr/bin/env python3
"""Run compile.py through submit.py on every open problem and print a summary.

    python run_all.py              # all problems
    python run_all.py p r          # only problem ids starting with p or r
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORIES = {"p": "pointwise", "r": "reductions", "m": "matmul", "f": "fused", "a": "attention", "e": "extra"}

prefixes = tuple(sys.argv[1:])
problems = sorted(p.name for p in (ROOT / "problems").iterdir() if (p / "meta.json").exists())
problems_extra = sorted(p.name for p in (ROOT / "problems_extra").iterdir() if (p / "meta.json").exists())
if prefixes:
    problems = [p for p in problems if p.startswith(prefixes)]
    problems_extra = [p for p in problems_extra if p.startswith(prefixes)]

results = []
for pid in problems:
    r = subprocess.run(
        [sys.executable, str(ROOT / "submit.py"), f"{ROOT}/problems/{pid}", "--compiler", str(ROOT / "compile.py")],
        capture_output=True, text=True,
    )
    out = (r.stdout + r.stderr).strip().splitlines()
    ok = r.returncode == 0
    detail = next((l for l in out if "PASS" in l), "") if ok else (out[0] if out else "no output")
    results.append((pid, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {pid:<18} {detail[:90]}")
for pid in problems_extra:
    r = subprocess.run(
        [sys.executable, str(ROOT / "submit.py"), f"{ROOT}/problems_extra/{pid}", "--compiler", str(ROOT / "compile.py")],
        capture_output=True, text=True,
    )
    out = (r.stdout + r.stderr).strip().splitlines()
    ok = r.returncode == 0
    detail = next((l for l in out if "PASS" in l), "") if ok else (out[0] if out else "no output")
    results.append((pid, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {pid:<18} {detail[:90]}")
    if not ok: 
        print(f"python problems_extra/{pid} --compiler compile.py")
    
print()
for letter, cat in CATEGORIES.items():
    rs = [ok for pid, ok in results if pid.startswith(letter)]
    if rs:
        print(f"{cat:<11} {sum(rs)}/{len(rs)}")
sys.exit(0 if all(ok for _, ok in results) else 1)