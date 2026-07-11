"""
Reproduce the entire project end to end.

    python run_all.py            # tests + synthetic benchmarks + visuals (offline)
    python run_all.py --real     # also run the real-genome studies (needs internet
                                 # on first run; cached offline afterwards)

Each step is timed and its exit status reported. Real-data steps are skipped by
default so a fresh checkout runs fully offline.
"""
import subprocess
import sys
import time

CORE = [
    ("Unit tests", [sys.executable, "-m", "pytest", "tests/", "-q"]),
    ("Synthetic benchmarks + figures 1-5", [sys.executable, "benchmarks/bench.py"]),
    ("Phylogeny statistics (25 trees, CIs) + fig 9", [sys.executable, "benchmarks/phylo_stats.py"]),
    ("Interactive 3-D artifacts", [sys.executable, "examples/make_visuals.py"]),
    ("Quickstart (all four abilities)", [sys.executable, "examples/quickstart.py"]),
]

REAL = [
    ("7 coronaviruses", [sys.executable, "examples/real_genomes.py"]),
    ("28-genome, 5-family study + figs 7,8", [sys.executable, "examples/big_study.py"]),
    ("40 real proteins + fig 10", [sys.executable, "examples/protein_study.py"]),
    ("Robinson-Foulds on mitogenomes + fig 11", [sys.executable, "benchmarks/rf_benchmark.py"]),
]


def run(steps):
    ok = True
    for name, cmd in steps:
        print(f"\n{'='*66}\n>>> {name}\n{'='*66}")
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"--- {name}: {status} in {dt:.1f}s")
        ok = ok and rc == 0
    return ok


if __name__ == "__main__":
    steps = list(CORE)
    if "--real" in sys.argv:
        steps += REAL
    ok = run(steps)
    print(f"\n{'='*66}\nALL STEPS {'PASSED' if ok else 'HAD FAILURES'}.")
    sys.exit(0 if ok else 1)
