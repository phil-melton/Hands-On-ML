"""Why exec() is not a verifier: grade the reference and nine plausible mistakes.

    cd opencv-specialist && python -m cvbench.demo_exec_vs_mms

Every candidate goes through four independent gates (see cvbench/evaluate.py). Read
the exec-only column first: it is all the blueprint's exec() filter would have seen.
"""
from __future__ import annotations

import argparse

from cvbench.evaluate import evaluate_solution
from cvbench.families import contour_area as family

GATES = ("exec-only", "lint", "manufactured", "metamorphic")


def run(n_scenes: int = 20, n_meta: int = 5) -> dict[str, dict]:
    candidates = {"reference": ("The correct solution.", family.REFERENCE), **family.MUTANTS}
    return {name: evaluate_solution(family, code, n_scenes=n_scenes, n_meta=n_meta)
            for name, (_, code) in candidates.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--meta", type=int, default=5)
    args = ap.parse_args()
    results = run(args.scenes, args.meta)

    width = max(map(len, results))
    print(f"\nfamily: {family.SPEC.family}  |  {args.scenes} manufactured scenes, "
          f"{args.meta} scenes x 4 symmetries\n")
    print(f"{'candidate':<{width}}  " + "  ".join(f"{g:<12}" for g in GATES))
    for name, gates in results.items():
        print(f"{name:<{width}}  " + "  ".join(f"{str(gates[g]):<12}" for g in GATES))

    wrong = [n for n in results if n not in ("reference", "rgb2gray (harmless here)")]
    passed_exec = [n for n in wrong if results[n]["exec-only"].passed]
    print(f"\nexec-only accepted {len(passed_exec)} of {len(wrong)} wrong solutions.")
    caught = [n for n in wrong if not all(results[n][g].passed for g in GATES[1:])]
    print(f"lint + manufactured + metamorphic rejected {len(caught)} of {len(wrong)}.\n")
    print("First failure per gate:")
    for name, gates in results.items():
        fails = [f"  {g}: {gates[g].detail}" for g in GATES if not gates[g].passed]
        if fails:
            print(f"- {name}")
            print("\n".join(fails))


if __name__ == "__main__":
    main()
