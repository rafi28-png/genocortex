"""
Demo of the OPTIONAL plain-English explanation layer (Claude API).

Runs the offline detection first, then:
  * if ANTHROPIC_API_KEY is set  -> calls Claude to explain the findings,
  * otherwise                    -> prints the structured summary + how to enable it.

Run:  python examples/explain_demo.py
Enable the API layer:  pip install anthropic   and set ANTHROPIC_API_KEY
"""
import json
import os

from fpc import DNA, tasks, datasets
from fpc.explain import summarize_detection, explain_detection


def main():
    # --- offline: run a real detection ---
    host = datasets.markov_sequence(6000, DNA, order=2, concentration=0.15, seed=2)
    donor = datasets.markov_sequence(400, DNA, order=3, concentration=0.4, seed=999)
    chimera, span = datasets.insert_foreign(host, donor, position=3000, seed=0)
    res = tasks.detect_against_reference(host, chimera, DNA, order=8,
                                         window=40, z_threshold=3.0)

    summary = summarize_detection(res, sequence_name="chimera (host + foreign insert)")
    print("Structured findings (produced fully offline):")
    print(json.dumps(summary, indent=2))
    print(f"\n(ground-truth insert was at {span})\n")

    # --- optional: ask Claude to explain, if a key is available ---
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY found -- asking Claude (Fable 5) to explain ...\n")
        try:
            print(explain_detection(res, sequence_name="chimera (host + foreign insert)"))
        except Exception as exc:
            print(f"(explanation call failed: {exc})")
    else:
        print("No ANTHROPIC_API_KEY set -> skipping the API call.")
        print("To enable the plain-English layer:")
        print('    pip install anthropic')
        print('    setx ANTHROPIC_API_KEY "sk-ant-..."   (reopen the shell)')
        print("    python examples/explain_demo.py")


if __name__ == "__main__":
    main()
