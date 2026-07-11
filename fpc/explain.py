"""
OPTIONAL plain-English explanation layer (uses the Claude API).
===============================================================

This is the *clearly separate, optional* part the project brief anticipated: the
offline core produces numbers and structured findings; this layer turns them
into a readable, plain-language interpretation you could drop into a paper's
discussion or hand to a collaborator. It is the ONLY module that calls an
external API, and nothing in the core imports it -- the offline guarantee of
`fpc/` is untouched.

Requirements
------------
* ``pip install anthropic``  (the official Anthropic SDK)
* An API key on the environment: set ``ANTHROPIC_API_KEY`` (the SDK reads it
  automatically). Never hardcode a key in source.

Model
-----
Defaults to **Claude Fable 5** (``claude-fable-5``), Anthropic's most capable
model, with an automatic server-side fallback to ``claude-opus-4-8`` if a
request is declined. Pass ``model="claude-opus-4-8"`` to use Opus directly.
(Note: Fable 5 requires an org data-retention setting of 30 days; if requests
fail with a 400, switch ``model`` to ``claude-opus-4-8``.)

Example
-------
    from fpc import DNA, tasks, datasets
    from fpc.explain import explain_detection

    host = datasets.markov_sequence(6000, DNA, seed=2)
    donor = datasets.markov_sequence(400, DNA, seed=9)
    chimera, span = datasets.insert_foreign(host, donor, position=3000)
    res = tasks.detect_against_reference(host, chimera, DNA)
    print(explain_detection(res, sequence_name="chimera"))
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You are a careful bioinformatics research assistant. You are given the "
    "numerical output of the Fractal Predictive Cortex (FPC) -- a training-free, "
    "alignment-free sequence model that scores how 'surprised' it is (in bits) at "
    "each position, and derives prediction, anomaly detection, classification, and "
    "phylogeny from that surprise signal. Explain the results in clear, plain "
    "language for a researcher who is not an expert in this method. Be accurate and "
    "honest: state what the numbers support and what they do not, note limitations "
    "and alternative explanations, and never overclaim. Use short paragraphs. Do not "
    "invent numbers that are not in the provided data."
)


def _client(api_key: str | None = None):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The explain layer needs the Anthropic SDK. Install it with:\n"
            "    pip install anthropic\n"
            "and set your key:  setx ANTHROPIC_API_KEY \"sk-ant-...\"  (then reopen the shell)"
        ) from exc
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def explain(summary: dict, model: str = "claude-fable-5", effort: str = "medium",
            max_tokens: int = 3000, api_key: str | None = None) -> str:
    """Turn a structured FPC ``summary`` dict into a plain-English interpretation.

    Streams the response (robust to long outputs) and returns the text. If the
    model declines the request, returns a short placeholder rather than raising.
    """
    client = _client(api_key)
    user = (
        "Here are results produced by the Fractal Predictive Cortex. Interpret them "
        "in plain language for a research write-up:\n\n"
        + json.dumps(summary, indent=2)
    )
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": user}],
    )

    is_fable = model.startswith(("claude-fable", "claude-mythos"))
    if is_fable:
        # Opt into a server-side fallback so a rare policy decline is re-served
        # by Opus inside the same call (recommended default for Fable 5).
        with client.beta.messages.stream(
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            **kwargs,
        ) as stream:
            msg = stream.get_final_message()
    else:
        with client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()

    if getattr(msg, "stop_reason", None) == "refusal":
        return "[The model declined to generate an explanation for this input.]"
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


# --------------------------------------------------------------------- #
#  summary builders -- turn FPC result objects into JSON-able dicts      #
# --------------------------------------------------------------------- #
def summarize_detection(result, sequence_name: str = "query", top: int = 6) -> dict:
    """Build an explanation summary from a :class:`fpc.tasks.DetectionResult`."""
    track = result.track
    anoms = sorted(result.anomalies, key=lambda a: -a.peak_z)[:top]
    return {
        "task": "anomaly / mutation detection (per-position surprise)",
        "sequence": sequence_name,
        "mean_surprise_bits_per_symbol": round(track.mean_bits, 3),
        "perplexity": round(track.perplexity, 3),
        "n_flagged_regions": len(result.anomalies),
        "top_flagged_regions": [
            {
                "start": int(a.start), "end": int(a.end),
                "peak_robust_z": round(a.peak_z, 2),
                "mean_bits": round(a.mean_bits, 2),
                "p_value": float(a.p_value),
                "significant_after_correction": bool(a.significant),
            }
            for a in anoms
        ],
        "note": "Higher surprise means the model found a region hard to predict "
                "from the rest of the sequence; robust-z and p-value quantify how "
                "unusual each flagged region is.",
    }


def summarize_classification(class_names, predictions, score_matrix=None) -> dict:
    """Build a summary from classifier output (labels + optional bits matrix)."""
    out = {
        "task": "generative classification (which model is least surprised)",
        "classes": list(class_names),
        "predictions": list(predictions),
    }
    if score_matrix is not None:
        out["cross_entropy_bits_per_query"] = [
            [round(float(v), 3) for v in row] for row in score_matrix
        ]
    return out


def explain_detection(result, sequence_name: str = "query", **kwargs) -> str:
    """Convenience: summarize a detection result and explain it in one call."""
    return explain(summarize_detection(result, sequence_name), **kwargs)
