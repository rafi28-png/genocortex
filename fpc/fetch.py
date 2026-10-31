"""
OPTIONAL real-data fetcher (the only internet-touching module).
===============================================================

The GenoCortex *core* never needs the network. This helper is a convenience for
running the engine on real published sequences, so a researcher can go from an
accession number to a learned cortex in one line. It is intentionally isolated:
nothing in ``fpc`` imports it, so the offline guarantee of the core is intact.

It uses NCBI E-utilities (``efetch``), which needs **no key** for light use. Be a
good citizen: pass your ``email`` (and an ``api_key`` if you have one) to lift the
rate limit, and do not hammer the service.

Example
-------
    from fpc import fetch, tasks, DNA
    rec = fetch.ncbi_fasta("NC_045512.2", email="you@example.com")  # SARS-CoV-2
    res = tasks.detect_anomalies(rec.seq, DNA, order=10)
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from .io_fasta import Record

_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def ncbi_search(term: str, db: str = "protein", email: str | None = None,
                api_key: str | None = None, retmax: int = 1,
                timeout: float = 30.0) -> list[str]:
    """Search NCBI (esearch) and return a list of UID strings for ``term``.

    Lets you fetch real records by name/organism instead of hardcoding
    accessions -- much more robust. Feed a returned UID straight to
    :func:`ncbi_fasta` (efetch accepts UIDs).
    """
    params = {"db": db, "term": term, "retmax": retmax, "retmode": "json"}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{_ESEARCH}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "GenoCortex/1.0.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "ignore"))
    return list(data.get("esearchresult", {}).get("idlist", []))


def ncbi_fasta(accession: str, db: str = "nucleotide", email: str | None = None,
               api_key: str | None = None, timeout: float = 30.0,
               retries: int = 3) -> Record:
    """Fetch a single FASTA record from NCBI by accession.

    Parameters
    ----------
    accession:  e.g. ``"NC_045512.2"``.
    db:         NCBI database (``"nucleotide"`` or ``"protein"``).
    email:      your contact email (NCBI etiquette; raises the rate limit).
    api_key:    optional NCBI API key for higher throughput.
    """
    params = {"db": db, "id": accession, "rettype": "fasta", "retmode": "text"}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{_EFETCH}?{urllib.parse.urlencode(params)}"

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GenoCortex/1.0.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "ignore")
            return _parse_single(text, accession)
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to fetch {accession!r} from NCBI: {last}")


def _parse_single(text: str, accession: str) -> Record:
    header = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                break
            header = line[1:].strip()
        else:
            chunks.append(line)
    if header is None:
        raise ValueError(f"no FASTA record returned for {accession!r}")
    rid = header.split()[0] if header.split() else accession
    return Record(id=rid, description=header, seq="".join(chunks).upper())
