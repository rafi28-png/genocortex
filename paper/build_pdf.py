"""
Render paper/paper.md to a publication-styled PDF.
==================================================

No pandoc / LaTeX required. Pipeline:
    Markdown  --(python-markdown)-->  styled HTML  --(headless Chromium)-->  PDF

Chromium is located from the Playwright install (already on this machine) or any
Chrome/Edge on PATH. Run:  python paper/build_pdf.py
"""
import glob
import os
import re
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper.md")
HTML = os.path.join(HERE, "paper.html")
PDF = os.path.join(HERE, "paper.pdf")

CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.5; color: #16171a;
  max-width: 100%; margin: 0;
}
h1, h2, h3, h4 { font-family: "Segoe UI", Helvetica, Arial, sans-serif; color: #0b0b0b; line-height: 1.25; }
h1 { font-size: 19pt; text-align: center; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin-top: 20pt; border-bottom: 1px solid #d9d9d6; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin-top: 14pt; color: #2a3a4a; }
p { margin: 6pt 0; text-align: justify; }
em { color: #333; }
a { color: #1c5cab; text-decoration: none; }
code {
  font-family: "Consolas", "SFMono-Regular", monospace; font-size: 9pt;
  background: #f3f4f6; padding: 1px 4px; border-radius: 3px;
}
pre {
  background: #f7f8fa; border: 1px solid #e3e5e8; border-radius: 6px;
  padding: 9pt 11pt; overflow-x: auto; line-height: 1.4;
}
pre code { background: none; padding: 0; font-size: 8.6pt; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.3pt; }
th, td { border: 1px solid #d2d4d7; padding: 4pt 7pt; text-align: left; }
th { background: #eef1f4; font-family: "Segoe UI", sans-serif; }
img { max-width: 100%; height: auto; display: block; margin: 10pt auto 2pt; border: 1px solid #e3e5e8; border-radius: 4px; }
img + em, p > img { }
blockquote { margin: 8pt 0; padding: 4pt 12pt; border-left: 3px solid #9db4cf; color: #444; background: #f6f8fb; }
hr { border: none; border-top: 1px solid #d9d9d6; margin: 16pt 0; }
figure { page-break-inside: avoid; }
h2, h3 { page-break-after: avoid; }
"""


import shutil


def find_chromium():
    # 1) System binaries on PATH (Linux / macOS / Windows)
    for exe in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "chrome", "msedge"):
        found = shutil.which(exe)
        if found:
            return found

    # 2) Playwright bundled chromium across Linux, macOS, and Windows
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, ".cache", "ms-playwright", "chromium-*", "chrome-linux*", "chrome"),
        os.path.join(home, "Library", "Caches", "ms-playwright", "chromium-*", "chrome-mac*", "Chromium.app", "Contents", "MacOS", "Chromium"),
        os.path.join(home, "AppData", "Local", "ms-playwright", "chromium-*", "chrome-win*", "chrome.exe"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return sorted(hits)[-1]

    # 3) Common macOS / Windows install paths
    for cand in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]:
        if os.path.exists(cand):
            return cand
    return None


def to_file_url(path):
    abspath = os.path.abspath(path).replace("\\", "/")
    if abspath.startswith("/"):
        return "file://" + abspath
    return "file:///" + abspath


def build_html():
    with open(MD, encoding="utf-8") as fh:
        text = fh.read()
    # Resolve image paths (../output/...) to absolute file URLs so Chromium loads them.
    def fix(m):
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "file:")):
            return m.group(0)
        abspath = os.path.normpath(os.path.join(HERE, src))
        return f"![{alt}]({to_file_url(abspath)})"
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix, text)

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Fractal Predictive Cortex</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    with open(HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    return HTML


def build_pdf():
    chrome = find_chromium()
    if not chrome:
        print("No Chromium/Chrome/Edge found; wrote paper.html — open it and 'Print to PDF'.")
        return None
    cmd = [
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", f"--print-to-pdf={PDF}", to_file_url(HTML),
    ]
    subprocess.run(cmd, check=True, timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return PDF


if __name__ == "__main__":
    build_html()
    print(f"wrote {HTML}")
    out = build_pdf()
    if out and os.path.exists(out):
        kb = os.path.getsize(out) / 1024
        print(f"wrote {out} ({kb:.0f} KB)")
    else:
        sys.exit(0)
