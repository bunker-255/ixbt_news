#!/usr/bin/env python3
"""
mdview — renders a Markdown file to a self-contained HTML page.

Used to preview the README in a browser, since the terminal cannot
render Markdown and no converter was installed.

Usage:
    ./mdview.py README.md            -> /tmp/mdview-<name>.html, prints path
    ./mdview.py README.md --out x.html
"""
import html
import re
import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; padding: 2.5rem 1.5rem 6rem;
  max-width: 54rem;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  color: #1f2328; background: #fff;
}
@media (prefers-color-scheme: dark) {
  body { color: #e6edf3; background: #0d1117; }
  h1, h2, h3, h4, h5, h6 { color: #f0f6fc; border-color: #30363d; }
  code { background: #161b22 !important; color: #e6edf3; }
  pre { background: #161b22 !important; border-color: #30363d; }
  th { background: #161b22; }
  tr, li { border-color: #30363d !important; }
  a { color: #4493f8; }
  blockquote { border-left-color: #3d444d; color: #9198a1; }
}

h1, h2, h3, h4, h5, h6 { line-height: 1.25; margin: 2rem 0 1rem; }
h1 { font-size: 2.1rem; margin-top: 1rem; }
h2 { font-size: 1.5rem; padding-bottom: .3rem; border-bottom: 1px solid #d1d9e0; }
h3 { font-size: 1.2rem; }
h4 { font-size: 1rem; }
p, li { margin: .6rem 0; }

a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }

code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: .87em;
  background: #f2f4f7; padding: .15em .35em; border-radius: 5px;
}
pre {
  background: #f6f8fa; padding: 1rem; border-radius: 8px;
  border: 1px solid #d1d9e0; overflow-x: auto; line-height: 1.5;
}
pre code { background: none; padding: 0; font-size: .85em; }

table { border-collapse: collapse; width: 100%; margin: 1.2rem 0; display: block; overflow-x: auto; }
th, td { border: 1px solid #d1d9e0; padding: .55rem .8rem; text-align: left; vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; }

blockquote { border-left: 4px solid #d1d9e0; margin: 1.2rem 0; padding: .3rem 1rem; color: #59636e; }
blockquote > :first-child { margin-top: 0; }
blockquote > :last-child { margin-bottom: 0; }

details { margin: 1rem 0; border: 1px solid #d1d9e0; border-radius: 8px; padding: .8rem 1rem; }
summary { cursor: pointer; font-weight: 600; }
details[open] summary { margin-bottom: .8rem; }

hr { border: 0; border-top: 1px solid #d1d9e0; margin: 2rem 0; }
img { max-width: 100%; }
ul, ol { padding-left: 1.6rem; }

/* ASCII flowcharts must not wrap — they lose their shape otherwise */
pre.diagram { overflow-x: auto; }
"""


def add_code_copy_script(html_text: str) -> str:
    return html_text


def main() -> int:
    argv = sys.argv[1:]
    out = None
    src_arg = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            i += 1
            out = Path(argv[i]) if i < len(argv) else None
        elif a.startswith("--out="):
            out = Path(a.split("=", 1)[1])
        elif not a.startswith("-"):
            src_arg = a
        i += 1

    if not src_arg:
        print(__doc__)
        return 2

    src = Path(src_arg)
    if not src.is_file():
        print(f"no such file: {src}", file=sys.stderr)
        return 1

    text = src.read_text(encoding="utf-8")

    md = markdown.Markdown(
        extensions=["extra", "tables", "fenced_code", "toc", "sane_lists", "admonition"],
        output_format="html5",
    )
    body = md.convert(text)

    # <div align="center"> from GitHub-style headers: flatten to centered block
    body = re.sub(
        r'<div align="center">(.*?)</div>',
        r'<div class="centered">\1</div>',
        body,
        flags=re.S,
    )

    # ASCII flowcharts: pre blocks with box-drawing chars get their own class
    body = re.sub(
        r"<pre><code>(.*?)</code></pre>",
        lambda m: (
            '<pre class="diagram"><code>' + m.group(1) + "</code></pre>"
            if any(ch in m.group(1) for ch in "┌└├▼►│─")
            else "<pre><code>" + m.group(1) + "</code></pre>"
        ),
        body,
        flags=re.S,
    )

    centered_css = """
.centered { text-align: center; margin: 2rem 0; }
.centered h1 { margin-top: 0; }
.centered img { display: inline-block; }
"""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(src.name)} — {html.escape(src.stem)}</title>
<style>{CSS}{centered_css}</style>
</head>
<body>
{body}
</body>
</html>
"""

    if out is None:
        out = Path("/tmp") / f"mdview-{src.stem}.html"

    out.write_text(page, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())