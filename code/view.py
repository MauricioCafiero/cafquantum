#!/usr/bin/env python
"""Structure viewer: write a self-contained 3Dmol.js HTML page and open it.

Works with the *system* python (stdlib only) -- no venv needed.

Usage:
    python3 code/view.py out/CCO            # run dir: input + opt_<method> as tabs
    python3 code/view.py mol.xyz            # single file
    python3 code/view.py mol.sdf -o page.html
    python3 code/view.py out/CCO --no-open  # just write structure.html

3Dmol.js loads from a CDN, so the page needs internet the first time it opens.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js"

_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<script src="{cdn}"></script>
<style>
  body {{ font-family: -apple-system, Helvetica, sans-serif; margin: 0; }}
  #tabs {{ position: fixed; top: 0; left: 0; right: 0; z-index: 1; padding: 8px 12px;
          background: rgba(242,242,242,.95); border-bottom: 1px solid #ddd; }}
  #tabs button {{ border: 1px solid #ccc; background: #fff; padding: 4px 12px;
                 margin-right: 4px; border-radius: 4px; cursor: pointer; font-size: 13px; }}
  #tabs button.active {{ background: #3b6fa0; color: #fff; }}
  .viewer {{ position: fixed; top: 42px; bottom: 24px; left: 0; right: 0; display: none; }}
  .viewer.active {{ display: block; }}
  #banner {{ position: fixed; bottom: 0; left: 0; font-size: 12px; color: #777; }}
</style></head>
<body>
<div id="tabs"></div>
<div id="banner">{banner}</div>
<script>
const MODELS = {models};  // each: name, data, fmt
let cur = -1;
const viewers = {{}};

function render(i) {{
  if (viewers[i]) return;
  viewers[i] = $3Dmol.createViewer(document.getElementById('v' + i), {{ backgroundColor: 'white' }});
  viewers[i].addModel(MODELS[i].data, MODELS[i].fmt);
  viewers[i].setStyle({{}}, {{ stick: {{ radius: 0.18 }}, sphere: {{ scale: 0.28 }} }});
  viewers[i].zoomTo();
  viewers[i].render();
}}

function select(i) {{
  i = (i + MODELS.length) % MODELS.length;
  cur = i;
  document.querySelectorAll('.viewer').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('#tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('v' + i).classList.add('active');
  document.getElementById('t' + i).classList.add('active');
  render(i);
}}

const tabs = document.getElementById('tabs');
MODELS.forEach((m, i) => {{
  const d = document.createElement('div');
  d.className = 'viewer' + (i === 0 ? ' active' : '');
  d.id = 'v' + i;
  document.body.appendChild(d);
  const b = document.createElement('button');
  b.id = 't' + i;
  b.textContent = m.name;
  if (i === 0) b.classList.add('active');
  b.onclick = () => select(i);
  tabs.appendChild(b);
}});
if (MODELS.length) select(0);
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') select(cur + 1);
  if (e.key === 'ArrowLeft') select(cur - 1);
}});
</script></body></html>
"""


def collect(path: Path) -> list[dict]:
    """[(name, data, fmt)] for one structure file or a whole run directory."""
    models: list[dict] = []
    if path.is_dir():
        for xyz in sorted(path.glob("input_*.xyz")) + sorted(path.glob("opt_*.xyz")):
            models.append({
                "name": ("INPUT" if xyz.name.startswith("input_")
                         else xyz.stem.replace("opt_", "opt ")),
                "data": xyz.read_text(),
                "fmt": "xyz",
            })
    elif path.is_file():
        fmt = path.suffix.lower().lstrip(".")
        if fmt not in ("xyz", "sdf", "mol", "mol2"):
            sys.exit(f"unsupported format: {path.suffix} (use xyz/sdf/mol/mol2 or a run dir)")
        models.append({"name": path.stem, "data": path.read_text(), "fmt": fmt})
    else:
        sys.exit(f"no such file or run directory: {path}")
    if not models:
        sys.exit(f"no structures found in {path} (looked for input_*.xyz / opt_*.xyz)")
    return models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="structure file (.xyz/.sdf) or cafquant run dir (out/<label>)")
    p.add_argument("-o", "--output", default=None,
                   help="html output path (default: <dir>/structure.html)")
    p.add_argument("--no-open", action="store_true", help="write html, do not open browser")
    args = p.parse_args(argv)

    path = Path(args.path)
    models = collect(path)
    html_path = Path(args.output) if args.output else (
        path / "structure.html" if path.is_dir() else path.with_suffix(".html"))
    html_path.write_text(_TEMPLATE.format(
        title=path.name, cdn=CDN, models=json.dumps(models), banner=path.name))
    print(f"wrote {html_path} ({len(models)} structure"
          f"{'' if len(models) == 1 else 's'}: {', '.join(m['name'] for m in models)})")
    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())