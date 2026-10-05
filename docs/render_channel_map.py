#!/usr/bin/env python3
"""Render docs/architecture-spec.json into docs/capability-map.html.

Dumb templater: the spec carries the semantics AND the geometry (node x/y/w/h,
channel path `d`, label positions). This script only emits SVG from them, so the
HTML is a literal projection of the spec, no re-derived topology. Edit the spec,
re-run this, the map updates. That is the anti-drift contract.

    python3 docs/render_channel_map.py
"""
import json, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(HERE, "architecture-spec.json")))

CLS = spec["classes"]
ARROW = {"g": "a-g", "s": "a-s", "c": "a-c"}


def esc(s): return html.escape(str(s))


def node_svg(n):
    c = CLS[n["cls"]]
    dash = ' stroke-dasharray="7 5"' if c.get("dashed") else ""
    sw = c.get("sw", 2.5)
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    cx = x + w / 2
    parts = [f'<g class="node" id="{n["id"]}">',
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="{sw}"{dash}/>']
    ty = y + 26
    parts.append(f'<text class="nt" x="{cx}" y="{ty}" text-anchor="middle">{esc(n["t"])}</text>')
    ly = ty + 19
    for line in n.get("lines", []):
        parts.append(f'<text class="ns" x="{cx}" y="{ly}" text-anchor="middle">{esc(line)}</text>')
        ly += 17
    parts.append('</g>')
    return "\n".join(parts)


def edge_svg(ch):
    kind = ch.get("kind", "soft")
    cls = {"gold": "edge gold", "soft": "edge soft", "voice": "edge voice"}.get(kind, "edge")
    mk = ARROW.get(ch.get("arrow", "s"), "a-s")
    return (f'<path class="{cls}" data-n="{ch["dn"]}" d="{ch["d"]}" '
            f'marker-end="url(#{mk})"/>')


def elabel_svg(ch):
    if not ch.get("label"):
        return ""
    lc = {"gold": "elabel gold", "voice": "elabel voice"}.get(ch.get("kind"), "elabel")
    anc = ch.get("anchor", "start")
    rot = f' transform="rotate(-90 {ch["lx"]} {ch["ly"]})"' if ch.get("rot") else ""
    return f'<text class="{lc}" x="{ch["lx"]}" y="{ch["ly"]}" text-anchor="{anc}"{rot}>{esc(ch["label"])}</text>'


def panels_svg(gates, x, y, w):
    """Stacked closed-set panels bottom-left."""
    out = []
    cy = y
    for g in gates:
        h = 34 + 18 * len(g["lines"])
        out.append(f'<rect class="panel" x="{x}" y="{cy}" width="{w}" height="{h}" rx="8"/>')
        out.append(f'<text class="pt" x="{x+16}" y="{cy+24}">{esc(g["t"])}</text>')
        ly = cy + 44
        for line in g["lines"]:
            out.append(f'<text class="pl" x="{x+16}" y="{ly}">{esc(line)}</text>')
            ly += 18
        cy += h + 12
    return "\n".join(out)


def legend_svg(legend, x, y, w):
    out = [f'<rect class="panel" x="{x}" y="{y}" width="{w}" height="{34+len(legend)*22+18}" rx="8"/>',
           f'<text class="pt" x="{x+16}" y="{y+24}">Legend</text>']
    ly = y + 42
    for item in legend:
        c = CLS[item["cls"]]
        dash = ' stroke-dasharray="4 3"' if c.get("dashed") else ""
        out.append(f'<rect x="{x+16}" y="{ly-11}" width="16" height="12" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.5"{dash}/>')
        out.append(f'<text class="pl" x="{x+40}" y="{ly}">{esc(item["t"])}</text>')
        ly += 22
    out.append(f'<text class="pl" x="{x+16}" y="{ly+2}" style="font-style:italic;fill:var(--slate)">dashed = you fetch / optional · hover a box to light its channels</text>')
    return "\n".join(out)


ui = spec["uiLayer"]
ob = spec["operatorBand"]
edges = "\n".join(edge_svg(c) for c in spec["channels"])
elabels = "\n".join(elabel_svg(c) for c in spec["channels"])
nodes = "\n".join(node_svg(n) for n in spec["nodes"])
gates = panels_svg(spec["gates"], 50, 760, 500)
tc = panels_svg(spec["toolCommands"], 580, 760, 370)
legend = legend_svg(spec["legend"], 970, 760, 450)
m = spec["meta"]

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(m['title'])} — capability map</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root{{
    --paper:#F5F5F3;--container:#F8F8F6;--carbon:#2D3033;--slate:#8390A6;
    --blue-f:#C5D5FF;--blue-s:#3B5BDB;--green-f:#B8E0C8;--green-s:#2D5A3D;
    --amber-f:#FFE4A3;--amber-s:#B8920F;--coral-f:#FFCAB0;--coral-s:#C45A10;
    --teal-f:#A8DDE0;--teal-s:#0A7B83;--steel-f:#C8CED8;--steel-s:#5A6577;
    --serif:'Instrument Serif',Georgia,serif;--sans:'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif;
    --mono:'JetBrains Mono',Consolas,monospace;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--paper);color:var(--carbon);font-family:var(--sans)}}
  header{{max-width:1480px;margin:0 auto;padding:32px 24px 6px}}
  h1{{font-family:var(--serif);font-weight:400;font-size:clamp(28px,3.4vw,40px)}}
  .sub{{color:var(--slate);font-size:14px;max-width:940px;margin-top:6px}}
  .stamp{{font-family:var(--mono);font-size:11px;color:var(--slate);margin-top:8px}}
  #wrap{{max-width:1480px;margin:0 auto;padding:6px 12px 30px}}
  svg{{width:100%;height:auto;display:block}}
  .uilayer{{fill:none;stroke:var(--teal-s);stroke-width:1.4;stroke-dasharray:2 4;opacity:.8}}
  .uilab{{font-family:var(--mono);font-size:12px;fill:var(--teal-s);letter-spacing:.08em}}
  .opband{{fill:var(--steel-f);stroke:var(--steel-s);stroke-width:1}}
  .oblab{{font-family:var(--mono);font-size:12px;fill:var(--carbon);letter-spacing:.05em;font-weight:600}}
  .node rect{{cursor:pointer}} .node text{{pointer-events:none}}
  .nt{{font-family:var(--sans);font-weight:700;font-size:15px;fill:var(--carbon)}}
  .ns{{font-family:var(--sans);font-size:11.5px;fill:var(--carbon);opacity:.78}}
  .edge{{fill:none;stroke:var(--slate);stroke-width:1.8}}
  .edge.soft{{stroke-dasharray:6 5;opacity:.72}}
  .edge.gold{{stroke:var(--amber-s);stroke-width:2.4}}
  .edge.voice{{stroke:var(--coral-s);stroke-width:2}}
  .elabel{{font-family:var(--mono);font-size:11px;fill:var(--slate)}}
  .elabel.gold{{fill:var(--amber-s);font-weight:600}} .elabel.voice{{fill:var(--coral-s)}}
  .panel{{fill:var(--container);stroke:var(--slate);stroke-width:1}}
  .pt{{font-family:var(--sans);font-weight:700;font-size:13px;fill:var(--carbon)}}
  .pl{{font-family:var(--mono);font-size:11px;fill:var(--carbon)}}
  .dim{{opacity:.12;transition:opacity .18s}} .lit rect{{stroke-width:4}}
  figcaption{{max-width:1480px;margin:0 auto;padding:4px 24px 40px;font-size:12.5px;font-style:italic;color:var(--slate)}}
</style>
</head>
<body>
<header>
  <h1>{esc(m['title'])}</h1>
  <div class="sub">{esc(m['subtitle'])}</div>
  <div class="stamp">{esc(m['stamp'])}</div>
</header>
<div id="wrap">
<svg viewBox="{m['viewBox']}" xmlns="http://www.w3.org/2000/svg" id="map">
  <defs>
    <marker id="a-g" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto"><polygon points="0 0,9 3.2,0 6.4" fill="#B8920F"/></marker>
    <marker id="a-s" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto"><polygon points="0 0,9 3.2,0 6.4" fill="#8390A6"/></marker>
    <marker id="a-c" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto"><polygon points="0 0,9 3.2,0 6.4" fill="#C45A10"/></marker>
  </defs>
  <rect class="opband" x="{ob['x']}" y="{ob['y']}" width="{ob['w']}" height="{ob['h']}" rx="8"/>
  <text class="oblab" x="{ob['x']+16}" y="{ob['y']+20}">{esc(ob['label'])}</text>
  <rect class="uilayer" x="{ui['x']}" y="{ui['y']}" width="{ui['w']}" height="{ui['h']}" rx="12"/>
  <text class="uilab" x="{ui['x']+16}" y="{ui['y']+24}">{esc(ui['label'])}</text>
  <g id="edges">
{edges}
  </g>
{elabels}
{nodes}
{gates}
{tc}
{legend}
</svg>
</div>
<figcaption>Figure: the voice-driven SDA harness, projected from docs/architecture-spec.json. One UI layer of four surfaces feeds one reentrant router; inputs unify to text; gate 1 splits tool-command from the embed-RAG-LLM path; gate 2 routes the LLM output to user, tool, artifact-build, or the operator; loop 3 returns a tool call's output to the router. Hover any box to isolate its channels.</figcaption>
<script>
const nodes=[...document.querySelectorAll('.node')];
const edges=[...document.querySelectorAll('.edge')];
function reset(){{nodes.forEach(n=>n.classList.remove('dim','lit'));edges.forEach(e=>e.classList.remove('dim'));}}
nodes.forEach(n=>{{
  n.addEventListener('mouseenter',()=>{{
    const id=n.id,nbr=new Set([id]);
    edges.forEach(e=>{{const m=(e.dataset.n||'').split(',');if(m.includes(id))m.forEach(x=>nbr.add(x));}});
    nodes.forEach(o=>o.classList.toggle('dim',!nbr.has(o.id)));
    n.classList.add('lit');
    edges.forEach(e=>e.classList.toggle('dim',!(e.dataset.n||'').split(',').includes(id)));
  }});
  n.addEventListener('mouseleave',reset);
}});
</script>
</body>
</html>
"""

open(os.path.join(HERE, "capability-map.html"), "w").write(HTML)
print("wrote", os.path.join(HERE, "capability-map.html"))
