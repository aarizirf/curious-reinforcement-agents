"""Emit the two post figures as standalone theme-aware SVGs."""
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

STYLE = """
  <style>
    .surface { fill: #fcfcfb; }
    .ink     { fill: #0b0b0b; }
    .ink2    { fill: #52514e; }
    .rule    { stroke: #d9d8d4; }
    .cell    { fill: #eceae5; }
    .cellhot { fill: #52514e; }
    .bit0    { fill: #e4e2dc; }
    .bit1    { fill: #52514e; }
    .s1      { fill: #2a78d6; }
    .s2      { fill: #eb6834; }
    .s1s     { stroke: #2a78d6; }
    .s2s     { stroke: #eb6834; }
    .arrow   { stroke: #8a8985; fill: none; }
    text { font-family: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, sans-serif; }
    @media (prefers-color-scheme: dark) {
      .surface { fill: #1a1a19; }
      .ink     { fill: #ffffff; }
      .ink2    { fill: #c3c2b7; }
      .rule    { stroke: #3a3a37; }
      .cell    { fill: #2e2e2b; }
      .cellhot { fill: #c3c2b7; }
      .bit0    { fill: #2e2e2b; }
      .bit1    { fill: #c3c2b7; }
      .s1      { fill: #3987e5; }
      .s2      { fill: #d95926; }
      .s1s     { stroke: #3987e5; }
      .s2s     { stroke: #d95926; }
      .arrow   { stroke: #7a7975; }
    }
  </style>
"""

# ---------------------------------------------------------------- figure 1

rnd = random.Random(7)
BITS_A = [rnd.randint(0, 1) for _ in range(32)]
BITS_B = [rnd.randint(0, 1) for _ in range(32)]

CW, CH, BW, BH = 9, 15, 4, 15  # position cell, bit cell


def obs_glyph(x, y, pos, bits):
    p = []
    for i in range(8):
        cls = "cellhot" if i == pos else "cell"
        p.append(f'<rect class="{cls}" x="{x + i * (CW + 1)}" y="{y}" '
                 f'width="{CW}" height="{CH}" rx="1.5"/>')
    bx = x + 8 * (CW + 1) + 9
    for i, b in enumerate(bits):
        p.append(f'<rect class="{"bit1" if b else "bit0"}" x="{bx + i * (BW + 1)}" '
                 f'y="{y}" width="{BW}" height="{BH}" rx="1"/>')
    return "\n    ".join(p)


OBS_W = 8 * (CW + 1) + 9 + 32 * (BW + 1)  # ~249
ROWS = [
    ("FROZEN", "left",  BITS_A, BITS_A, "identical", False),
    ("FROZEN", "press", BITS_A, BITS_A, "identical", False),
    ("SLOT",   "left",  BITS_A, BITS_A, "identical", False),
    ("SLOT",   "press", BITS_A, BITS_B, "TV differs", True),
]

W1, H1 = 800, 268
parts = [f'<rect class="surface" x="0" y="0" width="{W1}" height="{H1}"/>']
parts.append(f'<text class="ink2" x="24" y="26" font-size="12" font-weight="600" '
             f'letter-spacing="0.06em">AT THE LEFT WALL, p=0</text>')

x0, y0 = 146, 52
for i, (cond, act, ba, bb, verdict, differs) in enumerate(ROWS):
    y = y0 + i * 52
    if i in (0, 2):
        parts.append(f'<text class="ink" x="24" y="{y + 12}" font-size="13" '
                     f'font-weight="600">{cond}</text>')
    if i == 2:
        parts.append(f'<line class="rule" x1="24" y1="{y - 16}" x2="{W1 - 24}" '
                     f'y2="{y - 16}" stroke-width="1"/>')
    parts.append(f'<text class="ink2" x="132" y="{y + 12}" font-size="12" '
                 f'text-anchor="end">{act}</text>')
    parts.append(obs_glyph(x0, y, 0, ba))
    ax = x0 + OBS_W + 14
    parts.append(f'<path class="arrow" d="M{ax} {y + 7.5} h22" stroke-width="1.5"/>')
    parts.append(f'<path class="arrow" d="M{ax + 17} {y + 3.5} l4 4 l-4 4" stroke-width="1.5"/>')
    parts.append(obs_glyph(ax + 36, y, 0, bb))
    vx = ax + 36 + OBS_W + 16
    cls = "s2" if differs else "ink2"
    parts.append(f'<text class="{cls}" x="{vx}" y="{y + 12}" font-size="12" '
                 f'font-weight="{"600" if differs else "400"}">{verdict}</text>')

parts.append(f'<text class="ink2" x="24" y="{H1 - 16}" font-size="11.5">'
             f'one_hot(position, 8) ++ 32 TV bits. Only the last row carries any '
             f'signal about which action was taken.</text>')

svg1 = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W1} {H1}" '
        f'width="{W1}" height="{H1}" role="img" '
        f'aria-label="Observation pairs at the left wall. In FROZEN both left and '
        f'press leave the observation identical. In SLOT, press changes the TV bits.">'
        f'{STYLE}\n  ' + "\n  ".join(parts) + "\n</svg>\n")
open(f"{OUT}/ambiguity.svg", "w").write(svg1)

# ---------------------------------------------------------------- figure 2

GROUPS = [
    ("Tell press from wall-move", "inv_wall", 0.5070, 0.9604),
    ("Detect the TV resampling", "change", None, 0.9916),
    ("Read the TV bits — MLP", "mlp", 0.5377, 0.5489),
    ("Read the TV bits — linear", "lin", 0.5020, 0.5233),
]

W2, H2 = 760, 340
LX, PX, PW = 24, 268, 400  # label x, plot x, plot width
BARH, PAIRGAP, GROUPGAP = 17, 3, 26
TOP = 78


def bar(x, y, w, h, cls):
    r = min(4, w)
    return (f'<path class="{cls}" d="M{x} {y} h{w - r} a{r} {r} 0 0 1 {r} {r} '
            f'v{h - 2 * r} a{r} {r} 0 0 1 {-r} {r} h{-(w - r)} z"/>')


p = [f'<rect class="surface" x="0" y="0" width="{W2}" height="{H2}"/>']
p.append(f'<text class="ink" x="{LX}" y="30" font-size="14.5" font-weight="600">'
         f'What a 2-dimensional inverse-dynamics encoder keeps</text>')
p.append(f'<text class="ink2" x="{LX}" y="50" font-size="12">'
         f'Test accuracy on held-out states, mean of 3 seeds. EMB_DIM = 2.</text>')

# legend
p.append(f'<rect class="s1" x="430" y="41" width="10" height="10" rx="2"/>')
p.append(f'<text class="ink2" x="446" y="50" font-size="12">FROZEN</text>')
p.append(f'<rect class="s2" x="516" y="41" width="10" height="10" rx="2"/>')
p.append(f'<text class="ink2" x="532" y="50" font-size="12">SLOT</text>')

block = BARH * 2 + PAIRGAP + GROUPGAP
plot_h = len(GROUPS) * block - GROUPGAP

# chance + ceiling guides
for frac, lab in ((0.5, "chance"), (1.0, "ceiling")):
    gx = PX + PW * frac
    p.append(f'<line class="rule" x1="{gx}" y1="{TOP - 10}" x2="{gx}" '
             f'y2="{TOP + plot_h + 4}" stroke-width="1" stroke-dasharray="3 3"/>')
    p.append(f'<text class="ink2" x="{gx}" y="{TOP - 16}" font-size="11" '
             f'text-anchor="middle">{lab}</text>')

for i, (label, col, frozen, slot) in enumerate(GROUPS):
    gy = TOP + i * block
    p.append(f'<text class="ink" x="{PX - 16}" y="{gy + 15}" font-size="12.5" '
             f'text-anchor="end">{label}</text>')
    p.append(f'<text class="ink2" x="{PX - 16}" y="{gy + 32}" font-size="11" '
             f'text-anchor="end" font-family="ui-monospace, monospace">{col}</text>')
    for j, (val, cls) in enumerate(((frozen, "s1"), (slot, "s2"))):
        y = gy + j * (BARH + PAIRGAP)
        if val is None:
            p.append(f'<text class="ink2" x="{PX + 2}" y="{y + 13}" font-size="11" '
                     f'font-style="italic">not defined in FROZEN</text>')
            continue
        p.append(bar(PX, y, PW * val, BARH, cls))
        p.append(f'<text class="ink" x="{PX + PW * val + 8}" y="{y + 13}" '
                 f'font-size="11.5" font-family="ui-monospace, monospace">{val:.3f}</text>')

p.append(f'<line class="rule" x1="{PX}" y1="{TOP - 10}" x2="{PX}" '
         f'y2="{TOP + plot_h + 4}" stroke-width="1"/>')
p.append(f'<text class="ink2" x="{LX}" y="{H2 - 14}" font-size="11.5">'
         f'The encoder answers "did the TV change" almost perfectly while barely '
         f'beating chance on what the TV said.</text>')

svg2 = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}" '
        f'width="{W2}" height="{H2}" role="img" '
        f'aria-label="Bar chart. SLOT reaches 0.96 telling press from wall-move and '
        f'0.99 detecting a TV resample, but only 0.55 reading the TV bits.">'
        f'{STYLE}\n  ' + "\n  ".join(p) + "\n</svg>\n")
open(f"{OUT}/detection-vs-content.svg", "w").write(svg2)

print("wrote", OUT)
