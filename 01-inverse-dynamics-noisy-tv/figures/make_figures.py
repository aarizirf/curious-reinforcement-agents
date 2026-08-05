"""Emit the post figures.

Styled to sit inside the write-up rather than to stand alone: the site's ink,
muted and rule colours, a mono face, no titles (the caption carries those), and
a white ground so they stay readable on a dark GitHub theme too.
"""
import os

OUT = os.path.dirname(os.path.abspath(__file__))

INK, MUTED, RULE = "#1a1a1a", "#767676", "#e2e2e2"
FILL, S1, S2 = "#efefed", "#2a78d6", "#eb6834"
MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def svg(w, h, body, label):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
        f'height="{h}" role="img" aria-label="{label}">\n'
        f'  <style>text {{ font-family: {MONO}; }}</style>\n'
        f'  <rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>\n  '
        + "\n  ".join(body)
        + "\n</svg>\n"
    )


def text(x, y, s, size=13, fill=INK, anchor="start", weight="400", style="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-style="{style}">{s}</text>')


# --- 1. what the encoder kept ------------------------------------------------

GROUPS = [
    ("Tell press from wall-move", 0.5070, 0.9604),
    ("Detect the TV reshuffling", None, 0.9916),
    ("Read the TV bits, neural", 0.5377, 0.5489),
    ("Read the TV bits, linear", 0.5020, 0.5233),
]

W2, H2 = 760, 300
PX, PW, TOP = 250, 400, 44
BARH, PAIRGAP, GROUPGAP = 15, 4, 24


def bar(x, y, w, h, fill):
    r = min(3, w)
    return (f'<path d="M{x} {y} h{w - r} a{r} {r} 0 0 1 {r} {r} v{h - 2 * r} '
            f'a{r} {r} 0 0 1 {-r} {r} h{-(w - r)} z" fill="{fill}"/>')


b = []
block = BARH * 2 + PAIRGAP + GROUPGAP
plot_h = len(GROUPS) * block - GROUPGAP

for x, lab in ((PX + PW * 0.5, "chance"), (PX + PW, "ceiling")):
    b.append(f'<line x1="{x}" y1="{TOP - 8}" x2="{x}" y2="{TOP + plot_h + 6}" '
             f'stroke="{RULE}" stroke-width="1" stroke-dasharray="3 4"/>')
    b.append(text(x, TOP - 16, lab, 12, MUTED, anchor="middle"))

for i, (label, frozen, slot) in enumerate(GROUPS):
    gy = TOP + i * block
    b.append(text(PX - 18, gy + 21, label, 13, INK, anchor="end"))
    for j, (val, fill) in enumerate(((frozen, S1), (slot, S2))):
        y = gy + j * (BARH + PAIRGAP)
        if val is None:
            continue
        b.append(bar(PX, y, PW * val, BARH, fill))
        b.append(text(PX + PW * val + 8, y + 12, f"{val:.2f}", 12, MUTED))

b.append(f'<line x1="{PX}" y1="{TOP - 8}" x2="{PX}" y2="{TOP + plot_h + 6}" '
         f'stroke="{RULE}" stroke-width="1"/>')

ly = TOP + plot_h + 32
b.append(f'<rect x="{PX}" y="{ly - 9}" width="9" height="9" fill="{S1}" rx="2"/>')
b.append(text(PX + 15, ly, "Frozen", 12, MUTED))
b.append(f'<rect x="{PX + 80}" y="{ly - 9}" width="9" height="9" fill="{S2}" rx="2"/>')
b.append(text(PX + 95, ly, "Slot", 12, MUTED))

open(f"{OUT}/detection-vs-content.svg", "w").write(svg(
    W2, H2, b,
    "The same frozen encoder answers four questions. It detects the reshuffle "
    "at 0.99 but reads the bits at 0.55."))


# --- 2. when it finds the TV -------------------------------------------------

CURVE = {
    "Frozen": [0.5025, 0.489, 0.4972, 0.5093, 0.4949, 0.5093, 0.4915, 0.4907,
               0.4977, 0.5051, 0.4907, 0.5079, 0.5085, 0.5073, 0.4907],
    "Slot": [0.5085, 0.401, 0.4943, 0.5646, 0.4139, 0.5275, 0.9384, 0.9716,
             0.9725, 0.9849, 0.9903, 0.9943, 0.9925, 0.9943, 0.9928],
}

W3, H3 = 760, 280
L, R, T, B = 104, 636, 28, 226
YLO, YHI = 0.35, 1.0


def px(i):
    return L + (R - L) * i / (len(CURVE["Slot"]) - 1)


def py(v):
    return B - (B - T) * (v - YLO) / (YHI - YLO)


c = []
for v, lab in ((0.5, "0.5  chance"), (1.0, "1.0  ceiling")):
    c.append(f'<line x1="{L}" y1="{py(v)}" x2="{R}" y2="{py(v)}" stroke="{RULE}" '
             f'stroke-width="1" stroke-dasharray="3 4"/>')
    c.append(text(L - 12, py(v) + 4, lab, 12, MUTED, anchor="end"))

c.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{RULE}" stroke-width="1"/>')
for i in (0, 4, 9, 14):
    c.append(text(px(i), B + 20, str(i + 1), 12, MUTED, anchor="middle"))
c.append(text((L + R) / 2, B + 42, "epoch", 12, MUTED, anchor="middle"))

for name, colour in (("Frozen", S1), ("Slot", S2)):
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(CURVE[name]))
    c.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
             f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    c.append(text(R + 10, py(CURVE[name][-1]) + 4, name, 12, colour, weight="700"))

open(f"{OUT}/learning.svg", "w").write(svg(
    W3, H3, c,
    "Wall-stationary accuracy after each epoch. FROZEN stays at 0.5 throughout. "
    "SLOT sits at 0.5 for six epochs, then jumps to 0.94."))

print("wrote detection-vs-content.svg, learning.svg")
