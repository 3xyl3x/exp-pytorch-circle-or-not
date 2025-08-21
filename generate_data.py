# generate_data.py
from PIL import Image, ImageDraw, ImageOps
import os, random, shutil, math

# ===== tweakable variables =====
OUT_SIZE = 32          # final saved size
CANVAS   = 96          # big draw area; we crop+resize after
MARGIN_FRAC = 0.12     # padding around cropped content
THICKNESS_RANGE = (1, 3)

# "Circle-ish" (positives) allow mild oval and tiny gaps:
CIRCLE_STRETCH = (0.90, 1.10)  # rx/ry within ±10%
MAX_SMALL_GAP_DEG = 18         # <= 5% of circumference

# Negatives
STRONG_OVAL_MIN = 1.5          # strong stretch for NOT circle
OPEN_GAP_DEG_RANGE = (30, 160) # > 5% open gap for NOT circle

# Mix of positive/negative shape probabilities
P_POS_WOBBLY_CIRCLE = 0.70     # rest split between mild oval and tiny-gap arc
P_POS_MILD_OVAL     = 0.20
P_POS_TINY_GAP      = 0.10

P_NEG_SQUARE   = 0.25
P_NEG_TRIANGLE = 0.25
P_NEG_STRONG_OVAL = 0.25
P_NEG_OPEN_ARC = 0.25

JITTER_FRAC = 0.08   # how wobbly "hand-drawn" stroke is
STEP_DEG    = 6      # segment step (smaller -> smoother)


# ===== helpers =====
def _crop_pad_resize(img, out_size=OUT_SIZE, margin_frac=MARGIN_FRAC):
    g = img.convert("L")
    bbox = ImageOps.invert(g).getbbox()
    if bbox is None:
        return Image.new("L", (out_size, out_size), 255)
    crop = g.crop(bbox)
    w, h = crop.size
    side = max(w, h)
    margin = int(side * margin_frac)
    sq = Image.new("L", (side + 2*margin, side + 2*margin), 255)
    sq.paste(crop, (margin + (side - w)//2, margin + (side - h)//2))
    return sq.resize((out_size, out_size), Image.BILINEAR)


def _ellipse_bbox(cx, cy, rx, ry):
    return [cx - rx, cy - ry, cx + rx, cy + ry]


def _draw_wobbly_circle(draw, cx, cy, r, width=2, jitter_frac=JITTER_FRAC):
    pts = []
    for deg in range(0, 360 + STEP_DEG, STEP_DEG):
        jr = r * (1 + random.uniform(-jitter_frac, jitter_frac))
        ang = math.radians(deg)
        x = cx + jr * math.cos(ang)
        y = cy + jr * math.sin(ang)
        pts.append((x, y))
    draw.line(pts, fill=0, width=width)


def _draw_wobbly_arc_with_gap(draw, cx, cy, r, gap_deg, gap_center=None, width=2,
                              jitter_frac=JITTER_FRAC):
    """
    Draw a circle-like stroke with a gap (open arc). gap_deg is the size of the gap in degrees.
    Used for:
      - positives: tiny gap (<= MAX_SMALL_GAP_DEG)
      - negatives: big gap (OPEN_GAP_DEG_RANGE)
    """
    if gap_center is None:
        gap_center = random.uniform(0, 360)
    half = gap_deg / 2.0

    def in_gap(d):
        # True if angle d (deg) lies within the gap window [center-half, center+half] modulo 360
        a = (d - (gap_center - half)) % 360.0
        return a <= gap_deg

    seg = []
    for deg in range(0, 360 + STEP_DEG, STEP_DEG):
        if in_gap(deg):
            if len(seg) > 1:
                draw.line(seg, fill=0, width=width)
            seg = []
            continue
        jr = r * (1 + random.uniform(-jitter_frac, jitter_frac))
        ang = math.radians(deg)
        x = cx + jr * math.cos(ang)
        y = cy + jr * math.sin(ang)
        seg.append((x, y))
    if len(seg) > 1:
        draw.line(seg, fill=0, width=width)


# ===== main image generator =====
def make_image(circle=True):
    """
    Draw on a big canvas with randomness, then crop->pad->resize to OUT_SIZE.
    Positive class (circle=True):
      - wobbly closed circle
      - mild oval (rx≈ry)
      - tiny-gap arc (gap <= MAX_SMALL_GAP_DEG)
    Negative class (circle=False):
      - square, triangle, strong oval, open arc with big gap (> 5%)
    """
    img = Image.new("L", (CANVAS, CANVAS), 255)
    d = ImageDraw.Draw(img)
    w = random.randint(*THICKNESS_RANGE)

    if circle:
        u = random.random()
        # Choose a base radius
        base = random.randint(int(CANVAS*0.15), int(CANVAS*0.35))
        rx = base
        ry = int(base * random.uniform(*CIRCLE_STRETCH))
        r  = int((rx + ry) / 2)

        # center so stroke stays on canvas
        cx = random.randint(r + 2, CANVAS - r - 2)
        cy = random.randint(r + 2, CANVAS - r - 2)

        if u < P_POS_WOBBLY_CIRCLE:
            _draw_wobbly_circle(d, cx, cy, r, width=w)
        elif u < P_POS_WOBBLY_CIRCLE + P_POS_MILD_OVAL:
            d.ellipse(_ellipse_bbox(cx, cy, rx, ry), outline=0, width=w)
        else:
            # Tiny-gap (<= 5%) circle -> still positive
            gap = random.uniform(0.0, MAX_SMALL_GAP_DEG)
            _draw_wobbly_arc_with_gap(d, cx, cy, r, gap_deg=gap, width=w)

    else:
        v = random.random()
        if v < P_NEG_SQUARE:
            a = random.randint(int(CANVAS*0.2), int(CANVAS*0.5))
            x = random.randint(0, CANVAS - a)
            y = random.randint(0, CANVAS - a)
            d.rectangle([x, y, x+a, y+a], outline=0, width=w)

        elif v < P_NEG_SQUARE + P_NEG_TRIANGLE:
            pts = [(random.randint(0, CANVAS), random.randint(0, CANVAS)) for _ in range(3)]
            d.polygon(pts, outline=0)

        elif v < P_NEG_SQUARE + P_NEG_TRIANGLE + P_NEG_STRONG_OVAL:
            rx = random.randint(int(CANVAS*0.12), int(CANVAS*0.28))
            stretch = random.uniform(STRONG_OVAL_MIN, STRONG_OVAL_MIN + 0.6)
            if random.random() < 0.5:
                ry = int(rx * stretch)
            else:
                ry = rx
                rx = int(ry * stretch)
            rx = max(1, min(rx, CANVAS//2 - 2))
            ry = max(1, min(ry, CANVAS//2 - 2))
            cx = random.randint(rx + 2, CANVAS - rx - 2)
            cy = random.randint(ry + 2, CANVAS - ry - 2)
            d.ellipse(_ellipse_bbox(cx, cy, rx, ry), outline=0, width=w)

        else:
            # Open arc with a big gap (> 5%): NOT a circle
            r = random.randint(int(CANVAS*0.12), int(CANVAS*0.32))
            cx = random.randint(r + 2, CANVAS - r - 2)
            cy = random.randint(r + 2, CANVAS - r - 2)
            gap = random.uniform(*OPEN_GAP_DEG_RANGE)
            _draw_wobbly_arc_with_gap(d, cx, cy, r, gap_deg=gap, width=w)

    return _crop_pad_resize(img)


# ===== dataset writer =====
def save_split(root="data", split="train", n_per_class=500, clear=False):
    split_dir = os.path.join(root, split)
    if clear and os.path.isdir(split_dir):
        shutil.rmtree(split_dir)

    for label, is_circle in [("circle", True), ("not_circle", False)]:
        folder = os.path.join(split_dir, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_per_class):
            img = make_image(circle=is_circle)
            img.save(os.path.join(folder, f"{label}_{i:04d}.png"))


if __name__ == "__main__":
    save_split(split="train", n_per_class=500, clear=True)
    save_split(split="val",   n_per_class=100, clear=True)
    print("Wrote data/train/... and data/val/...")
