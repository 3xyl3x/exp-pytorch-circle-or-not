from PIL import Image, ImageDraw
import os, random, shutil

# ---- settings ----
IMG_SIZE = 32
THICKNESS_RANGE = (1, 3)
CIRCLE_STRETCH = (0.85, 1.15)   # circle-ish (±15%)
STRONG_OVAL_MIN = 1.5           # stretched oval = not_circle

def _ellipse_bbox(cx, cy, rx, ry):
    return [cx - rx, cy - ry, cx + rx, cy + ry]

def _clamp_radii(size, rx, ry, margin=2):
    """Ensure rx, ry fit inside the image with a small margin."""
    max_r = max(1, size // 2 - margin)
    rx = max(1, min(rx, max_r))
    ry = max(1, min(ry, max_r))
    return rx, ry

def make_image(size=IMG_SIZE, circle=True):
    """
    circle=True  -> round/oval (slight distortion allowed)
    circle=False -> square, triangle, scribble, or strong oval
    """
    img = Image.new("L", (size, size), 255)  # white background
    d = ImageDraw.Draw(img)
    w = random.randint(*THICKNESS_RANGE)

    if circle:
        # base radius and mild stretch
        rx = random.randint(int(size*0.15), int(size*0.35))
        ry = int(rx * random.uniform(*CIRCLE_STRETCH))
        rx, ry = _clamp_radii(size, rx, ry)

        # choose a valid center after clamping
        cx = random.randint(rx, size - rx)
        cy = random.randint(ry, size - ry)

        bbox = _ellipse_bbox(cx, cy, rx, ry)
        d.ellipse(bbox, outline=0, width=w)

    else:
        choice = random.random()
        if choice < 0.33:  # square
            a = random.randint(int(size*0.2), int(size*0.55))
            x = random.randint(0, size - a)
            y = random.randint(0, size - a)
            d.rectangle([x, y, x+a, y+a], outline=0, width=w)

        elif choice < 0.66:  # triangle
            pts = [(random.randint(0, size), random.randint(0, size)) for _ in range(3)]
            d.polygon(pts, outline=0)

        else:  # strong oval
            rx = random.randint(int(size*0.15), int(size*0.30))
            stretch = random.uniform(STRONG_OVAL_MIN, STRONG_OVAL_MIN + 0.6)
            if random.random() < 0.5:
                ry = int(rx * stretch)
            else:
                ry = rx
                rx = int(ry * stretch)

            rx, ry = _clamp_radii(size, rx, ry)
            cx = random.randint(rx, size - rx)
            cy = random.randint(ry, size - ry)

            bbox = _ellipse_bbox(cx, cy, rx, ry)
            d.ellipse(bbox, outline=0, width=w)

    return img

def save_split(root="data", split="train", n_per_class=500, size=IMG_SIZE, clear=False):
    """
    Writes images to data/<split>/{circle,not_circle}.
    If clear=True, deletes old split folder first.
    """
    split_dir = os.path.join(root, split)
    if clear and os.path.isdir(split_dir):
        shutil.rmtree(split_dir)

    for label, is_circle in [("circle", True), ("not_circle", False)]:
        folder = os.path.join(split_dir, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(n_per_class):
            img = make_image(size=size, circle=is_circle)
            img.save(os.path.join(folder, f"{label}_{i:04d}.png"))

if __name__ == "__main__":
    save_split(split="train", n_per_class=500, size=IMG_SIZE, clear=True)
    save_split(split="val",   n_per_class=100, size=IMG_SIZE, clear=True)
    print("Wrote data/train/... and data/val/...")
