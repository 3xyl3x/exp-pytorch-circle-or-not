# live_draw_classify.py
import tkinter as tk
from PIL import Image, ImageDraw,ImageOps
import torch
from torchvision import transforms
from model_cnn import TinyCNN

# --- Settings ---
CKPT_PATH = "models/circle_vs_not.pth"  # trained weights from train.py
CANVAS_SIZE = 280                       # pixels (square)
BRUSH = 10                              # stroke width in UI
NORMALIZE = False                       # set True ONLY if you normalized during training

# --- Load model once ---
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CKPT_PATH, map_location=device)
model = TinyCNN().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
classes = ckpt.get("classes", ["circle", "not_circle"])
idx_circle = classes.index("circle") if "circle" in classes else 0

# --- Transforms (must match training preprocessing) ---
steps = [
    transforms.Grayscale(1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
]
if NORMALIZE:
    steps.append(transforms.Normalize(mean=[0.5], std=[0.5]))
tfm = transforms.Compose(steps)

# --- Tkinter app with a PIL buffer we draw into ---
root = tk.Tk()
root.title("Draw a circle (press & drag). Release to classify.")

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="crosshair")
canvas.pack(padx=8, pady=8)

status = tk.Label(root, text="Draw a circle, then release mouse to classify.", font=("Segoe UI", 11))
status.pack(pady=(0,8))

# Keep a white PIL image buffer mirroring the canvas
buf_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)  # white bg, black ink
buf_draw = ImageDraw.Draw(buf_img)

last = None
drawing = False
reset_after_id = None
def preprocess_drawn(img, out_size=32, margin_frac=0.12):
    """
    Crop to the drawn content, pad to square with white, then resize.
    Works best for hand drawings on white background.
    """
    g = img.convert("L")  # ensure grayscale
    # bbox of non-white content (invert so strokes become >0)
    bbox = ImageOps.invert(g).getbbox()
    if bbox is None:
        # nothing drawn; just return resized white
        return g.resize((out_size, out_size), Image.BILINEAR)

    crop = g.crop(bbox)
    w, h = crop.size
    side = max(w, h)
    margin = int(side * margin_frac)

    # square canvas with white background + margin
    sq = Image.new("L", (side + 2*margin, side + 2*margin), 255)
    sq.paste(crop, (margin + (side - w)//2, margin + (side - h)//2))

    # final resize
    return sq.resize((out_size, out_size), Image.BILINEAR)
def clear_canvas():
    global buf_img, buf_draw, last, drawing
    canvas.delete("all")
    buf_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    buf_draw = ImageDraw.Draw(buf_img)
    last = None
    drawing = False
    status.config(text="Draw a circle, then release mouse to classify.")

def on_press(event):
    global last, drawing, reset_after_id
    drawing = True
    last = (event.x, event.y)
    # If a reset was scheduled, cancel it (user started a new drawing)
    if reset_after_id is not None:
        root.after_cancel(reset_after_id)

def on_move(event):
    global last
    if not drawing or last is None:
        return
    x0, y0 = last
    x1, y1 = event.x, event.y
    # draw on the visible canvas
    canvas.create_line(x0, y0, x1, y1, width=BRUSH, fill="black", capstyle=tk.ROUND, smooth=True)
    # mirror on the PIL buffer
    buf_draw.line([x0, y0, x1, y1], fill=0, width=BRUSH)
    last = (x1, y1)

def classify_current():
    # Convert PIL buffer -> model input
    proc = preprocess_drawn(buf_img, out_size=32)
    x = tfm(proc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    conf, idx = float(probs.max().item()), int(probs.argmax().item())
    label = classes[idx]
    # Present as YES (circle) / NO
    if idx == idx_circle:
        status.config(text=f"YES (circle)  —  confidence {conf:.1%}")
    else:
        status.config(text=f"NO (not circle)  —  confidence {conf:.1%}")
    # Auto reset after 2s
    schedule_reset()

def schedule_reset():
    global reset_after_id
    reset_after_id = root.after(2000, clear_canvas)

def on_release(event):
    global drawing, last
    drawing = False
    last = None
    classify_current()

canvas.bind("<ButtonPress-1>", on_press)
canvas.bind("<B1-Motion>", on_move)
canvas.bind("<ButtonRelease-1>", on_release)

clear_canvas()
root.mainloop()
