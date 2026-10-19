"""Synthetic condition augmentation for aligned 112x112 BGR face crops (ArcFace template geometry).

Targets the statement's conditions: masks, head coverings (hijab/dupatta, turban, cap), glasses, facial hair,
poor lighting, blur, low resolution and CCTV/mobile compression. Pure NumPy/OpenCV so it can run inside a
Colab DataLoader. These are rough overlays, not photorealistic; they are used for TRAINING robustness and
for a clearly-labelled "stress test" of the baseline, never mixed into the real-photo evidence.

Template landmarks (112x112): eyes (38,52) (74,52), nose (56,72), mouth y~92, chin y~110.
"""
import cv2
import numpy as np

S = 112
# BGR palettes
MASK_COLORS = [(200, 150, 70), (235, 235, 235), (45, 45, 45), (150, 190, 110), (170, 140, 210)]
CLOTH_COLORS = [(60, 40, 160), (140, 60, 90), (40, 110, 60), (30, 30, 30), (200, 200, 205), (60, 120, 190), (150, 90, 40)]


def _alpha_blend(img, layer, mask, soft=1.5):
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), soft)[..., None]
    return np.clip(img * (1 - m) + layer * m, 0, 255).astype(np.uint8)


def _poly_mask(pts):
    m = np.zeros((S, S), np.float32)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0)
    return m


def _jitter(pts, rng, shift=3.0, scale=0.06):
    pts = np.asarray(pts, np.float32)
    c = np.array([56.0, 66.0], np.float32)
    return (pts - c) * (1 + rng.uniform(-scale, scale)) + c + rng.uniform(-shift, shift, 2)


def _fabric(rng, color, strength=14):
    """Cloth texture: base colour with soft folds and fine weave noise."""
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    fold = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) / rng.uniform(5, 11) + rng.uniform(0, 6)) * strength
    tex = np.ones((S, S, 3), np.float32) * np.array(color, np.float32) + fold[..., None] + rng.normal(0, 3, (S, S, 1))
    return np.clip(tex, 0, 255)


def surgical_mask(img, rng):
    pts = _jitter([(14, 70), (40, 61), (56, 58), (72, 61), (98, 70), (100, 92), (86, 110), (56, 114), (26, 110), (12, 92)], rng)
    m = _poly_mask(pts)
    layer = np.ones_like(img, np.float32) * np.array(MASK_COLORS[rng.integers(len(MASK_COLORS))], np.float32)
    for y in np.linspace(pts[:, 1].min() + 10, pts[:, 1].max() - 6, 3):  # pleats
        cv2.line(layer, (10, int(y)), (102, int(y + rng.uniform(-2, 2))), tuple(int(v * 0.7) for v in layer[0, 0]), 1)
    layer *= 0.92 + 0.08 * rng.random()
    return _alpha_blend(img, layer, m, soft=1.2)


def cloth_face_cover(img, rng):
    """Scarf / dupatta wrapped over nose and mouth (common in South Asia, e.g. the IMFW dataset's scarves)."""
    top = rng.uniform(54, 62)
    pts = _jitter([(8, 74), (36, top), (56, top - 4), (76, top), (104, 74), (106, 100), (88, 116), (56, 120), (24, 116), (6, 100)], rng)
    layer = _fabric(rng, CLOTH_COLORS[rng.integers(len(CLOTH_COLORS))])
    return _alpha_blend(img, layer, _poly_mask(pts), soft=1.6)


def _frame_cover(img, rng, cx, cy, rx, ry, color_idx=None):
    """Fabric everywhere outside an elliptical visible-face window (hijab / dupatta)."""
    win = np.zeros((S, S), np.float32)
    cv2.ellipse(win, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    layer = _fabric(rng, CLOTH_COLORS[rng.integers(len(CLOTH_COLORS))], strength=18)
    return _alpha_blend(img, layer, 1.0 - win, soft=2.2)


def hijab(img, rng):
    return _frame_cover(img, rng, 56 + rng.uniform(-2, 2), 70 + rng.uniform(-2, 3), rng.uniform(31, 35), rng.uniform(40, 44))


def dupatta(img, rng):
    """Looser head covering: more forehead/hair visible, covers one side more."""
    out = _frame_cover(img, rng, 56 + rng.uniform(-6, 6), 66 + rng.uniform(-2, 4), rng.uniform(34, 40), rng.uniform(46, 50))
    return out


def cap(img, rng):
    drop = rng.uniform(28, 36)
    pts = [(0, 0), (112, 0), (112, drop - 4), (84, drop + 2), (56, drop + 4), (28, drop + 2), (0, drop - 4)]
    m = _poly_mask(_jitter(pts, rng, shift=1.5, scale=0.02))
    layer = _fabric(rng, CLOTH_COLORS[rng.integers(len(CLOTH_COLORS))], strength=6)
    out = _alpha_blend(img, layer, m, soft=1.0)
    shade = np.zeros((S, S), np.float32)  # brim shadow on forehead
    shade[int(drop):int(drop) + 9] = np.linspace(0.45, 0.0, 9)[:, None]
    return np.clip(out * (1 - shade[..., None]), 0, 255).astype(np.uint8)


def turban(img, rng):
    pts = [(0, 0), (112, 0), (112, 50), (102, 42), (82, 35), (56, 33), (30, 35), (10, 42), (0, 50)]
    m = _poly_mask(_jitter(pts, rng, shift=1.5, scale=0.03))
    layer = _fabric(rng, CLOTH_COLORS[rng.integers(len(CLOTH_COLORS))], strength=22)
    for k in range(-3, 8):  # diagonal wraps
        x0 = k * 18 + rng.uniform(-2, 2)
        cv2.line(layer, (int(x0), 0), (int(x0 + 40), 50), (0, 0, 0), 1, cv2.LINE_AA)
    return _alpha_blend(img, layer, m, soft=1.2)


def glasses(img, rng, sunglasses=False):
    out = img.copy()
    c = [(38 + rng.uniform(-1, 1), 52 + rng.uniform(-1, 1)), (74 + rng.uniform(-1, 1), 52 + rng.uniform(-1, 1))]
    ax = (int(rng.uniform(12, 15)), int(rng.uniform(9, 12)))
    if sunglasses:
        m = np.zeros((S, S), np.float32)
        for p in c:
            cv2.ellipse(m, (int(p[0]), int(p[1])), ax, 0, 0, 360, 1.0, -1)
        out = _alpha_blend(out, np.ones_like(img, np.float32) * rng.uniform(15, 50), m, soft=1.0)
    frame = tuple(int(v) for v in rng.integers(10, 90, 3))
    th = int(rng.integers(1, 3))
    for p in c:
        cv2.ellipse(out, (int(p[0]), int(p[1])), ax, 0, 0, 360, frame, th, cv2.LINE_AA)
    cv2.line(out, (int(c[0][0] + ax[0]), int(c[0][1] - 2)), (int(c[1][0] - ax[0]), int(c[1][1] - 2)), frame, th, cv2.LINE_AA)
    cv2.line(out, (int(c[0][0] - ax[0]), int(c[0][1])), (0, int(c[0][1] - 2)), frame, th, cv2.LINE_AA)
    cv2.line(out, (int(c[1][0] + ax[0]), int(c[1][1])), (S - 1, int(c[1][1] - 2)), frame, th, cv2.LINE_AA)
    return out


def beard(img, rng):
    """Dark stubble/beard on jaw + moustache. Crude on purpose (texture, not a drawn shape)."""
    jaw = [(16, 76), (30, 84), (56, 80), (82, 84), (96, 76), (98, 96), (86, 112), (56, 118), (26, 112), (14, 96)]
    m = _poly_mask(_jitter(jaw, rng, shift=2, scale=0.03))
    cv2.ellipse(m, (56, 83), (14, 4), 0, 0, 360, 1.0, -1)  # moustache
    cv2.ellipse(m, (56, 98), (12, 8), 0, 0, 360, 0.0, -1)   # keep mouth region lighter
    dark = rng.uniform(15, 55)
    layer = np.ones_like(img, np.float32) * dark + rng.normal(0, 14, (S, S, 1))
    return _alpha_blend(img, np.clip(layer, 0, 255), m * rng.uniform(0.55, 0.9), soft=2.0)


# ---- degradations -------------------------------------------------------------------------------------------
def low_light(img, rng):
    f = (img.astype(np.float32) / 255.0) ** rng.uniform(1.6, 2.8) * rng.uniform(0.35, 0.8)
    f = f * 255 + rng.normal(0, rng.uniform(4, 12), img.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def harsh_light(img, rng):
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    g = ((xx - 56) * np.cos(ang) + (yy - 56) * np.sin(ang)) / 56.0
    gain = 1.0 + g * rng.uniform(0.5, 0.9)
    return np.clip(img.astype(np.float32) * np.clip(gain, 0.15, 1.9)[..., None], 0, 255).astype(np.uint8)


def blur(img, rng):
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5, 7, 9]))
        return cv2.GaussianBlur(img, (k, k), 0)
    k = int(rng.integers(5, 15))
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    kern = cv2.warpAffine(kern, M, (k, k))
    return cv2.filter2D(img, -1, kern / max(kern.sum(), 1e-6))


def low_res(img, rng):
    s = int(rng.integers(20, 57))
    small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (S, S), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))


def jpeg(img, rng):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(8, 45))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def cctv(img, rng):
    """Cheap-camera look: low resolution + colour cast + sensor noise + heavy compression."""
    out = low_res(img, rng).astype(np.float32)
    out = out * rng.uniform(0.85, 1.15, 3) + rng.normal(0, rng.uniform(3, 9), out.shape)
    gray = out.mean(2, keepdims=True)
    out = gray + (out - gray) * rng.uniform(0.5, 1.0)
    return jpeg(np.clip(out, 0, 255).astype(np.uint8), rng)


OCCLUSIONS = {"surgical_mask": surgical_mask, "cloth_cover": cloth_face_cover, "hijab": hijab, "dupatta": dupatta,
              "cap": cap, "turban": turban}
ACCESSORIES = {"glasses": glasses, "sunglasses": lambda i, r: glasses(i, r, True), "beard": beard}
DEGRADATIONS = {"low_light": low_light, "harsh_light": harsh_light, "blur": blur, "low_res": low_res, "jpeg": jpeg, "cctv": cctv}


def apply(img, name, rng):
    for table in (OCCLUSIONS, ACCESSORIES, DEGRADATIONS):
        if name in table:
            return table[name](img, rng)
    raise KeyError(name)


def random_augment(img, rng=None, p_occlusion=0.45, p_accessory=0.25, p_degrade=0.6):
    """One occlusion at most (masks/coverings are mutually exclusive), optional accessory, 0-2 degradations.
    Returns (image, list_of_tags). Order: occlusion -> accessory -> degradation (like a real camera pipeline)."""
    rng = rng or np.random.default_rng()
    tags = []
    if rng.random() < p_occlusion:
        n = str(rng.choice(list(OCCLUSIONS)))
        img = OCCLUSIONS[n](img, rng); tags.append(n)
    if rng.random() < p_accessory:
        covered_mouth = bool({"surgical_mask", "cloth_cover"} & set(tags))  # a beard under a mask would be invisible
        w = {"glasses": 0.4, "sunglasses": 0.4, "beard": 0.0 if covered_mouth else 0.2}  # beard rare: gender/age unknown here
        n = str(rng.choice(list(w), p=np.array(list(w.values())) / sum(w.values())))
        img = ACCESSORIES[n](img, rng); tags.append(n)
    if rng.random() < p_degrade:
        for n in rng.choice(list(DEGRADATIONS), size=int(rng.integers(1, 3)), replace=False):
            img = DEGRADATIONS[str(n)](img, rng); tags.append(str(n))
    return img, tags
