"""Base64 image OCR and math expression solver using ddddocr."""
import asyncio
import base64
import binascii
import io
import operator
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

_ocr_instance = None
_init_lock = threading.Lock()
# ponytail: global lock, per-worker locks if OCR throughput demands it
_ocr_lock = asyncio.Lock()

_MATH_RE = re.compile(
    r"^\s*(-?\d+)\s*([\+\-\*xX\/\:\u00d7\u00f7–—])\s*(-?\d+)\s*$"
)
_OP_MAP = {
    "+": "+",
    "-": "-",
    "–": "-",
    "—": "-",
    "*": "*",
    "x": "*",
    "X": "*",
    "\u00d7": "*",
    "/": "/",
    ":": "/",
    "\u00f7": "/",
}
_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.floordiv,
}

def get_ocr():
    """Thread-safe lazy-loading singleton for ddddocr."""
    global _ocr_instance
    if _ocr_instance is None:
        with _init_lock:
            if _ocr_instance is None:
                import ddddocr
                _ocr_instance = ddddocr.DdddOcr(show_ad=False)
    return _ocr_instance


def decode_image(b64_str: str) -> bytes:
    """Decode raw base64 or data URL into image bytes.

    Strips whitespace and pads missing '=' if needed.
    Raises ValueError on invalid/corrupted base64 or non-string input.
    """
    if not isinstance(b64_str, str):
        raise ValueError("Image data must be a string")
    s = b64_str.strip()
    if not s:
        raise ValueError("Empty image data")
    if s.startswith("data:") and ";base64," in s:
        s = s.split(";base64,", 1)[1].strip()
    s = re.sub(r"\s+", "", s)
    if not s:
        raise ValueError("Empty base64 payload")
    missing_padding = len(s) % 4
    if missing_padding:
        s += "=" * (4 - missing_padding)
    try:
        data = base64.b64decode(s, validate=True)
        if not data:
            raise ValueError("Decoded image data is empty")
        return data
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 string: {e}") from e


def _detect_and_trim_icon(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w < 30:
        return im
    gray_arr = np.array(im.convert("L"))
    dark_counts = (gray_arr < 170).sum(axis=0)
    has_ink = dark_counts > 2

    min_gap = max(4, int(w * 0.05))
    runs = []
    in_gap = False
    start = 0
    for x in range(w):
        if not has_ink[x]:
            if not in_gap:
                in_gap = True
                start = x
        else:
            if in_gap:
                runs.append((start, x))
                in_gap = False
    if in_gap:
        runs.append((start, w))

    right_35_start = int(w * 0.65)
    for g_start, g_end in runs:
        gap_len = g_end - g_start
        if gap_len >= min_gap and g_end >= right_35_start and any(has_ink[x] for x in range(0, g_start)):
            icon_cols = [x for x in range(g_end, w) if has_ink[x]]
            if icon_cols:
                icon_width = icon_cols[-1] - icon_cols[0] + 1
                if icon_width < w * 0.25:
                    return im.crop((0, 0, g_start, h))
    return im


def _is_plus_operator_glyph(
    bw_arr: np.ndarray | Image.Image | bytes,
    x_start: Optional[int] = None,
    x_end: Optional[int] = None,
) -> bool:
    """Verify if an axis-aligned '+' operator glyph exists in the horizontal interval.

    Inspects connected components situated in [x_start, x_end] (defaulting to the center 50%
    of the image). Returns True if a component satisfies:
      - Width and height between 6px and 35px.
      - Aspect ratio 0.7 <= w / h <= 1.3.
      - Vertical center stroke coverage: vert_ink >= h * 0.7.
      - Horizontal center stroke coverage: horiz_ink >= w * 0.7.
    """
    if bw_arr is None:
        return False
    if isinstance(bw_arr, bytes):
        try:
            im = Image.open(io.BytesIO(bw_arr)).convert("L")
            arr = np.array(im)
        except Exception:
            return False
    elif isinstance(bw_arr, Image.Image):
        arr = np.array(bw_arr.convert("L"))
    elif isinstance(bw_arr, np.ndarray):
        arr = bw_arr
        if arr.size == 0:
            return False
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        return False

    if arr.dtype == bool:
        fg = arr.astype(np.uint8) * 255
    elif np.mean(arr) > 127:
        fg = (arr < 128).astype(np.uint8) * 255
    else:
        fg = (arr > 127).astype(np.uint8) * 255

    h_img, w_img = fg.shape[:2]
    if h_img < 6 or w_img < 6:
        return False

    if x_start is None:
        x_start = int(w_img * 0.25)
    if x_end is None:
        x_end = int(w_img * 0.75)
    if x_start > x_end:
        x_start, x_end = x_end, x_start

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx = x + w / 2.0

        if not (x_start <= cx <= x_end or (x >= x_start and x + w <= x_end)):
            continue

        if not (6 <= w <= 35 and 6 <= h <= 35):
            continue

        ar = w / float(h)
        if not (0.7 <= ar <= 1.3):
            continue

        comp_mask = (labels[y : y + h, x : x + w] == i)
        mid_x = w // 2
        mid_cols = range(max(0, mid_x - 2), min(w, mid_x + 3))
        vert_ink = max((comp_mask[:, c].sum() for c in mid_cols), default=0)

        mid_y = h // 2
        mid_rows = range(max(0, mid_y - 2), min(h, mid_y + 3))
        horiz_ink = max((comp_mask[r, :].sum() for r in mid_rows), default=0)

        if vert_ink >= h * 0.7 and horiz_ink >= w * 0.7:
            return True

    return False


def _correct_math_operator(text: str, img_data: bytes | np.ndarray | Image.Image) -> str:
    """If OCR returned '*' or 'x' operator, verify geometrically if the glyph is actually '+'."""
    if not re.search(r"[\*xX\u00d7]", text):
        return text
    try:
        if _is_plus_operator_glyph(img_data):
            if re.search(r"(?<=\d)\s*[\*xX\u00d7]\s*(?=\d)", text):
                return re.sub(r"(?<=\d)\s*[\*xX\u00d7]\s*(?=\d)", "+", text, count=1)
            return re.sub(r"[\*xX\u00d7]", "+", text, count=1)
    except Exception:
        pass
    return text


def _scaled_recovery_image(data: bytes) -> bytes:
    """Prepare scaled recovery image: 1.5x bicubic upscale, binarize, and pad 8px white margin."""
    try:
        im = Image.open(io.BytesIO(data)).convert("L")
        gray = np.array(im)
        up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        _, binarized = cv2.threshold(up, 160, 255, cv2.THRESH_BINARY)
        im_bin = Image.fromarray(binarized)
        padded = ImageOps.expand(im_bin, border=(8, 4, 8, 4), fill="white")
        out = io.BytesIO()
        padded.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return data


def preprocess_image(data: bytes) -> bytes:
    """Preprocess image bytes: handle transparency, dark mode, neon text, and trim icons."""
    try:
        im = Image.open(io.BytesIO(data))
    except Exception:
        return data

    try:
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")

        arr = np.array(im)
        if arr.size == 0:
            return data
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return data
        mean_lum = np.mean(gray)
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        is_dark = mean_lum < 115 or np.mean(corners) < 100

        if is_dark:
            g = arr[:, :, 1].astype(int)
            r = arr[:, :, 0].astype(int)
            neon_mask = (g > 110) & (g > r + 20)
            if neon_mask.sum() > 150:
                text_white = np.where(neon_mask, 255, 0).astype(np.uint8)
            else:
                _, text_white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened = cv2.morphologyEx(text_white, cv2.MORPH_OPEN, kernel)

            y_idx, x_idx = np.where(opened > 0)
            if y_idx.size > 0:
                cropped = opened[
                    max(0, int(y_idx.min()) - 5) : min(h, int(y_idx.max()) + 6),
                    max(0, int(x_idx.min()) - 5) : min(w, int(x_idx.max()) + 6),
                ]
            else:
                cropped = opened

            final_arr = 255 - cropped
            im = Image.fromarray(final_arr)
        else:
            im = _detect_and_trim_icon(im)

        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return data


def _enhance_image(data: bytes) -> bytes:
    """Enhance image contrast for second-pass OCR retry."""
    try:
        im = Image.open(io.BytesIO(data))
        gray = im.convert("L")
        enhanced = ImageEnhance.Contrast(gray).enhance(2.5).convert("RGB")
        out = io.BytesIO()
        enhanced.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return data


def evaluate_math_expression(text: str) -> tuple[bool, int | None, str]:
    """Parse arithmetic expression (add, sub, mul, floordiv).

    Normalizes x/X/× -> *, :/÷ -> /, –/— -> -, + -> +, - -> -.
    Ignores trailing = or question prompts.
    Strips trailing prompt hallucination patterns (e.g. -0, =0, =7).
    Returns (True, result_int, str(result_int)) if valid arithmetic.
    Returns (False, None, text.strip()) if not matched or division by zero.
    """
    if not text or not isinstance(text, str):
        return (False, None, "")
    raw = text.strip()

    # Base cleanup: strip trailing equals, question marks, dashes, whitespace
    has_prior_op = bool(re.search(r"\d\s*[\+\-\*xX\/\:\u00d7\u00f7–—]\s*\d", raw))
    if has_prior_op:
        s = re.sub(r"[\=\-\?_–—\s]+$", "", raw)
    else:
        s = re.sub(r"[\=\?_–—\s]+$", "", raw)

    m = _MATH_RE.match(s)
    if not m:
        # Strip trailing prompt hallucination patterns (e.g. 30+1-0 -> 30+1, 27+27=7 -> 27+27)
        cand = re.sub(r"[\-\=]\s*[0-9\?_\–—]+\s*$", "", s)
        cand = re.sub(r"[\=\-\?_–—\s]+$", "", cand)
        m_cand = _MATH_RE.match(cand)
        if m_cand:
            m = m_cand
            s = cand

    if not m:
        return (False, None, raw)

    a_str, op_sym, b_str = m.groups()
    try:
        a = int(a_str)
        b = int(b_str)
    except ValueError:
        return (False, None, raw)

    op = _OP_MAP.get(op_sym)
    if not op:
        return (False, None, raw)
    if op == "/" and b == 0:
        return (False, None, raw)

    op_fn = _OPS.get(op)
    if not op_fn:
        return (False, None, raw)

    res = op_fn(a, b)
    return (True, res, str(res))

async def solve_image_captcha(image_b64: str, math_mode: str = "auto") -> dict:
    """Solve an image captcha from base64 data.

    math_mode options:
      - 'auto': OCR first, evaluate as math if matched, else return raw OCR text.
      - 'math' / 'force_math': Must evaluate as math; returns error if not math.
      - 'text': Always return raw OCR text even if it looks like math.
    """
    start_time = time.time()
    img_bytes = decode_image(image_b64)
    processed_bytes = preprocess_image(img_bytes)

    raw_ocr = ""
    try:
        async with _ocr_lock:
            ocr = get_ocr()
            raw_ocr = await asyncio.to_thread(ocr.classification, processed_bytes)
    except Exception:
        raw_ocr = ""

    raw_ocr_str = (raw_ocr or "").strip()
    mode = (math_mode or "auto").lower()

    is_math = False
    value = None
    final_answer_str = raw_ocr_str
    error: Optional[str] = None

    if mode == "text":
        is_math = False
        value = None
        final_answer_str = raw_ocr_str
    else:
        # Operator disambiguation on Pass 1
        raw_ocr_str = _correct_math_operator(raw_ocr_str, processed_bytes)

        is_matched, val, ans_str = evaluate_math_expression(raw_ocr_str)
        if is_matched:
            is_math = True
            value = val
            final_answer_str = ans_str

        # Check for suspicious math in Pass 1:
        # e.g. operator * with product > 500 or second operand > 99 on a 2-operand captcha (such as 30 * 190 = 5700)
        is_suspicious = False
        if is_matched and val is not None:
            m_op = re.search(r"(-?\d+)\s*([\+\-\*xX\/\:\u00d7\u00f7–—])\s*(-?\d+)", raw_ocr_str)
            if m_op:
                op_sym = _OP_MAP.get(m_op.group(2))
                op2 = abs(int(m_op.group(3)))
                if op_sym == "*" and (abs(val) > 500 or op2 > 99):
                    is_suspicious = True
                elif op2 > 99:
                    is_suspicious = True

        has_arithmetic_hint = bool(re.search(r"[\+\-\*xX\/\:\u00d7\u00f7–—=\d]", raw_ocr_str))
        trigger_pass2 = (
            (not is_matched and (mode in ("math", "force_math") or has_arithmetic_hint))
            or is_suspicious
        )

        if trigger_pass2:
            try:
                pass2_bytes = _scaled_recovery_image(processed_bytes)
                async with _ocr_lock:
                    ocr = get_ocr()
                    pass2_ocr = await asyncio.to_thread(ocr.classification, pass2_bytes)
                pass2_ocr_str = (pass2_ocr or "").strip()
                pass2_ocr_str = _correct_math_operator(pass2_ocr_str, pass2_bytes)

                p2_matched, p2_val, p2_ans = evaluate_math_expression(pass2_ocr_str)
                if p2_matched:
                    is_math = True
                    value = p2_val
                    final_answer_str = p2_ans
                    raw_ocr_str = pass2_ocr_str
                elif not is_matched:
                    # Pass 2 didn't yield math, and Pass 1 was not matched: try contrast enhancement fallback
                    enh_bytes = _enhance_image(processed_bytes)
                    async with _ocr_lock:
                        ocr = get_ocr()
                        retry_ocr = await asyncio.to_thread(ocr.classification, enh_bytes)
                    retry_ocr_str = (retry_ocr or "").strip()
                    retry_ocr_str = _correct_math_operator(retry_ocr_str, enh_bytes)
                    retry_matched, retry_val, retry_ans = evaluate_math_expression(retry_ocr_str)
                    if retry_matched:
                        is_math = True
                        value = retry_val
                        final_answer_str = retry_ans
                        raw_ocr_str = retry_ocr_str
            except Exception:
                pass

        if not is_math:
            if mode in ("math", "force_math"):
                final_answer_str = ""
                error = f"Failed to parse math expression from OCR text: '{raw_ocr_str}'"
            else:
                final_answer_str = raw_ocr_str
    result = {
        "type": "image",
        "result": final_answer_str,
        "raw_text": raw_ocr_str,
        "is_math": is_math,
        "value": value,
        "elapsed": round(time.time() - start_time, 4),
        "method": "ddddocr-onnx",
    }
    if error:
        result["error"] = error
    return result
