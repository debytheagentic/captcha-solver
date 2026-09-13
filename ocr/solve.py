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

from PIL import Image, ImageEnhance

_ocr_instance = None
_init_lock = threading.Lock()
# ponytail: global lock, per-worker locks if OCR throughput demands it
_ocr_lock = asyncio.Lock()

_MATH_RE = re.compile(
    r"^\s*(-?\d+)\s*([\+\-\*xX\/\:\u00d7\u00f7–—])\s*(-?\d+)(?:\s*=.*)?\s*$"
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
    gray = im.convert("L")
    pixels = gray.load()
    has_ink = [
        any(pixels[x, y] < 200 for y in range(h))
        for x in range(w)
    ]
    min_gap = max(6, int(w * 0.08))
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

    for g_start, g_end in runs:
        gap_len = g_end - g_start
        if gap_len >= min_gap and g_end >= int(w * 0.65) and any(has_ink[x] for x in range(0, g_start)):
            icon_cols = [x for x in range(g_end, w) if has_ink[x]]
            if icon_cols:
                icon_width = icon_cols[-1] - icon_cols[0] + 1
                if icon_width < w * 0.25:
                    return im.crop((0, 0, g_start, h))
    return im


def preprocess_image(data: bytes) -> bytes:
    """Preprocess image bytes: handle transparency and trim embedded UI icons."""
    try:
        im = Image.open(io.BytesIO(data))
    except Exception:
        return data

    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")

    im = _detect_and_trim_icon(im)

    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


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
    Returns (True, result_int, str(result_int)) if valid arithmetic.
    Returns (False, None, text.strip()) if not matched or division by zero.
    """
    m = _MATH_RE.match(text)
    if not m:
        return (False, None, text.strip())
    a_str, op_sym, b_str = m.groups()
    try:
        a = int(a_str)
        b = int(b_str)
    except ValueError:
        return (False, None, text.strip())

    op = _OP_MAP.get(op_sym)
    if not op:
        return (False, None, text.strip())
    if op == "/" and b == 0:
        return (False, None, text.strip())

    op_fn = _OPS.get(op)
    if not op_fn:
        return (False, None, text.strip())

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
        is_matched, val, ans_str = evaluate_math_expression(raw_ocr_str)
        if is_matched:
            is_math = True
            value = val
            final_answer_str = ans_str
        else:
            has_arithmetic_hint = bool(re.search(r"[\+\-\*xX\/\:\u00d7\u00f7–—=\d]", raw_ocr_str))
            if mode in ("math", "force_math") or has_arithmetic_hint:
                try:
                    enh_bytes = _enhance_image(processed_bytes)
                    async with _ocr_lock:
                        ocr = get_ocr()
                        retry_ocr = await asyncio.to_thread(ocr.classification, enh_bytes)
                    retry_ocr_str = (retry_ocr or "").strip()
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
