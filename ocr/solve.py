"""Base64 image OCR and math expression solver using ddddocr."""
import asyncio
import base64
import binascii
import operator
import re
import threading
import time
from typing import Optional

_ocr_instance = None
_init_lock = threading.Lock()
# ponytail: global lock, per-worker locks if OCR throughput demands it
_ocr_lock = asyncio.Lock()

_MATH_RE = re.compile(r"^\s*(-?\d+)\s*([\+\-\*xX\/\:])\s*(-?\d+)\s*=?\s*$")
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


def evaluate_math_expression(text: str) -> tuple[bool, int | None, str]:
    """Parse strict arithmetic expression (add, sub, mul, floordiv).

    Normalizes x/X -> * and : -> /.
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

    op = "*" if op_sym in ("x", "X") else ("/" if op_sym == ":" else op_sym)
    op_fn = _OPS.get(op)
    if not op_fn:
        return (False, None, text.strip())
    if op == "/" and b == 0:
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

    async with _ocr_lock:
        ocr = get_ocr()
        raw_ocr = await asyncio.to_thread(ocr.classification, img_bytes)

    raw_ocr_str = (raw_ocr or "").strip()
    mode = (math_mode or "auto").lower()

    is_math = False
    value = None
    final_answer_str = raw_ocr_str
    error: Optional[str] = None

    if mode in ("math", "force_math"):
        is_matched, val, ans_str = evaluate_math_expression(raw_ocr_str)
        if is_matched:
            is_math = True
            value = val
            final_answer_str = ans_str
        else:
            is_math = False
            value = None
            final_answer_str = ""
            error = f"Failed to parse math expression from OCR text: '{raw_ocr_str}'"
    elif mode == "text":
        is_math = False
        value = None
        final_answer_str = raw_ocr_str
    else:  # "auto"
        is_matched, val, ans_str = evaluate_math_expression(raw_ocr_str)
        if is_matched:
            is_math = True
            value = val
            final_answer_str = ans_str
        else:
            is_math = False
            value = None
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
