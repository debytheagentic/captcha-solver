"""Comprehensive test suite for OCR and math captcha solver."""
import base64
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from ocr.solve import (
    decode_image,
    evaluate_math_expression,
    preprocess_image,
    solve_image_captcha,
)
from server import app, _is_solved


client = TestClient(app)


def _make_test_image_b64(text: str = "8*2=") -> str:
    """Generate a clean base64 PNG image containing text."""
    img = Image.new("RGB", (120, 32), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── 1. Mathematical expression parsing ─────────────────────────────────

def test_math_expression_addition():
    matched, val, s = evaluate_math_expression("3+1")
    assert matched is True
    assert val == 4
    assert s == "4"


def test_math_expression_multiplication_x():
    matched, val, s = evaluate_math_expression("3x2=")
    assert matched is True
    assert val == 6
    assert s == "6"


def test_math_expression_multiplication_X():
    matched, val, s = evaluate_math_expression("4X5 =")
    assert matched is True
    assert val == 20
    assert s == "20"


def test_math_expression_subtraction():
    matched, val, s = evaluate_math_expression("34 - 7")
    assert matched is True
    assert val == 27
    assert s == "27"


def test_math_expression_division_slash():
    matched, val, s = evaluate_math_expression("10 / 2")
    assert matched is True
    assert val == 5
    assert s == "5"


def test_math_expression_division_colon():
    matched, val, s = evaluate_math_expression("10:2=")
    assert matched is True
    assert val == 5
    assert s == "5"


def test_math_expression_negative_operands():
    matched, val, s = evaluate_math_expression("-5 + 8")
    assert matched is True
    assert val == 3
    assert s == "3"


def test_math_expression_division_by_zero():
    matched, val, s = evaluate_math_expression("10 / 0")
    assert matched is False
    assert val is None
    assert s == "10 / 0"

    matched_colon, val_colon, s_colon = evaluate_math_expression("10:0=")
    assert matched_colon is False
    assert val_colon is None



def test_math_expression_trailing_question_prompts():
    cases = [
        ("30 + 1 = ?", 31),
        ("4 + 16 = ?", 20),
        ("8 * 2 = ?", 16),
        ("3+0=?", 3),
        ("30 + 1 = _", 31),
        ("30 + 1 = 7", 31),
        ("30 + 1 =?", 31),
        ("4 + 16 =", 20),
    ]
    for expr, expected in cases:
        matched, val, s = evaluate_math_expression(expr)
        assert matched is True, f"Failed to match: {expr}"
        assert val == expected, f"Expected {expected}, got {val} for {expr}"
        assert s == str(expected)


def test_math_expression_dangling_prompts():
    cases = [
        ("27+27-", 54),
        ("30+1=?", 31),
        ("15-5=", 10),
    ]
    for expr, expected in cases:
        matched, val, s = evaluate_math_expression(expr)
        assert matched is True, f"Failed to match: {expr}"
        assert val == expected, f"Expected {expected}, got {val} for {expr}"
        assert s == str(expected)

def test_math_expression_unicode_operators():
    cases = [
        ("6 \u00d7 3", 18),
        ("20 \u00f7 4", 5),
        ("6\u00d73=", 18),
        ("20\u00f74=?", 5),
    ]
    for expr, expected in cases:
        matched, val, s = evaluate_math_expression(expr)
        assert matched is True, f"Failed to match: {expr}"
        assert val == expected, f"Expected {expected}, got {val} for {expr}"
        assert s == str(expected)


def test_math_expression_dash_variants_and_negatives():
    cases = [
        ("15 – 5", 10),
        ("15 — 5", 10),
        ("-5 + 8", 3),
        ("-10 – -4", -6),
    ]
    for expr, expected in cases:
        matched, val, s = evaluate_math_expression(expr)
        assert matched is True, f"Failed to match: {expr}"
        assert val == expected, f"Expected {expected}, got {val} for {expr}"
        assert s == str(expected)

# ── 2. Non-math text disambiguation ────────────────────────────────────

def test_non_math_disambiguation_alphanumeric():
    for sample in ("9x4b", "k8Xp", "7a+2", "hello", "123"):
        matched, val, s = evaluate_math_expression(sample)
        assert matched is False, f"Expected {sample} to not match math"
        assert val is None
        assert s == sample


# ── 3. Base64 decoding ────────────────────────────────────────────────

def test_decode_image_raw_base64():
    raw = base64.b64encode(b"fake-image-bytes").decode("utf-8")
    assert decode_image(raw) == b"fake-image-bytes"


def test_decode_image_data_url():
    raw = base64.b64encode(b"fake-image-bytes").decode("utf-8")
    data_url = f"data:image/png;base64,{raw}"
    assert decode_image(data_url) == b"fake-image-bytes"


def test_decode_image_whitespace_and_newlines():
    raw = base64.b64encode(b"fake-image-bytes").decode("utf-8")
    padded_str = f"  data:image/png;base64,\n {raw[:4]} \n {raw[4:]} \t \n"
    assert decode_image(padded_str) == b"fake-image-bytes"


def test_decode_image_missing_padding():
    # "hello world" -> aGVsbG8gd29ybGQ= (len 16)
    # "hello" -> aGVsbG8= (len 8)
    # "a" -> YQ== (len 4)
    # "ab" -> YWI= (len 4)
    raw = base64.b64encode(b"hello").decode("utf-8").rstrip("=")
    assert decode_image(raw) == b"hello"


def test_decode_image_invalid_inputs():
    invalid_cases = [
        "",
        "   ",
        "data:image/png;base64,",
        "not_valid_b64!!!@@#",
        12345,
    ]
    for case in invalid_cases:
        with pytest.raises(ValueError):
            decode_image(case)  # type: ignore


# ── 4. Async solve_image_captcha function ─────────────────────────────

@pytest.mark.asyncio
async def test_solve_image_captcha_auto_and_text():
    b64 = _make_test_image_b64("8*2=")
    res_auto = await solve_image_captcha(b64, math_mode="auto")
    assert res_auto["type"] == "image"
    assert res_auto["method"] == "ddddocr-onnx"
    assert "elapsed" in res_auto
    assert "raw_text" in res_auto

    res_text = await solve_image_captcha(b64, math_mode="text")
    assert res_text["is_math"] is False
    assert res_text["value"] is None
    assert res_text["result"] == res_text["raw_text"]


@pytest.mark.asyncio
async def test_solve_image_captcha_force_math_failure(monkeypatch):
    b64 = _make_test_image_b64("nonmath")
    # Simulate OCR returning non-math text
    class FakeOcr:
        def classification(self, b):
            return "k8Xp"

    monkeypatch.setattr("ocr.solve.get_ocr", lambda: FakeOcr())
    res = await solve_image_captcha(b64, math_mode="force_math")
    assert res["is_math"] is False
    assert res["value"] is None
    assert res["result"] == ""
    assert "error" in res


# ── 5. Server API integration tests ───────────────────────────────────

def test_health_endpoint_supported_types():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    for expected in ("image", "ocr", "math"):
        assert expected in data["supported_types"]


def test_solve_endpoint_validation_missing_image():
    resp = client.post("/solve", json={"type": "image"})
    assert resp.status_code == 400
    assert "image is required" in resp.json()["detail"]


def test_solve_endpoint_validation_invalid_base64():
    resp = client.post("/solve", json={"type": "image", "image": "invalid!!!"})
    assert resp.status_code == 400


def test_solve_endpoint_image_type(monkeypatch):
    class FakeOcr:
        def classification(self, b):
            return "3x2="

    monkeypatch.setattr("ocr.solve.get_ocr", lambda: FakeOcr())
    raw_b64 = base64.b64encode(b"fake").decode("utf-8")

    resp = client.post("/solve", json={"type": "image", "image": raw_b64, "math_mode": "auto"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "image"
    assert data["solved"] is True
    assert data["result"] == "6"
    assert data["value"] == 6
    assert data["is_math"] is True
    assert data["raw_text"] == "3x2="
    assert data["method"] == "ddddocr-onnx"


def test_solve_endpoint_math_type(monkeypatch):
    class FakeOcr:
        def classification(self, b):
            return "10 / 2"

    monkeypatch.setattr("ocr.solve.get_ocr", lambda: FakeOcr())
    raw_b64 = f"data:image/png;base64,{base64.b64encode(b'fake').decode('utf-8')}"

    resp = client.post("/solve", json={"type": "math", "image": raw_b64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "math"
    assert data["solved"] is True
    assert data["result"] == "5"
    assert data["value"] == 5
    assert data["is_math"] is True


def test_solve_endpoint_ocr_type(monkeypatch):
    class FakeOcr:
        def classification(self, b):
            return "10 / 2"

    monkeypatch.setattr("ocr.solve.get_ocr", lambda: FakeOcr())
    raw_b64 = base64.b64encode(b"fake").decode("utf-8")

    resp = client.post("/solve", json={"type": "ocr", "image": raw_b64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "ocr"
    assert data["solved"] is True
    assert data["result"] == "10 / 2"
    assert data["is_math"] is False
    assert data["value"] is None


def test_is_solved_predicate():
    assert _is_solved({"result": "6"}) is True
    assert _is_solved({"result": "k8Xp"}) is True
    assert _is_solved({"value": 0}) is True
    assert _is_solved({"value": None, "result": ""}) is False
    assert _is_solved({}) is False


# ── 6. Preprocessing and transparency ──────────────────────────────────

def test_preprocess_image_rgba_transparency():
    # Create RGBA image with fully transparent background and solid black pixel
    rgba = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    rgba.putpixel((10, 10), (0, 0, 0, 255))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")

    processed_bytes = preprocess_image(buf.getvalue())
    im = Image.open(io.BytesIO(processed_bytes))
    assert im.mode == "RGB"
    # Transparent background composited onto white (255, 255, 255)
    assert im.getpixel((0, 0)) == (255, 255, 255)
    # Black pixel remains black
    assert im.getpixel((10, 10)) == (0, 0, 0)


def test_preprocess_image_trim_right_edge_icon():
    # 150x40 image with text on the left and an isolated small icon on the right
    img = Image.new("RGB", (150, 40), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "30 + 1 =", fill=(0, 0, 0))
    d.rectangle([135, 10, 145, 20], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    trimmed_bytes = preprocess_image(buf.getvalue())
    im_trimmed = Image.open(io.BytesIO(trimmed_bytes))
    assert im_trimmed.size[0] < 150
    assert im_trimmed.size[0] <= 80

    # Image without right icon should remain untrimmed
    img_no_icon = Image.new("RGB", (150, 40), (255, 255, 255))
    d2 = ImageDraw.Draw(img_no_icon)
    d2.text((10, 10), "30 + 1 =", fill=(0, 0, 0))
    buf2 = io.BytesIO()
    img_no_icon.save(buf2, format="PNG")

    untrimmed_bytes = preprocess_image(buf2.getvalue())
    im_untrimmed = Image.open(io.BytesIO(untrimmed_bytes))
    assert im_untrimmed.size[0] == 150


@pytest.mark.asyncio
async def test_solve_image_captcha_retry_contrast(monkeypatch):
    b64 = _make_test_image_b64("30+1=?")
    calls = []

    class MockOcr:
        def classification(self, b):
            calls.append(b)
            if len(calls) == 1:
                # First pass: returns noisy non-matching text containing arithmetic hint
                return "30 + ? 1"
            # Second pass (enhanced contrast): returns clean math text
            return "30 + 1 = ?"

    monkeypatch.setattr("ocr.solve.get_ocr", lambda: MockOcr())
    res = await solve_image_captcha(b64, math_mode="auto")
    assert len(calls) == 2
    assert res["is_math"] is True
    assert res["value"] == 31
    assert res["result"] == "31"
    assert res["raw_text"] == "30 + 1 = ?"


@pytest.mark.asyncio
async def test_solve_image_captcha_corrupt_image_force_math():
    raw_b64 = base64.b64encode(b"not-a-valid-image").decode("utf-8")
    res = await solve_image_captcha(raw_b64, math_mode="force_math")
    assert res["is_math"] is False
    assert res["value"] is None
    assert res["result"] == ""
    assert "error" in res


def test_preprocess_image_dark_mode_neon_text():
    w, h = 200, 50
    im = Image.new("RGB", (w, h), (11, 14, 20))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), "27 + 27 = ?", fill=(57, 255, 20), font=font)
    draw.line([(5, 45), (195, 5)], fill=(0, 200, 150), width=1)

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    processed = preprocess_image(buf.getvalue())
    prep_im = Image.open(io.BytesIO(processed))
    assert prep_im.getpixel((0, 0)) == 255
    assert prep_im.getpixel((prep_im.width - 1, 0)) == 255
    assert prep_im.getpixel((0, prep_im.height - 1)) == 255


@pytest.mark.asyncio
async def test_solve_dark_mode_neon_text_with_noise_line():
    w, h = 200, 50
    im = Image.new("RGB", (w, h), (11, 14, 20))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), "27 + 27 = ?", fill=(57, 255, 20), font=font)
    draw.line([(5, 45), (195, 5)], fill=(0, 200, 150), width=1)

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    res = await solve_image_captcha(b64, math_mode="auto")
    assert res["is_math"] is True
    assert res["value"] == 54
    assert res["result"] == "54"
