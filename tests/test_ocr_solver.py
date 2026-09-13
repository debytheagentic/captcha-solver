"""Comprehensive test suite for OCR and math captcha solver."""
import base64
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from ocr.solve import (
    decode_image,
    evaluate_math_expression,
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
