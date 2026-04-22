# enhanced_scanner.py - Production Scanner Module for NOODH POS
# Supports: OpenCV + pyzbar (full mode), pyzbar-only, and manual/upload fallback
from __future__ import annotations

import io
import logging
import re
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Optional dependency probes (import-time, no Streamlit side-effects) ──────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("OpenCV not installed. Enhanced image preprocessing disabled.")

try:
    from pyzbar import pyzbar as _pyzbar
    _PYZBAR_AVAILABLE = True
except ImportError:
    _PYZBAR_AVAILABLE = False
    logger.warning("pyzbar not installed. Image-based barcode decoding disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────
_BARCODE_PATTERNS: list[tuple[str, str, float]] = [
    (r"^\d{13}$",                        "EAN-13",   0.95),
    (r"^\d{12}$",                        "UPC-A",    0.95),
    (r"^\d{8}$",                         "EAN-8",    0.95),
    (r"^\d{6,18}$",                      "Numeric",  0.80),
    (r"^[A-Z0-9\-\.\$\/\+\%\s]{1,43}$", "CODE-39",  0.75),
    (r"^[\x20-\x7E]{1,48}$",            "CODE-128", 0.70),
]
_MIN_CODE_LEN = 4
_CACHE_MAX_SIZE = 128


# ── Data helpers ──────────────────────────────────────────────────────────────

def _make_detection(
    data: str,
    code_type: str,
    confidence: float,
    method: str,
    polygon: Any = None,
    rect: Any = None,
) -> dict[str, Any]:
    return {
        "data": data,
        "type": code_type,
        "confidence": round(confidence, 4),
        "method": method,
        "polygon": polygon,
        "rect": rect,
    }


# ── Core scanner class ────────────────────────────────────────────────────────

class EnhancedBarcodeScanner:
    """
    Production-grade barcode scanner with graceful degradation.

    Modes (auto-selected at runtime):
      1. Full   – OpenCV preprocessing + pyzbar decoding
      2. Basic  – pyzbar decoding only
      3. Manual – Pattern validation only (no image decoding)
    """

    def __init__(
        self,
        debounce_time: float = 1.5,
        max_history: int = 50,
        confidence_threshold: float = 0.70,
    ) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")

        self.debounce_time = debounce_time
        self.confidence_threshold = confidence_threshold

        self._scan_history: deque[dict] = deque(maxlen=max_history)
        self._last_scan_time: dict[str, float] = {}
        self._result_cache: dict[str, dict] = {}  # LRU-like via insertion order
        self._processing = False

    # ── Capability query ──────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        if _CV2_AVAILABLE and _PYZBAR_AVAILABLE:
            return "full"
        if _PYZBAR_AVAILABLE:
            return "basic"
        return "manual"

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "opencv": _CV2_AVAILABLE,
            "pyzbar": _PYZBAR_AVAILABLE,
            "image_scan": _PYZBAR_AVAILABLE,
            "enhanced_preprocessing": _CV2_AVAILABLE and _PYZBAR_AVAILABLE,
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_barcode_pattern(
        self, code_data: str
    ) -> tuple[bool, float, str]:
        """
        Returns (is_valid, confidence, barcode_type).
        Validates purely by pattern; no image required.
        """
        code = code_data.strip() if code_data else ""
        if len(code) < _MIN_CODE_LEN:
            return False, 0.0, "Unknown"

        for pattern, btype, base_conf in _BARCODE_PATTERNS:
            if re.match(pattern, code):
                # Slight boost for pure-numeric codes (less ambiguous)
                conf = base_conf + (0.04 if code.isdigit() else 0.0)
                conf = min(conf, 1.0)
                return conf >= self.confidence_threshold, conf, btype

        # Generic fallback
        conf = 0.55
        return conf >= self.confidence_threshold, conf, "Generic"

    # ── Image scanning ────────────────────────────────────────────────────────

    def _preprocess_variants(self, image: np.ndarray) -> list[np.ndarray]:
        """Return a list of preprocessed grayscale variants for better decode rate."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if image.ndim == 3
            else image.copy()
        )
        variants: list[np.ndarray] = [gray]

        ops = [
            lambda g: cv2.GaussianBlur(g, (3, 3), 0),
            lambda g: cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            ),
            lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda g: cv2.equalizeHist(g),
        ]
        for op in ops:
            try:
                variants.append(op(gray))
            except Exception as exc:
                logger.debug("Preprocessing op failed: %s", exc)

        return variants

    def _decode_with_pyzbar(self, images: list[np.ndarray]) -> list[dict]:
        """Run pyzbar over a list of images; return deduplicated detections."""
        seen: dict[str, dict] = {}
        for idx, img in enumerate(images):
            try:
                codes = _pyzbar.decode(img)
            except Exception as exc:
                logger.debug("pyzbar decode error on variant %d: %s", idx, exc)
                continue

            for code in codes:
                try:
                    raw = code.data.decode("utf-8").strip()
                except (UnicodeDecodeError, AttributeError):
                    continue

                is_valid, conf, btype = self.validate_barcode_pattern(raw)
                if not is_valid:
                    continue

                method = "pyzbar_cv2" if _CV2_AVAILABLE else "pyzbar"
                det = _make_detection(
                    raw, btype, conf, f"{method}_v{idx}",
                    getattr(code, "polygon", None),
                    getattr(code, "rect", None),
                )
                if raw not in seen or det["confidence"] > seen[raw]["confidence"]:
                    seen[raw] = det

        return sorted(seen.values(), key=lambda d: d["confidence"], reverse=True)

    def scan_image(self, image: Image.Image) -> list[dict]:
        """
        Scan a PIL Image for barcodes.
        Returns a list of detection dicts sorted by confidence (descending).
        """
        if not _PYZBAR_AVAILABLE:
            logger.warning("scan_image called but pyzbar is not available.")
            return []

        arr = np.array(image.convert("RGB"))

        if _CV2_AVAILABLE:
            variants = self._preprocess_variants(arr)
        else:
            gray = np.mean(arr, axis=2).astype(np.uint8) if arr.ndim == 3 else arr
            variants = [gray]

        return self._decode_with_pyzbar(variants)

    def scan_from_bytes(self, image_bytes: bytes) -> list[dict]:
        """Decode image bytes and scan for barcodes."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.error("Failed to open image bytes: %s", exc)
            raise ValueError(f"Cannot open image: {exc}") from exc
        return self.scan_image(image)

    # ── Manual validation ─────────────────────────────────────────────────────

    def validate_manual_code(self, code: str) -> dict | None:
        """
        Validate a manually entered barcode string.
        Returns a detection dict or None if invalid.
        """
        code = code.strip()
        is_valid, conf, btype = self.validate_barcode_pattern(code)
        if not is_valid:
            return None
        return _make_detection(code, btype, conf, "manual_entry")

    # ── Debouncing ────────────────────────────────────────────────────────────

    def apply_debounce(self, detections: list[dict]) -> list[dict]:
        """Filter detections that appeared too recently."""
        now = time.monotonic()
        passed: list[dict] = []
        for det in detections:
            code = det["data"]
            if now - self._last_scan_time.get(code, 0.0) >= self.debounce_time:
                self._last_scan_time[code] = now
                passed.append(det)
                self._record_history(det)
        return passed

    def _record_history(self, det: dict) -> None:
        self._scan_history.append({
            "code": det["data"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "confidence": det["confidence"],
            "type": det["type"],
        })

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        history = list(self._scan_history)
        if not history:
            return {"total_scans": 0, "capabilities": self.capabilities, "mode": self.mode}
        return {
            "total_scans": len(history),
            "unique_codes": len({s["code"] for s in history}),
            "average_confidence": round(
                sum(s["confidence"] for s in history) / len(history), 3
            ),
            "recent_scans": history[-5:],
            "capabilities": self.capabilities,
            "mode": self.mode,
        }

    def reset_history(self) -> None:
        self._scan_history.clear()
        self._last_scan_time.clear()


# ── Session-state helpers ─────────────────────────────────────────────────────

def _get_scanner(key: str) -> EnhancedBarcodeScanner:
    if key not in st.session_state:
        st.session_state[key] = EnhancedBarcodeScanner()
    return st.session_state[key]


def _empty_result() -> dict:
    return {"code": None, "source": None, "confidence": 0.0, "type": None, "details": None}


# ── UI components ─────────────────────────────────────────────────────────────

def _render_status_badge(scanner: EnhancedBarcodeScanner) -> None:
    mode_labels = {
        "full":   ("🟢", "Full Mode", "OpenCV + pyzbar active"),
        "basic":  ("🟡", "Basic Mode", "pyzbar active, OpenCV missing"),
        "manual": ("🔴", "Manual Only", "pyzbar & OpenCV unavailable"),
    }
    icon, label, detail = mode_labels[scanner.mode]
    st.caption(f"{icon} **{label}** — {detail}")
    if scanner.mode != "full":
        missing = [k for k, v in scanner.capabilities.items() if not v]
        st.info(
            f"Missing: `{'`, `'.join(missing)}`. "
            "Install via `pip install opencv-python pyzbar` for full scanning support."
        )


def _render_detection(det: dict, source: str) -> dict:
    conf = det["confidence"]
    emoji = "🟢" if conf > 0.85 else "🟡" if conf > 0.65 else "🔴"
    st.success(f"{emoji} **Detected:** `{det['data']}`")
    st.caption(f"Type: {det['type']} · Confidence: {conf:.0%} · Source: {source}")
    return {
        "code": det["data"],
        "source": source,
        "confidence": conf,
        "type": det["type"],
        "details": det,
    }


# ── Main interface ────────────────────────────────────────────────────────────

def create_enhanced_scanner_interface(
    key_prefix: str = "scanner",
    show_advanced: bool = True,
    debounce_time: float = 1.5,
    confidence_threshold: float = 0.70,
) -> dict:
    """
    Render the full scanner interface and return a result dict:
      { code, source, confidence, type, details }
    All keys are None / 0.0 if no barcode was detected this render cycle.
    """
    scanner_key = f"_scanner_{key_prefix}"
    scanner: EnhancedBarcodeScanner = _get_scanner(scanner_key)
    # Apply any externally changed settings
    scanner.debounce_time = debounce_time
    scanner.confidence_threshold = confidence_threshold

    st.markdown("### 📱 Product Scanner")

    if show_advanced:
        _render_status_badge(scanner)

    # ── Guide banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            border:2px dashed #22c55e;border-radius:10px;padding:16px;
            text-align:center;background:rgba(34,197,94,.06);margin:8px 0 16px;
            font-size:.85rem;
        ">
            <strong style="color:#22c55e;">Scanning Options</strong>
            &nbsp;·&nbsp; 📷 Camera &nbsp;·&nbsp; 🖼️ Upload &nbsp;·&nbsp; ⌨️ Manual
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = _empty_result()
    tab_camera, tab_upload, tab_manual = st.tabs(["📷 Camera", "🖼️ Upload", "⌨️ Manual Entry"])

    # ── Camera tab ────────────────────────────────────────────────────────────
    with tab_camera:
        if not _PYZBAR_AVAILABLE:
            st.warning("Image scanning unavailable. Use **Manual Entry** tab.")
        else:
            show_preview = st.checkbox("Show preview", value=True, key=f"{key_prefix}_cam_preview")

            camera_input = st.camera_input(
                "Capture barcode",
                key=f"{key_prefix}_camera",
                help="Position the barcode clearly in frame",
            )

            if camera_input is not None:
                if show_preview:
                    st.image(camera_input, caption="Captured image", width=300)

                with st.spinner("Scanning…"):
                    try:
                        detections = scanner.scan_from_bytes(camera_input.getvalue())
                        detections = scanner.apply_debounce(detections)
                    except ValueError as exc:
                        st.error(str(exc))
                        detections = []

                if detections:
                    result = _render_detection(detections[0], "camera")
                    if len(detections) > 1:
                        with st.expander(f"ℹ️ {len(detections)} barcodes found"):
                            for det in detections[1:]:
                                st.text(f"{det['data']} ({det['type']}, {det['confidence']:.0%})")
                else:
                    st.warning("No barcode detected. Ensure good lighting and a steady hand.")

    # ── Upload tab ────────────────────────────────────────────────────────────
    with tab_upload:
        if not _PYZBAR_AVAILABLE:
            st.warning("Image scanning unavailable. Use **Manual Entry** tab.")
        else:
            uploaded = st.file_uploader(
                "Upload barcode image",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                key=f"{key_prefix}_upload",
            )

            if uploaded is not None:
                raw_bytes = uploaded.read()
                col_img, col_result = st.columns([1, 2])

                with col_img:
                    st.image(raw_bytes, width=200)

                with col_result:
                    with st.spinner("Scanning…"):
                        try:
                            detections = scanner.scan_from_bytes(raw_bytes)
                            detections = scanner.apply_debounce(detections)
                        except ValueError as exc:
                            st.error(str(exc))
                            detections = []

                    if detections:
                        result = _render_detection(detections[0], "upload")
                        if len(detections) > 1:
                            with st.expander(f"ℹ️ {len(detections)} barcodes found"):
                                for i, det in enumerate(detections[1:], start=2):
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.text(f"{i}. {det['data']} ({det['type']})")
                                    with col_b:
                                        if st.button("Use", key=f"{key_prefix}_use_{i}"):
                                            result = _render_detection(det, "upload")
                    else:
                        st.warning("No barcode detected. Try a clearer image or manual entry.")

    # ── Manual entry tab ──────────────────────────────────────────────────────
    with tab_manual:
        col_input, col_btn = st.columns([4, 1])

        with col_input:
            manual_code = st.text_input(
                "Enter barcode",
                placeholder="Type, paste, or use a keyboard wedge scanner…",
                key=f"{key_prefix}_manual",
            )

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.button("✓ Submit", key=f"{key_prefix}_manual_submit")

        # Only fire on explicit submit to avoid per-keystroke validation
        if submit:
            code = manual_code.strip()
            if not code:
                st.warning("Please enter a barcode first.")
            else:
                validation = scanner.validate_manual_code(code)
                if validation:
                    scanner.apply_debounce([validation])
                    result = _render_detection(validation, "manual")
                else:
                    st.error(
                        f"Invalid barcode: `{code}` — must be ≥{_MIN_CODE_LEN} characters "
                        "and match a recognised format."
                    )

    # ── Recent history ────────────────────────────────────────────────────────
    if show_advanced:
        stats = scanner.get_statistics()
        recent = stats.get("recent_scans", [])
        if recent:
            with st.expander("📊 Recent Scans"):
                for scan in reversed(recent):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    c1.code(scan["code"])
                    c2.caption(scan["type"])
                    c3.caption(f"{scan['confidence']:.0%}")

    # ── Result summary ────────────────────────────────────────────────────────
    if result["code"]:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Code** `{result['code']}`")
        c2.markdown(f"**Type** {result['type'] or '—'}")
        conf = result["confidence"]
        badge = "🟢" if conf > 0.85 else "🟡" if conf > 0.65 else "🔴"
        c3.markdown(f"**Quality** {badge} {conf:.0%}")

    return result


# ── Sidebar settings panel ────────────────────────────────────────────────────

def create_scanner_settings_panel(key_prefix: str = "scanner_settings") -> dict:
    """
    Render sidebar scanner settings. Returns the settings dict.
    Pass the returned values into create_enhanced_scanner_interface().
    """
    with st.sidebar:
        st.markdown("### ⚙️ Scanner Settings")

        with st.expander("🎯 Scan Quality", expanded=False):
            debounce_time = st.slider(
                "Scan cooldown (s)",
                min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                help="Minimum seconds before the same code is accepted again.",
                key=f"{key_prefix}_debounce",
            )
            confidence_threshold = st.slider(
                "Min confidence",
                min_value=0.10, max_value=1.0, value=0.70, step=0.05,
                help="Detections below this threshold are discarded.",
                key=f"{key_prefix}_confidence",
            )

        with st.expander("🖼️ Interface", expanded=False):
            show_advanced = st.checkbox(
                "Show advanced info", value=True, key=f"{key_prefix}_advanced"
            )

    return {
        "debounce_time": debounce_time,
        "confidence_threshold": confidence_threshold,
        "show_advanced": show_advanced,
    }


# ── Standalone demo ───────────────────────────────────────────────────────────

def _demo() -> None:
    """Entry-point for running the scanner as a standalone Streamlit page."""
    st.set_page_config(page_title="NOODH Scanner", page_icon="📱", layout="wide")
    st.title("📱 NOODH Barcode Scanner")

    settings = create_scanner_settings_panel()

    result = create_enhanced_scanner_interface(
        key_prefix="demo",
        show_advanced=settings["show_advanced"],
        debounce_time=settings["debounce_time"],
        confidence_threshold=settings["confidence_threshold"],
    )

    if result["code"]:
        st.divider()
        st.markdown("### 🎯 Actions")
        c1, c2 = st.columns(2)
        with c1:
            st.button("🔍 Lookup Product", key="demo_lookup")
        with c2:
            if st.button("🗑️ Clear History", key="demo_clear"):
                scanner = _get_scanner("_scanner_demo")
                scanner.reset_history()
                st.rerun()


if __name__ == "__main__":
    _demo()
