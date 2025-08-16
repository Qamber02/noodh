# scanner.py - Optimized Scanner Module for NOODH POS
import cv2
import numpy as np
import streamlit as st
from pyzbar import pyzbar
from datetime import datetime, timedelta
import threading
import time
from collections import deque

class OptimizedBarcodeScanner:
    """
    Optimized barcode scanner with better performance and reliability
    """
    
    def __init__(self, debounce_time=2.0, max_results=5):
        self.debounce_time = debounce_time
        self.max_results = max_results
        self.last_results = deque(maxlen=max_results)
        self.last_scan_time = {}
        self.processing = False
        
    def preprocess_image(self, image):
        """Enhanced image preprocessing for better barcode detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply multiple preprocessing techniques
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. Adaptive threshold for better contrast
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        return [gray, blurred, adaptive, morphed]
    
    def scan_multiple_processed(self, processed_images):
        """Scan multiple processed versions of the same image"""
        all_codes = []
        
        for img in processed_images:
            try:
                # Try different pyzbar configurations
                codes = pyzbar.decode(img)
                for code in codes:
                    try:
                        decoded_data = code.data.decode('utf-8').strip()
                        if decoded_data and decoded_data not in all_codes:
                            all_codes.append(decoded_data)
                    except UnicodeDecodeError:
                        continue
            except Exception:
                continue
                
        return all_codes
    
    def scan_image(self, image):
        """Main scanning function with enhanced detection"""
        if self.processing:
            return []
            
        self.processing = True
        
        try:
            # Preprocess image in multiple ways
            processed_images = self.preprocess_image(image)
            
            # Scan all processed versions
            codes = self.scan_multiple_processed(processed_images)
            
            # Filter and debounce results
            current_time = datetime.now()
            valid_codes = []
            
            for code in codes:
                last_time = self.last_scan_time.get(code, datetime.min)
                if (current_time - last_time).total_seconds() >= self.debounce_time:
                    self.last_scan_time[code] = current_time
                    valid_codes.append(code)
                    
            return valid_codes
            
        finally:
            self.processing = False
    
    def scan_from_bytes(self, image_bytes):
        """Scan barcode from image bytes"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return []
            return self.scan_image(image)
        except Exception:
            return []

def create_scanner_interface(key_prefix="scanner", auto_scan=False):
    """
    Create an optimized scanner interface with multiple input methods
    Returns: {"code": str|None, "source": str|None, "confidence": float}
    """
    
    # Initialize scanner in session state
    scanner_key = f"{key_prefix}_scanner"
    if scanner_key not in st.session_state:
        st.session_state[scanner_key] = OptimizedBarcodeScanner()
    
    scanner = st.session_state[scanner_key]
    
    st.markdown("### 📱 Product Scanner")
    
    # Create tabs for different scanning methods
    scan_tabs = st.tabs(["📷 Camera Scan", "🖼️ Upload Image", "⌨️ Manual Entry"])
    
    result = {"code": None, "source": None, "confidence": 0.0}
    
    # Tab 1: Camera Scan
    with scan_tabs[0]:
        st.markdown("#### Camera Scanner")
        
        # Camera selection
        col1, col2 = st.columns([1, 1])
        with col1:
            camera_facing = st.radio(
                "Camera", 
                ["Back Camera", "Front Camera"], 
                horizontal=True,
                key=f"{key_prefix}_camera_choice"
            )
        
        with col2:
            scan_mode = st.radio(
                "Mode",
                ["Single Scan", "Continuous"],
                horizontal=True,
                key=f"{key_prefix}_scan_mode"
            )
        
        # Camera feed placeholder
        camera_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # Camera capture using native streamlit camera
        camera_input = st.camera_input(
            "Take a picture of the barcode",
            key=f"{key_prefix}_camera",
            help="Align the barcode within the frame and take a clear picture"
        )
        
        if camera_input is not None:
            # Process the captured image
            with st.spinner("Scanning barcode..."):
                codes = scanner.scan_from_bytes(camera_input.read())
                
                if codes:
                    result["code"] = codes[0]
                    result["source"] = "camera"
                    result["confidence"] = 0.9
                    result_placeholder.success(f"✅ Scanned: **{codes[0]}**")
                    
                    if len(codes) > 1:
                        st.info(f"Multiple codes found: {', '.join(codes)}")
                        selected_code = st.selectbox(
                            "Select code to use:",
                            codes,
                            key=f"{key_prefix}_multi_select"
                        )
                        if selected_code:
                            result["code"] = selected_code
                else:
                    result_placeholder.warning("❌ No barcode detected. Try adjusting lighting or angle.")
    
    # Tab 2: Upload Image
    with scan_tabs[1]:
        st.markdown("#### Upload Image")
        uploaded_file = st.file_uploader(
            "Upload an image containing a barcode",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key=f"{key_prefix}_upload",
            help="Supported formats: PNG, JPG, JPEG, WebP, BMP"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            # Scan the uploaded image
            with st.spinner("Processing image..."):
                codes = scanner.scan_from_bytes(uploaded_file.read())
                
                if codes:
                    result["code"] = codes[0]
                    result["source"] = "upload"
                    result["confidence"] = 0.95
                    st.success(f"✅ Detected: **{codes[0]}**")
                    
                    if len(codes) > 1:
                        st.info("Multiple barcodes found:")
                        selected_code = st.selectbox(
                            "Select barcode:",
                            codes,
                            key=f"{key_prefix}_upload_select"
                        )
                        if selected_code:
                            result["code"] = selected_code
                else:
                    st.error("❌ No barcode found in the uploaded image.")
                    st.info("💡 Tips: Ensure the barcode is clear, well-lit, and not blurry.")
    
    # Tab 3: Manual Entry
    with scan_tabs[2]:
        st.markdown("#### Manual Entry")
        manual_code = st.text_input(
            "Enter barcode manually",
            placeholder="Type or paste barcode here...",
            key=f"{key_prefix}_manual",
            help="Use this if scanning is not working or for quick entry"
        )
        
        if manual_code.strip():
            # Validate the manual code (basic validation)
            cleaned_code = manual_code.strip()
            if len(cleaned_code) >= 6:  # Minimum reasonable barcode length
                result["code"] = cleaned_code
                result["source"] = "manual"
                result["confidence"] = 1.0
                st.success(f"✅ Manual entry: **{cleaned_code}**")
            else:
                st.warning("⚠️ Barcode seems too short. Please verify.")
    
    # Display final result
    if result["code"]:
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Scanned Code:** `{result['code']}`")
        with col2:
            st.markdown(f"**Source:** {result['source'].title()}")
        with col3:
            confidence_color = "green" if result["confidence"] > 0.8 else "orange" if result["confidence"] > 0.5 else "red"
            st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{result['confidence']:.0%}</span>", unsafe_allow_html=True)
    
    return result

def create_advanced_scanner_settings():
    """Create advanced scanner settings in sidebar"""
    with st.sidebar.expander("🔧 Scanner Settings"):
        st.markdown("**Advanced Scanner Options**")
        
        debounce_time = st.slider(
            "Debounce Time (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Minimum time between scanning the same code"
        )
        
        auto_enhance = st.checkbox(
            "Auto Image Enhancement",
            value=True,
            help="Automatically enhance images for better scanning"
        )
        
        multiple_attempts = st.checkbox(
            "Multiple Detection Attempts",
            value=True,
            help="Try multiple image processing techniques"
        )
        
        return {
            "debounce_time": debounce_time,
            "auto_enhance": auto_enhance,
            "multiple_attempts": multiple_attempts
        }

# Test function for the scanner
def test_scanner():
    """Test function to verify scanner functionality"""
    st.header("🧪 Scanner Test")
    st.write("Use this page to test the scanner functionality")
    
    settings = create_advanced_scanner_settings()
    scan_result = create_scanner_interface("test_scanner")
    
    if scan_result["code"]:
        st.json(scan_result)
        st.balloons()

if __name__ == "__main__":
    # Run test if executed directly
    st.set_page_config(page_title="Scanner Test", layout="wide")
    test_scanner()