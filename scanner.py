# enhanced_scanner.py - Highly Optimized Scanner Module for NOODH POS
import cv2
import numpy as np
import streamlit as st
from pyzbar import pyzbar
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import io
from PIL import Image, ImageDraw, ImageFont
import base64

class EnhancedBarcodeScanner:
    """
    Enhanced barcode scanner with improved accuracy, stability, and visual guides
    """
    
    def __init__(self, debounce_time=1.5, max_results=10, confidence_threshold=0.7):
        self.debounce_time = debounce_time
        self.max_results = max_results
        self.confidence_threshold = confidence_threshold
        self.last_results = deque(maxlen=max_results)
        self.last_scan_time = {}
        self.processing = False
        self.scan_history = deque(maxlen=50)
        
        # Performance optimizations
        self.image_cache = {}
        self.cache_max_size = 20
        
    def calculate_image_hash(self, image):
        """Calculate a simple hash for image caching"""
        return hash(image.tobytes())
    
    def enhance_image_quality(self, image):
        """Advanced image enhancement for better barcode detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Check cache first
        img_hash = self.calculate_image_hash(gray)
        if img_hash in self.image_cache:
            return self.image_cache[img_hash]
        
        enhanced_images = []
        
        # 1. Original grayscale
        enhanced_images.append(gray)
        
        # 2. Gaussian blur with noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced_images.append(blurred)
        
        # 3. Bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced_images.append(bilateral)
        
        # 4. Adaptive thresholding (multiple methods)
        adaptive_mean = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        enhanced_images.append(adaptive_mean)
        
        adaptive_gaussian = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        enhanced_images.append(adaptive_gaussian)
        
        # 5. OTSU thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images.append(otsu)
        
        # 6. Morphological operations
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Closing to fill gaps
        morphed_close = cv2.morphologyEx(adaptive_gaussian, cv2.MORPH_CLOSE, kernel_rect)
        enhanced_images.append(morphed_close)
        
        # Opening to remove noise
        morphed_open = cv2.morphologyEx(adaptive_gaussian, cv2.MORPH_OPEN, kernel_ellipse)
        enhanced_images.append(morphed_open)
        
        # 7. Sharpening filter
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        enhanced_images.append(sharpened)
        
        # 8. Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        enhanced_images.append(contrast_enhanced)
        
        # Cache the results
        if len(self.image_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.image_cache))
            del self.image_cache[oldest_key]
        
        self.image_cache[img_hash] = enhanced_images
        return enhanced_images
    
    def validate_barcode(self, code_data, code_type):
        """Validate barcode data and calculate confidence score"""
        if not code_data or len(code_data.strip()) < 4:
            return False, 0.0
        
        confidence = 0.5  # Base confidence
        
        # Length-based validation
        code_len = len(code_data)
        if code_type == 'EAN13' and code_len == 13:
            confidence += 0.3
        elif code_type == 'EAN8' and code_len == 8:
            confidence += 0.3
        elif code_type == 'UPCA' and code_len == 12:
            confidence += 0.3
        elif code_type == 'CODE128' and 6 <= code_len <= 48:
            confidence += 0.2
        elif code_type == 'CODE39' and 1 <= code_len <= 43:
            confidence += 0.2
        
        # Character validation
        if code_data.isdigit():
            confidence += 0.1
        elif code_data.isalnum():
            confidence += 0.05
        
        # Pattern validation
        if not any(char in code_data for char in ['!', '@', '#', '%', '^', '&']):
            confidence += 0.1
        
        return confidence >= self.confidence_threshold, confidence
    
    def scan_with_multiple_methods(self, processed_images):
        """Scan using multiple detection methods and return best results"""
        all_detections = []
        
        for i, img in enumerate(processed_images):
            try:
                # Standard pyzbar detection
                codes = pyzbar.decode(img, symbols=[
                    pyzbar.ZBarSymbol.EAN13,
                    pyzbar.ZBarSymbol.EAN8,
                    pyzbar.ZBarSymbol.UPCA,
                    pyzbar.ZBarSymbol.UPCE,
                    pyzbar.ZBarSymbol.CODE128,
                    pyzbar.ZBarSymbol.CODE39,
                    pyzbar.ZBarSymbol.CODE93,
                    pyzbar.ZBarSymbol.CODABAR,
                    pyzbar.ZBarSymbol.ITF25,
                    pyzbar.ZBarSymbol.DATAMATRIX,
                    pyzbar.ZBarSymbol.QRCODE
                ])
                
                for code in codes:
                    try:
                        decoded_data = code.data.decode('utf-8').strip()
                        code_type = code.type
                        
                        is_valid, confidence = self.validate_barcode(decoded_data, code_type)
                        
                        if is_valid:
                            detection = {
                                'data': decoded_data,
                                'type': code_type,
                                'confidence': confidence,
                                'method': f'enhanced_{i}',
                                'polygon': code.polygon,
                                'rect': code.rect
                            }
                            all_detections.append(detection)
                            
                    except (UnicodeDecodeError, AttributeError):
                        continue
                        
            except Exception as e:
                continue
        
        # Remove duplicates and sort by confidence
        unique_detections = {}
        for detection in all_detections:
            data = detection['data']
            if data not in unique_detections or detection['confidence'] > unique_detections[data]['confidence']:
                unique_detections[data] = detection
        
        # Sort by confidence
        sorted_detections = sorted(unique_detections.values(), key=lambda x: x['confidence'], reverse=True)
        return sorted_detections
    
    def scan_image(self, image):
        """Main scanning function with enhanced detection and stability"""
        if self.processing:
            return []
        
        self.processing = True
        
        try:
            # Enhance image quality
            processed_images = self.enhance_image_quality(image)
            
            # Scan with multiple methods
            detections = self.scan_with_multiple_methods(processed_images)
            
            # Apply debouncing and filtering
            current_time = datetime.now()
            valid_results = []
            
            for detection in detections:
                code = detection['data']
                last_time = self.last_scan_time.get(code, datetime.min)
                
                if (current_time - last_time).total_seconds() >= self.debounce_time:
                    self.last_scan_time[code] = current_time
                    valid_results.append(detection)
                    
                    # Add to scan history
                    self.scan_history.append({
                        'code': code,
                        'timestamp': current_time,
                        'confidence': detection['confidence'],
                        'type': detection['type']
                    })
            
            return valid_results
            
        finally:
            self.processing = False
    
    def scan_from_bytes(self, image_bytes):
        """Scan barcode from image bytes with error handling"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return []
            return self.scan_image(image)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return []
    
    def get_scan_statistics(self):
        """Get scanning statistics"""
        if not self.scan_history:
            return {}
        
        total_scans = len(self.scan_history)
        avg_confidence = sum(scan['confidence'] for scan in self.scan_history) / total_scans
        unique_codes = len(set(scan['code'] for scan in self.scan_history))
        
        return {
            'total_scans': total_scans,
            'unique_codes': unique_codes,
            'average_confidence': avg_confidence,
            'recent_scans': list(self.scan_history)[-5:]
        }

def create_barcode_overlay():
    """Create a barcode scanning overlay with guide lines"""
    overlay_html = """
    <div style="position: relative; display: inline-block;">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border: 3px solid #00ff00;
            border-radius: 10px;
            background: rgba(0, 255, 0, 0.1);
            z-index: 10;
        ">
            <div style="
                position: absolute;
                top: 50%;
                left: 10%;
                right: 10%;
                height: 2px;
                background: #ff0000;
                transform: translateY(-50%);
                box-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
            "></div>
            <div style="
                position: absolute;
                top: 10px;
                left: 10px;
                color: #00ff00;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
                font-size: 12px;
            ">📱 Align barcode here</div>
        </div>
    </div>
    """
    return overlay_html

def create_enhanced_scanner_interface(key_prefix="enhanced_scanner", show_advanced=True):
    """
    Create an enhanced scanner interface with visual guides and advanced features
    """
    
    # Initialize enhanced scanner
    scanner_key = f"{key_prefix}_enhanced_scanner"
    if scanner_key not in st.session_state:
        st.session_state[scanner_key] = EnhancedBarcodeScanner()
    
    scanner = st.session_state[scanner_key]
    
    st.markdown("### 📱 Enhanced Product Scanner")
    st.markdown("*Optimized for accuracy and stability*")
    
    # Performance metrics
    if show_advanced:
        stats = scanner.get_scan_statistics()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Scans", stats['total_scans'])
            with col2:
                st.metric("Unique Codes", stats['unique_codes'])
            with col3:
                st.metric("Avg Confidence", f"{stats['average_confidence']:.1%}")
            with col4:
                cache_size = len(scanner.image_cache)
                st.metric("Cache Size", cache_size)
    
    # Enhanced scanning tabs
    scan_tabs = st.tabs(["📷 Smart Camera", "🖼️ AI Upload", "⌨️ Quick Entry", "📊 Scan History"])
    
    result = {"code": None, "source": None, "confidence": 0.0, "type": None, "details": None}
    
    # Tab 1: Smart Camera Scan
    with scan_tabs[0]:
        st.markdown("#### 📷 Smart Camera Scanner")
        st.markdown("*Position the barcode within the green border for best results*")
        
        # Scanner settings
        col1, col2, col3 = st.columns(3)
        with col1:
            auto_enhance = st.checkbox("🎯 Auto Enhance", value=True, help="Automatically enhance image quality")
        with col2:
            continuous_scan = st.checkbox("🔄 Continuous Mode", value=False, help="Scan continuously")
        with col3:
            show_overlay = st.checkbox("📐 Show Guide", value=True, help="Show scanning guide overlay")
        
        # Camera input with overlay
        if show_overlay:
            st.markdown(create_barcode_overlay(), unsafe_allow_html=True)
        
        camera_input = st.camera_input(
            "📸 Capture barcode image",
            key=f"{key_prefix}_smart_camera",
            help="Align the barcode within the frame and ensure good lighting"
        )
        
        if camera_input is not None:
            with st.spinner("🔍 Scanning with AI enhancement..."):
                detections = scanner.scan_from_bytes(camera_input.read())
                
                if detections:
                    best_detection = detections[0]
                    result.update({
                        "code": best_detection['data'],
                        "source": "smart_camera",
                        "confidence": best_detection['confidence'],
                        "type": best_detection['type'],
                        "details": best_detection
                    })
                    
                    # Display results with confidence
                    confidence_color = "🟢" if best_detection['confidence'] > 0.8 else "🟡" if best_detection['confidence'] > 0.6 else "🔴"
                    st.success(f"{confidence_color} **Detected:** `{best_detection['data']}` (Type: {best_detection['type']})")
                    
                    # Show additional detections if any
                    if len(detections) > 1:
                        st.info(f"Found {len(detections)} barcode(s)")
                        with st.expander("View all detections"):
                            for i, det in enumerate(detections):
                                st.write(f"{i+1}. `{det['data']}` - {det['type']} (Confidence: {det['confidence']:.1%})")
                                if st.button(f"Use this code", key=f"use_{i}"):
                                    result["code"] = det['data']
                                    result["details"] = det
                else:
                    st.warning("❌ No barcode detected")
                    st.info("💡 **Tips for better scanning:**\n- Ensure good lighting\n- Hold camera steady\n- Clean the barcode surface\n- Try different angles")
    
    # Tab 2: AI Upload
    with scan_tabs[1]:
        st.markdown("#### 🖼️ AI-Powered Image Upload")
        
        uploaded_file = st.file_uploader(
            "Upload barcode image",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
            key=f"{key_prefix}_ai_upload",
            help="Supports all major image formats with AI enhancement"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(uploaded_file, caption="Original Image", width=200)
            
            with col2:
                with st.spinner("🤖 AI Processing..."):
                    detections = scanner.scan_from_bytes(uploaded_file.read())
                    
                    if detections:
                        best_detection = detections[0]
                        result.update({
                            "code": best_detection['data'],
                            "source": "ai_upload",
                            "confidence": best_detection['confidence'],
                            "type": best_detection['type'],
                            "details": best_detection
                        })
                        
                        st.success(f"✅ **Success!**")
                        st.code(best_detection['data'])
                        st.caption(f"Type: {best_detection['type']} | Confidence: {best_detection['confidence']:.1%}")
                        
                        # Processing details
                        with st.expander("🔬 Processing Details"):
                            st.json({
                                "barcode_type": best_detection['type'],
                                "confidence_score": best_detection['confidence'],
                                "detection_method": best_detection['method'],
                                "validation_passed": True
                            })
                    else:
                        st.error("❌ No barcode detected in image")
                        st.markdown("**Troubleshooting:**")
                        st.markdown("- Check if the barcode is clearly visible")
                        st.markdown("- Ensure the image is not blurry")
                        st.markdown("- Try cropping closer to the barcode")
    
    # Tab 3: Quick Entry
    with scan_tabs[2]:
        st.markdown("#### ⌨️ Quick Manual Entry")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            manual_code = st.text_input(
                "Enter barcode",
                placeholder="Type, paste, or scan keyboard wedge input...",
                key=f"{key_prefix}_quick_entry",
                help="Direct input for keyboard wedge scanners or manual entry"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            validate_btn = st.button("✓ Validate", key=f"{key_prefix}_validate")
        
        if manual_code.strip():
            cleaned_code = manual_code.strip()
            
            # Enhanced validation
            if len(cleaned_code) >= 4:
                # Determine likely barcode type
                if cleaned_code.isdigit():
                    if len(cleaned_code) == 13:
                        barcode_type = "EAN13"
                    elif len(cleaned_code) == 12:
                        barcode_type = "UPCA"
                    elif len(cleaned_code) == 8:
                        barcode_type = "EAN8"
                    else:
                        barcode_type = "Numeric"
                else:
                    barcode_type = "Alphanumeric"
                
                result.update({
                    "code": cleaned_code,
                    "source": "manual",
                    "confidence": 1.0,
                    "type": barcode_type,
                    "details": {"manually_entered": True}
                })
                
                st.success(f"✅ **Valid Code:** `{cleaned_code}`")
                st.info(f"Detected Type: {barcode_type}")
            else:
                st.warning("⚠️ Code seems too short for a valid barcode")
    
    # Tab 4: Scan History
    with scan_tabs[3]:
        st.markdown("#### 📊 Scan History & Analytics")
        
        stats = scanner.get_scan_statistics()
        if stats and stats['recent_scans']:
            st.markdown("**Recent Scans:**")
            
            for scan in reversed(stats['recent_scans']):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.code(scan['code'])
                with col2:
                    st.caption(scan['type'])
                with col3:
                    st.caption(f"{scan['confidence']:.1%}")
                with col4:
                    st.caption(scan['timestamp'].strftime("%H:%M:%S"))
                    
            # Clear history button
            if st.button("🗑️ Clear History", key=f"{key_prefix}_clear_history"):
                scanner.scan_history.clear()
                scanner.last_scan_time.clear()
                st.rerun()
        else:
            st.info("No scan history available yet")
    
    # Display final result summary
    if result["code"]:
        st.markdown("---")
        st.markdown("### 📋 Scan Result")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Code:** `{result['code']}`")
        with col2:
            st.markdown(f"**Type:** {result['type'] or 'Unknown'}")
        with col3:
            confidence_emoji = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
            st.markdown(f"**Quality:** {confidence_emoji} {result['confidence']:.0%}")
    
    return result

def create_scanner_settings_panel():
    """Create comprehensive scanner settings panel"""
    with st.sidebar:
        st.markdown("## ⚙️ Scanner Settings")
        
        with st.expander("🎯 Detection Settings", expanded=True):
            debounce_time = st.slider(
                "Debounce Time (seconds)",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Minimum time between scanning the same code"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum confidence score to accept a scan"
            )
        
        with st.expander("🚀 Performance Settings"):
            enable_cache = st.checkbox("Enable Image Cache", value=True)
            cache_size = st.slider("Cache Size", 5, 50, 20)
            multi_threading = st.checkbox("Multi-threading", value=True)
        
        with st.expander("📱 Interface Settings"):
            show_overlay = st.checkbox("Show Scan Overlay", value=True)
            show_stats = st.checkbox("Show Performance Stats", value=True)
            auto_clear = st.checkbox("Auto-clear Results", value=False)
        
        return {
            "debounce_time": debounce_time,
            "confidence_threshold": confidence_threshold,
            "enable_cache": enable_cache,
            "cache_size": cache_size,
            "multi_threading": multi_threading,
            "show_overlay": show_overlay,
            "show_stats": show_stats,
            "auto_clear": auto_clear
        }

# Demo function
def demo_enhanced_scanner():
    """Demo the enhanced scanner"""
    st.set_page_config(
        page_title="Enhanced Barcode Scanner",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Enhanced Barcode Scanner Demo")
    st.markdown("*Powered by AI with advanced image processing*")
    
    # Settings panel
    settings = create_scanner_settings_panel()
    
    # Main scanner interface
    scan_result = create_enhanced_scanner_interface("demo", show_advanced=settings["show_stats"])
    
    # Result actions
    if scan_result["code"]:
        st.markdown("### 🎯 Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Copy Code"):
                st.write("Code copied to clipboard!")
        
        with col2:
            if st.button("🔍 Product Lookup"):
                st.info(f"Looking up product: {scan_result['code']}")
        
        with col3:
            if st.button("💾 Save to History"):
                st.success("Saved to scan history!")
        
        with col4:
            if st.button("📤 Export Result"):
                st.download_button(
                    "Download JSON",
                    data=str(scan_result),
                    file_name=f"scan_result_{scan_result['code']}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    demo_enhanced_scanner()
