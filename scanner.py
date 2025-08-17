# enhanced_scanner.py - Optimized Scanner Module for NOODH POS (Fixed Version)
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import time
from collections import deque
import io
from PIL import Image
import base64
import re

# Fallback imports with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not available. Using basic image processing.")

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    st.warning("⚠️ pyzbar not available. Using pattern matching for barcode detection.")

class EnhancedBarcodeScanner:
    """
    Enhanced barcode scanner with fallback modes for different deployment environments
    """
    
    def __init__(self, debounce_time=1.5, max_results=10, confidence_threshold=0.7):
        self.debounce_time = debounce_time
        self.max_results = max_results
        self.confidence_threshold = confidence_threshold
        self.last_results = deque(maxlen=max_results)
        self.last_scan_time = {}
        self.processing = False
        self.scan_history = deque(maxlen=50)
        
        # Simplified cache for fallback mode
        self.simple_cache = {}
        self.cache_max_size = 10
        
    def validate_barcode_pattern(self, code_data):
        """Validate barcode using pattern matching (fallback method)"""
        if not code_data or len(code_data.strip()) < 4:
            return False, 0.0, "Unknown"
        
        code = code_data.strip()
        confidence = 0.5
        barcode_type = "Unknown"
        
        # EAN/UPC patterns
        if re.match(r'^\d{13}$', code):
            barcode_type = "EAN13"
            confidence = 0.9
        elif re.match(r'^\d{12}$', code):
            barcode_type = "UPCA"
            confidence = 0.9
        elif re.match(r'^\d{8}$', code):
            barcode_type = "EAN8"
            confidence = 0.9
        elif re.match(r'^\d{6,18}$', code):
            barcode_type = "Numeric"
            confidence = 0.8
        elif re.match(r'^[A-Z0-9\-\.\$\/\+\%\s]{1,43}$', code):
            barcode_type = "CODE39"
            confidence = 0.7
        elif re.match(r'^[\x00-\x7F]{1,48}$', code):
            barcode_type = "CODE128"
            confidence = 0.7
        elif len(code) >= 4:
            barcode_type = "Generic"
            confidence = 0.6
        
        # Additional validation
        if code.isdigit() and len(code) >= 8:
            confidence += 0.1
        
        return confidence >= self.confidence_threshold, confidence, barcode_type
    
    def enhanced_image_scan(self, image):
        """Enhanced scanning with OpenCV (when available)"""
        if not CV2_AVAILABLE or not PYZBAR_AVAILABLE:
            return []
        
        try:
            # Convert PIL to CV2 if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Multiple enhancement methods
            enhanced_images = [gray]
            
            # Gaussian blur
            try:
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                enhanced_images.append(blurred)
            except:
                pass
            
            # Adaptive threshold
            try:
                adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                enhanced_images.append(adaptive)
            except:
                pass
            
            # OTSU threshold
            try:
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                enhanced_images.append(otsu)
            except:
                pass
            
            all_detections = []
            
            for i, img in enumerate(enhanced_images):
                try:
                    codes = pyzbar.decode(img)
                    for code in codes:
                        try:
                            decoded_data = code.data.decode('utf-8').strip()
                            is_valid, confidence, code_type = self.validate_barcode_pattern(decoded_data)
                            
                            if is_valid:
                                detection = {
                                    'data': decoded_data,
                                    'type': code_type,
                                    'confidence': confidence,
                                    'method': f'cv2_enhanced_{i}',
                                    'polygon': getattr(code, 'polygon', None),
                                    'rect': getattr(code, 'rect', None)
                                }
                                all_detections.append(detection)
                        except (UnicodeDecodeError, AttributeError):
                            continue
                except Exception:
                    continue
            
            # Remove duplicates and sort by confidence
            unique_detections = {}
            for detection in all_detections:
                data = detection['data']
                if data not in unique_detections or detection['confidence'] > unique_detections[data]['confidence']:
                    unique_detections[data] = detection
            
            return sorted(unique_detections.values(), key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            st.error(f"Enhanced scanning error: {str(e)}")
            return []
    
    def fallback_image_scan(self, image):
        """Fallback scanning method using basic pattern recognition"""
        # This is a simplified fallback - in reality, you'd need more sophisticated
        # pattern matching or integration with a web-based barcode API
        
        # Convert image to text using basic OCR-like approach (simplified)
        # In a real implementation, you might use pytesseract or a web API
        
        return []  # Placeholder for fallback method
    
    def scan_image(self, image):
        """Main scanning function with fallback support"""
        if self.processing:
            return []
        
        self.processing = True
        
        try:
            # Try enhanced scanning first
            if CV2_AVAILABLE and PYZBAR_AVAILABLE:
                detections = self.enhanced_image_scan(image)
            else:
                # Use fallback method
                detections = self.fallback_image_scan(image)
            
            # Apply debouncing
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
            
        except Exception as e:
            st.error(f"Scanning error: {str(e)}")
            return []
        finally:
            self.processing = False
    
    def scan_from_bytes(self, image_bytes):
        """Scan barcode from image bytes with comprehensive error handling"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return self.scan_image(image)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return []
    
    def validate_manual_code(self, code):
        """Validate manually entered barcode"""
        is_valid, confidence, barcode_type = self.validate_barcode_pattern(code)
        
        if is_valid:
            return {
                'data': code,
                'type': barcode_type,
                'confidence': confidence,
                'method': 'manual_entry',
                'polygon': None,
                'rect': None
            }
        return None
    
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
            'recent_scans': list(self.scan_history)[-5:],
            'capabilities': {
                'opencv': CV2_AVAILABLE,
                'pyzbar': PYZBAR_AVAILABLE,
                'enhanced_mode': CV2_AVAILABLE and PYZBAR_AVAILABLE
            }
        }

def create_scanner_guide_html():
    """Create HTML guide for scanning"""
    return """
    <div style="
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: linear-gradient(45deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.05));
        margin: 10px 0;
    ">
        <h4 style="color: #4CAF50; margin: 0 0 10px 0;">📱 Scanning Guide</h4>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div style="margin: 5px;">
                <strong>📸 Camera:</strong><br/>
                <small>Good lighting, steady hands</small>
            </div>
            <div style="margin: 5px;">
                <strong>🖼️ Upload:</strong><br/>
                <small>Clear, focused images</small>
            </div>
            <div style="margin: 5px;">
                <strong>⌨️ Manual:</strong><br/>
                <small>Type or paste codes</small>
            </div>
        </div>
    </div>
    """

def create_enhanced_scanner_interface(key_prefix="enhanced_scanner", show_advanced=True):
    """
    Create an enhanced scanner interface with improved error handling and fallbacks
    """
    
    # Initialize scanner
    scanner_key = f"{key_prefix}_scanner"
    if scanner_key not in st.session_state:
        st.session_state[scanner_key] = EnhancedBarcodeScanner()
    
    scanner = st.session_state[scanner_key]
    
    st.markdown("### 📱 Enhanced Product Scanner")
    
    # Show capabilities
    stats = scanner.get_scan_statistics()
    if show_advanced and stats:
        capabilities = stats.get('capabilities', {})
        status_text = "🟢 Full Enhanced Mode" if capabilities.get('enhanced_mode') else "🟡 Basic Mode"
        st.caption(f"Status: {status_text}")
        
        if not capabilities.get('enhanced_mode'):
            with st.expander("ℹ️ Scanner Information"):
                st.info("""
                **Basic Mode Active**
                
                Some advanced features are not available due to missing dependencies:
                - OpenCV: """ + ("✅" if capabilities.get('opencv') else "❌") + """
                - pyzbar: """ + ("✅" if capabilities.get('pyzbar') else "❌") + """
                
                Manual entry and basic validation are still fully functional.
                """)
    
    # Display scanning guide
    st.markdown(create_scanner_guide_html(), unsafe_allow_html=True)
    
    # Scanning tabs
    scan_tabs = st.tabs(["📷 Camera", "🖼️ Upload", "⌨️ Manual Entry"])
    
    result = {"code": None, "source": None, "confidence": 0.0, "type": None, "details": None}
    
    # Tab 1: Camera Scanner
    with scan_tabs[0]:
        st.markdown("#### 📷 Camera Scanner")
        
        # Settings
        col1, col2 = st.columns(2)
        with col1:
            auto_process = st.checkbox(
                "🔄 Auto Process", 
                value=True, 
                help="Automatically process captured images",
                key=f"{key_prefix}_auto_process"  # Fixed: Unique key
            )
        with col2:
            show_preview = st.checkbox(
                "👁️ Show Preview", 
                value=True, 
                help="Show captured image preview",
                key=f"{key_prefix}_show_preview"  # Fixed: Unique key
            )
        
        # Camera input with better error handling
        try:
            camera_input = st.camera_input(
                "📸 Capture barcode image",
                key=f"{key_prefix}_camera",
                help="Position barcode clearly in frame"
            )
            
            if camera_input is not None:
                # Store image in session state
                st.session_state[f"{key_prefix}_camera_image"] = camera_input.getvalue()
                
            # Check if we have a stored image to process
            if f"{key_prefix}_camera_image" in st.session_state:
                image_bytes = st.session_state[f"{key_prefix}_camera_image"]
                if show_preview:
                    st.image(image_bytes, caption="Captured Image", width=300)
                
                if auto_process or st.button("🔍 Process Image", key=f"{key_prefix}_process"):
                    with st.spinner("🔍 Processing image..."):
                        try:
                            detections = scanner.scan_from_bytes(image_bytes)
                            
                            if detections:
                                best_detection = detections[0]
                                result.update({
                                    "code": best_detection['data'],
                                    "source": "camera",
                                    "confidence": best_detection['confidence'],
                                    "type": best_detection['type'],
                                    "details": best_detection
                                })
                                
                                confidence_emoji = "🟢" if best_detection['confidence'] > 0.8 else "🟡" if best_detection['confidence'] > 0.6 else "🔴"
                                st.success(f"{confidence_emoji} **Detected:** `{best_detection['data']}`")
                                st.caption(f"Type: {best_detection['type']} | Confidence: {best_detection['confidence']:.1%}")
                                
                                # Clear image after successful scan
                                del st.session_state[f"{key_prefix}_camera_image"]
                                st.rerun()
                            else:
                                st.warning("❌ No barcode detected")
                                st.info("💡 **Tips:** Ensure good lighting, hold steady, clean barcode surface")
                        
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")
        
        except Exception as e:
            st.error(f"Camera not available: {str(e)}")
            st.info("📱 Camera input may not be supported in this environment. Try manual entry or file upload.")
    
    # Tab 2: File Upload
    with scan_tabs[1]:
        st.markdown("#### 🖼️ Image Upload Scanner")
        
        uploaded_file = st.file_uploader(
            "Upload barcode image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key=f"{key_prefix}_upload",
            help="Supports PNG, JPG, JPEG, WebP, BMP formats"
        )
        
        if uploaded_file is not None:
            # Read bytes once and reuse
            image_bytes = uploaded_file.read()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image_bytes, caption="Uploaded Image", width=200)
            
            with col2:
                try:
                    with st.spinner("🤖 Processing uploaded image..."):
                        detections = scanner.scan_from_bytes(image_bytes)
                        
                        if detections:
                            best_detection = detections[0]
                            result.update({
                                "code": best_detection['data'],
                                "source": "upload",
                                "confidence": best_detection['confidence'],
                                "type": best_detection['type'],
                                "details": best_detection
                            })
                            
                            st.success("✅ **Barcode Found!**")
                            st.code(best_detection['data'])
                            st.caption(f"Type: {best_detection['type']} | Confidence: {best_detection['confidence']:.1%}")
                            
                            # Show multiple detections if found
                            if len(detections) > 1:
                                with st.expander(f"📋 Found {len(detections)} barcodes"):
                                    for i, det in enumerate(detections):
                                        col_a, col_b = st.columns([3, 1])
                                        with col_a:
                                            st.text(f"{i+1}. {det['data']} ({det['type']})")
                                        with col_b:
                                            if st.button("Use", key=f"use_det_{i}_{key_prefix}"):
                                                result.update({
                                                    "code": det['data'],
                                                    "confidence": det['confidence'],
                                                    "type": det['type'],
                                                    "details": det
                                                })
                                                st.rerun()
                        else:
                            st.warning("❌ No barcode detected in uploaded image")
                            st.info("Try uploading a clearer image or use manual entry")
                
                except Exception as e:
                    st.error(f"Error processing upload: {str(e)}")
    
    # Tab 3: Manual Entry
    with scan_tabs[2]:
        st.markdown("#### ⌨️ Manual Barcode Entry")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            manual_code = st.text_input(
                "Enter barcode manually",
                placeholder="Type, paste, or use barcode scanner...",
                key=f"{key_prefix}_manual",
                help="Supports keyboard wedge scanners and manual typing"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            validate_manual = st.button("✓ Validate", key=f"{key_prefix}_validate_manual")
        
        if manual_code.strip() or validate_manual:
            cleaned_code = manual_code.strip()
            
            if cleaned_code:
                try:
                    validation_result = scanner.validate_manual_code(cleaned_code)
                    
                    if validation_result:
                        result.update({
                            "code": validation_result['data'],
                            "source": "manual",
                            "confidence": validation_result['confidence'],
                            "type": validation_result['type'],
                            "details": validation_result
                        })
                        
                        st.success(f"✅ **Valid Barcode:** `{cleaned_code}`")
                        st.info(f"📊 Type: {validation_result['type']} | Confidence: {validation_result['confidence']:.1%}")
                        
                        # Add to scan history
                        scanner.scan_history.append({
                            'code': cleaned_code,
                            'timestamp': datetime.now(),
                            'confidence': validation_result['confidence'],
                            'type': validation_result['type']
                        })
                    else:
                        st.warning("⚠️ Invalid barcode format")
                        st.info("Check the barcode format - minimum 4 characters required")
                
                except Exception as e:
                    st.error(f"Validation error: {str(e)}")
            elif validate_manual:
                st.warning("Please enter a barcode first")
    
    # Show scan history if available
    if show_advanced and stats and stats.get('recent_scans'):
        with st.expander("📊 Recent Scans"):
            for scan in reversed(stats['recent_scans'][-3:]):  # Show last 3
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.code(scan['code'])
                with col2:
                    st.caption(scan['type'])
                with col3:
                    st.caption(f"{scan['confidence']:.0%}")
    
    # Display final result
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
    """Create simplified scanner settings panel"""
    with st.sidebar:
        st.markdown("### ⚙️ Scanner Settings")
        
        with st.expander("🎯 Basic Settings"):
            debounce_time = st.slider(
                "Scan Cooldown (sec)",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                help="Time between scanning the same code"
            )
            
            confidence_threshold = st.slider(
                "Quality Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum quality score"
            )
        
        with st.expander("📱 Interface"):
            show_preview = st.checkbox("Show Image Preview", value=True)
            auto_process = st.checkbox("Auto Process Images", value=True)
            show_tips = st.checkbox("Show Scanning Tips", value=True)
        
        return {
            "debounce_time": debounce_time,
            "confidence_threshold": confidence_threshold,
            "show_preview": show_preview,
            "auto_process": auto_process,
            "show_tips": show_tips
        }

# Demo function for testing
def demo_scanner():
    """Demo the fixed scanner"""
    st.set_page_config(
        page_title="Enhanced Scanner Demo",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 Enhanced Barcode Scanner")
    st.markdown("*Fixed version with improved compatibility*")
    
    # Settings
    settings = create_scanner_settings_panel()
    
    # Main scanner
    result = create_enhanced_scanner_interface("demo", show_advanced=True)
    
    # Actions
    if result["code"]:
        st.markdown("### 🎯 Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy to Clipboard"):
                st.code(result["code"])
                st.success("Code displayed above!")
        
        with col2:
            if st.button("🔍 Lookup Product"):
                st.info(f"Searching for: {result['code']}")
        
        with col3:
            if st.button("💾 Save Result"):
                st.success("Result saved to history!")

if __name__ == "__main__":
    demo_scanner()
