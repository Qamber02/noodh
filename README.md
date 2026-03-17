# NOODH POS System - Enhanced with Returns & Optimized Scanner

A modern Point of Sale system built with Streamlit, featuring advanced barcode scanning, product management, sales tracking, and return processing.

##  Features

###  New Features Added
- **Optimized Scanner Module**: Separate, high-performance barcode scanner with multiple detection methods
- **Return Processing**: Complete return workflow with reason tracking
- **Enhanced UI**: Modern interface with better visual feedback
- **Transaction Types**: Support for both sales and returns
- **Improved Camera Support**: Multiple scanning options (camera, upload, manual)

###  Core Features
- **Multi-channel Pricing**: Retail and wholesale pricing support
- **Real-time Inventory**: Automatic stock management
- **User Management**: Role-based access (admin/staff)
- **Sales Dashboard**: Comprehensive analytics and reporting
- **Data Export**: CSV export functionality
- **Secure Authentication**: BCrypt password hashing

## Prerequisites

- Python 3.8 or higher
- Webcam (for barcode scanning)
- Modern web browser

##  Installation

1. **Clone or download the files**:
   ```bash
   # Create project directory
   mkdir noodh-pos
   cd noodh-pos
   
   # Copy the following files to this directory:
   # - app.py
   # - scanner.py
   # - requirements.txt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the system**:
   - Open your browser to `http://localhost:8501`
   - Create the initial admin account when prompted

## Usage Guide

### First Time Setup
1. **Create Admin Account**: The system will prompt you to create an admin account on first run
2. **Add Products**: Go to Products tab and add your inventory
3. **Start Selling**: Use the Sales & Returns tab to process transactions

### Scanner Usage
The enhanced scanner provides three methods:

1. **Camera Scan**: 
   - Choose front or back camera
   - Take a clear photo of the barcode
   - System automatically detects and decodes

2. ** Upload Image**: 
   - Upload photos containing barcodes
   - Supports PNG, JPG, JPEG, WebP, BMP formats

3. **⌨️ Manual Entry**: 
   - Type or paste barcodes directly
   - Useful when camera is unavailable

### Processing Sales
1. Scan or enter product barcode
2. Select retail or wholesale pricing
3. Set quantity
4. Click "Process Sale"

### Processing Returns
1. **Scan Product Method**: Scan the item being returned
2. **Recent Sales Method**: Select from recent transactions
3. Choose return reason
4. Process return (stock is automatically updated)

### Dashboard & Reports
- View sales trends and analytics
- Filter by date range and product
- Export transaction data to CSV
- Monitor return rates and top products

## Architecture

### File Structure
```
noodh-pos/
├── app.py          # Main application
├── scanner.py      # Optimized scanner module
├── requirements.txt # Dependencies
├── noodh.db        # SQLite database (auto-created)
└── README.md       # This file
```

### Database Schema
- **users**: User accounts and roles
- **products**: Inventory with dual pricing
- **sales**: All transactions (sales + returns)
- **settings**: System configuration

### Key Improvements Made

1. **Scanner Optimization**:
   - Separated scanner logic into dedicated module
   - Multiple image preprocessing techniques
   - Better error handling and debouncing
   - Support for various image formats

2. **Return System**:
   - Complete return workflow
   - Reason tracking
   - Automatic inventory adjustment
   - Link returns to original sales

3. **Enhanced UI**:
   - Modern card-based layout
   - Better visual feedback
   - Improved navigation
   - Status indicators

4. **Performance**:
   - Optimized database queries
   - Better caching strategies
   - Streamlined scanning process

## 🔧 Configuration

### Scanner Settings
Adjust scanner behavior in the sidebar:
- **Debounce Time**: Minimum time between scanning same code
- **Auto Enhancement**: Automatic image preprocessing
- **Multiple Attempts**: Try various detection methods

### System Settings
- **Low Stock Threshold**: Set inventory alerts
- **Return Window**: Days allowed for returns (configurable in code)

## 🐛 Troubleshooting

### Scanner Issues
- **Camera not working**: Try manual entry or image upload
- **Barcode not detected**: Ensure good lighting and clear image
- **Multiple codes detected**: System will show options to choose from

### Common Problems
- **Database locked**: Restart the application
- **Login issues**: Check username/password, recreate admin if needed
- **Performance slow**: Clear browser cache, restart application

### Browser Compatibility
- **Chrome/Edge**: Full camera support
- **Firefox**: Camera input supported
- **Safari**: Limited camera features, use upload instead

##  Security Features

- **Password Hashing**: BCrypt with salt
- **Role-based Access**: Admin and staff roles
- **Session Management**: Secure login/logout
- **Data Validation**: Input sanitization

## 📊 Analytics Features

- **Daily Sales Trends**: Line charts of daily revenue
- **Top Products**: Bar charts of best sellers
- **Return Analysis**: Return rates and reasons
- **Channel Performance**: Retail vs wholesale comparison

## 🚀 Future Enhancements

Potential improvements for future versions:
- **Multi-store Support**: Handle multiple locations
- **Advanced Reporting**: More detailed analytics
- **Receipt Printing**: Generate receipts
- **Supplier Management**: Track purchase orders
- **Backup Automation**: Scheduled database backups

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure camera permissions are granted in browser
4. Test with manual entry if scanner fails

## 📄 License

This project is for educational and commercial use. Modify as needed for your requirements.

---

**NOODH POS System** - Modern retail management made simple. 🏪
