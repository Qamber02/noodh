import streamlit as st
import sqlite3
import pandas as pd
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from datetime import datetime
import bcrypt

# ======================
# Database Setup (with st.cache_resource)
# ======================
DB_FILE = "app.db"

@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            barcode TEXT UNIQUE,
            name TEXT,
            price REAL,
            stock INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            quantity INTEGER,
            total_price REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
    """)
    conn.commit()

def bootstrap_admin_if_empty():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        hashed_pw = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", hashed_pw))
        conn.commit()
        st.success("✅ Default admin user created (username: admin, password: admin123)")
        st.rerun()

# ======================
# Authentication
# ======================
def login(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    if user and bcrypt.checkpw(password.encode(), user[2].encode()):
        return user
    return None

# ======================
# Product Management
# ======================
def add_product(barcode, name, price, stock):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO products (barcode, name, price, stock) VALUES (?, ?, ?, ?)",
                  (barcode, name, price, stock))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("❌ Product with this barcode already exists.")

@st.cache_resource
def get_products():
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM products", conn)

# ======================
# Barcode Scanner Processor
# ======================
class BarcodeProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_code = None
        self.color = (0, 0, 255)  # Default red

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        if bbox is not None:
            n = len(bbox)
            for i in range(n):
                cv2.line(img, tuple(bbox[i][0]), tuple(bbox[(i+1) % n][0]), self.color, 2)
            if data:
                self.last_code = data
                self.color = (0, 255, 0)  # Green on success
            else:
                self.color = (0, 0, 255)  # Red if no decode
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ======================
# Dashboard
# ======================
def dashboard():
    conn = get_connection()
    sales = pd.read_sql_query("SELECT * FROM sales", conn)

    st.subheader("📊 Sales Analytics")
    if sales.empty:
        st.info("No sales recorded yet.")
        return

    sales["date_only"] = pd.to_datetime(sales["timestamp"]).dt.date
    daily = sales.groupby("date_only")["total_price"].sum().reset_index()
    st.line_chart(daily.set_index("date_only")["total_price"])

    top_products = pd.read_sql_query("""
        SELECT p.name as product_name, SUM(s.total_price) as total_sales
        FROM sales s
        JOIN products p ON s.product_id = p.id
        GROUP BY s.product_id
        ORDER BY total_sales DESC
        LIMIT 5
    """, conn)

    if not top_products.empty:
        st.bar_chart(top_products.set_index("product_name")["total_sales"])

# ======================
# Main App
# ======================
def main():
    st.set_page_config(page_title="Admin POS", layout="wide")
    init_db()
    bootstrap_admin_if_empty()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔐 Admin Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                user = login(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.success("✅ Login successful")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    else:
        st.sidebar.title("📌 Navigation")
        choice = st.sidebar.radio("Go to", ["Dashboard", "Products", "Scanner", "Logout"])

        if choice == "Dashboard":
            dashboard()

        elif choice == "Products":
            st.subheader("📦 Manage Products")
            with st.form("add_product"):
                barcode = st.text_input("Barcode")
                name = st.text_input("Name")
                price = st.number_input("Price", min_value=0.0, step=0.01)
                stock = st.number_input("Stock", min_value=0, step=1)
                submit = st.form_submit_button("Add Product")
                if submit:
                    add_product(barcode, name, price, stock)
                    st.success("✅ Product added")
                    st.rerun()
            st.dataframe(get_products())

        elif choice == "Scanner":
            st.subheader("📷 Barcode Scanner")
            ctx = webrtc_streamer(
                key="scanner",
                mode=WebRtcMode.RECVONLY,
                video_processor_factory=BarcodeProcessor,
                async_processing=True,
                media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False}
            )
            if ctx.video_processor:
                if ctx.video_processor.last_code:
                    st.success(f"✅ Scanned Code: {ctx.video_processor.last_code}")

        elif choice == "Logout":
            st.session_state.logged_in = False
            st.rerun()

if __name__ == "__main__":
    main()
