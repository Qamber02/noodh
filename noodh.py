# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import sqlite3
import pandas as pd
import datetime
import io
from pyzbar import pyzbar
import cv2
import numpy as np
import os
import tempfile

# -----------------------
# Config & Setup
# -----------------------
st.set_page_config(page_title="NOODH Sales (Streamlit)", layout="centered")
DB_PATH = "noodh_sales.db"

# Ensure DB exists and schema
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        barcode TEXT UNIQUE NOT NULL,
        price_pkr REAL NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        sale_date TEXT NOT NULL,
        total_price_pkr REAL NOT NULL,
        FOREIGN KEY(product_id) REFERENCES products(id)
    )
    """)
    conn.commit()
    return conn, cur

conn, cur = init_db()

# -----------------------
# Helper functions
# -----------------------
def add_product(name, barcode, price):
    try:
        cur.execute("INSERT INTO products (name, barcode, price_pkr) VALUES (?, ?, ?)",
                    (name, barcode, price))
        conn.commit()
        return True, "Product added."
    except sqlite3.IntegrityError:
        return False, "Barcode already exists."

def update_product(pid, name, barcode, price):
    try:
        cur.execute("UPDATE products SET name=?, barcode=?, price_pkr=? WHERE id=?",
                    (name, barcode, price, pid))
        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        return False

def delete_product(pid):
    cur.execute("DELETE FROM products WHERE id=?", (pid,))
    conn.commit()

def get_product_by_barcode(barcode):
    cur.execute("SELECT id, name, price_pkr FROM products WHERE barcode=?", (barcode,))
    return cur.fetchone()

def get_all_products():
    df = pd.read_sql_query("SELECT * FROM products", conn)
    return df

def log_sale(product_id, quantity, sale_date=None):
    if sale_date is None:
        sale_date = datetime.datetime.now().isoformat()
    cur.execute("SELECT price_pkr FROM products WHERE id=?", (product_id,))
    price = cur.fetchone()[0]
    total = price * quantity
    cur.execute("INSERT INTO sales (product_id, quantity, sale_date, total_price_pkr) VALUES (?, ?, ?, ?)",
                (product_id, quantity, sale_date, total))
    conn.commit()
    return total

def get_sales_df(start_date=None, end_date=None, product_id=None):
    q = "SELECT s.id, p.name as product_name, p.barcode, s.quantity, s.sale_date, s.total_price_pkr FROM sales s JOIN products p ON s.product_id = p.id WHERE 1=1"
    params = []
    if start_date:
        q += " AND date(s.sale_date) >= date(?)"
        params.append(start_date)
    if end_date:
        q += " AND date(s.sale_date) <= date(?)"
        params.append(end_date)
    if product_id:
        q += " AND s.product_id = ?"
        params.append(product_id)
    q += " ORDER BY s.sale_date DESC"
    df = pd.read_sql_query(q, conn, params=params)
    return df

def export_sales_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

# -----------------------
# Barcode decode helpers
# -----------------------
def decode_barcodes_from_image_file(image_bytes):
    # image_bytes: bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    barcodes = pyzbar.decode(img)
    return [(b.data.decode("utf-8"), b.type) for b in barcodes]

class BarcodeTransformer(VideoTransformerBase):
    def __init__(self):
        self.last = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        barcodes = pyzbar.decode(img)
        if barcodes:
            # annotate first barcode
            b = barcodes[0]
            x, y, w, h = b.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            text = f"{b.type}:{b.data.decode('utf-8')}"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            self.last = b.data.decode('utf-8')
        return img

# -----------------------
# UI
# -----------------------
st.title("NOODH — Perfume Sales (Streamlit Mobile)")
st.markdown("Offline-capable Streamlit app. Scan or upload barcode images. Data stored locally (SQLite).")

menu = st.sidebar.selectbox("Menu", ["Scan / Quick Sale", "Products", "Sales Dashboard", "Backup & Export", "Settings"])

# --- Scan / Quick Sale ---
if menu == "Scan / Quick Sale":
    st.header("Scan or Upload Barcode")
    st.write("Option A: Live camera (if browser supports). Option B: Upload a photo of the barcode (works everywhere).")

    st.subheader("Live Camera Scanner (if supported)")
    webrtc_ctx = None
    try:
        webrtc_ctx = webrtc_streamer(
            key="noodh-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=BarcodeTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            desired_playing_state=True,
        )
    except Exception as e:
        st.info("Live camera not available in this environment/browser. Use image upload below.")

    last_barcode = None
    if webrtc_ctx and webrtc_ctx.video_transformer:
        st.write("When the camera sees a barcode it will be shown and available below.")
        if st.button("Get last scanned barcode"):
            last_barcode = webrtc_ctx.video_transformer.last
            if last_barcode:
                st.success(f"Last barcode: {last_barcode}")
            else:
                st.warning("No barcode detected yet from camera.")

    st.subheader("Upload Barcode Image (recommended fallback)")
    uploaded = st.file_uploader("Upload an image of the barcode", type=["jpg","jpeg","png"])
    found_codes = []
    if uploaded is not None:
        bytes_data = uploaded.read()
        found_codes = decode_barcodes_from_image_file(bytes_data)
        if not found_codes:
            st.warning("No barcode detected in the image. Try a clearer photo or different angle.")
        else:
            st.success(f"Detected {len(found_codes)} barcode(s).")
            for code, ctype in found_codes:
                st.write(f"- {ctype}: `{code}`")
    # unify barcode source
    barcode_to_use = None
    if found_codes:
        barcode_to_use = found_codes[0][0]
    elif last_barcode:
        barcode_to_use = last_barcode

    if barcode_to_use:
        st.subheader("Product / Sale")
        prod = get_product_by_barcode(barcode_to_use)
        if prod:
            pid, pname, price = prod
            st.write(f"Product found: **{pname}** — Price: **{price} PKR**")
            qty = st.number_input("Quantity", min_value=1, value=1)
            if st.button("Confirm Sale"):
                total = log_sale(pid, int(qty))
                st.success(f"Sale logged — total {total:.2f} PKR")
        else:
            st.warning("Barcode not found in products.")
            with st.form("add_product_form"):
                st.write("Add this product to the catalog")
                name = st.text_input("Product name")
                price = st.number_input("Price (PKR)", min_value=0.0, value=100.0, format="%.2f")
                add_sub = st.form_submit_button("Add product")
                if add_sub:
                    ok, msg = add_product(name, barcode_to_use, float(price))
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

# --- Products ---
elif menu == "Products":
    st.header("Products Catalog")
    df = get_all_products()
    st.dataframe(df)
    st.subheader("Add new product")
    with st.form("new_prod"):
        name = st.text_input("Name")
        barcode = st.text_input("Barcode (numbers)")
        price = st.number_input("Price (PKR)", min_value=0.0, value=500.0, format="%.2f")
        sub = st.form_submit_button("Add product")
        if sub:
            ok, msg = add_product(name, barcode, float(price))
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    st.subheader("Edit / Delete product")
    prod_list = df.to_dict('records')
    if prod_list:
        sel = st.selectbox("Select product", options=[f"{r['id']} - {r['name']}" for r in prod_list])
        pid = int(sel.split(" - ")[0])
        cur.execute("SELECT id, name, barcode, price_pkr FROM products WHERE id=?", (pid,))
        p = cur.fetchone()
        if p:
            _, pname, pbarcode, pprice = p
            with st.form("edit_prod"):
                new_name = st.text_input("Name", value=pname)
                new_barcode = st.text_input("Barcode", value=pbarcode)
                new_price = st.number_input("Price (PKR)", min_value=0.0, value=float(pprice), format="%.2f")
                save = st.form_submit_button("Save changes")
                if save:
                    ok = update_product(pid, new_name, new_barcode, new_price)
                    if ok:
                        st.success("Updated.")
                    else:
                        st.error("Error (maybe barcode duplicate).")
            if st.button("Delete product"):
                delete_product(pid)
                st.success("Deleted.")

# --- Sales Dashboard ---
elif menu == "Sales Dashboard":
    st.header("Sales Dashboard")
    today = datetime.date.today()
    d0 = st.date_input("Start date", value=today - datetime.timedelta(days=30))
    d1 = st.date_input("End date", value=today)
    prod_filter = st.selectbox("Filter by product", options=["All"] + [f"{r['id']} - {r['name']}" for r in get_all_products().to_dict('records')])
    pid = None
    if prod_filter != "All":
        pid = int(prod_filter.split(" - ")[0])
    df_sales = get_sales_df(start_date=d0.isoformat(), end_date=d1.isoformat(), product_id=pid)
    st.write(f"Total sales records: {len(df_sales)}")
    if not df_sales.empty:
        st.dataframe(df_sales)
        st.write("Totals by day:")
        df_tot = df_sales.copy()
        df_tot['date_only'] = pd.to_datetime(df_tot['sale_date']).dt.date
        daily = df_tot.groupby('date_only')['total_price_pkr'].sum().reset_index()
        st.dataframe(daily)
        total_sum = df_sales['total_price_pkr'].sum()
        st.success(f"Total: {total_sum:.2f} PKR")
        csv = export_sales_csv(df_sales)
        st.download_button("Export CSV", data=csv, file_name=f"noodh_sales_{d0}_{d1}.csv", mime="text/csv")
    else:
        st.info("No sales found for the selected range/filter.")

# --- Backup & Export ---
elif menu == "Backup & Export":
    st.header("Backup & Export")
    st.write("Download the SQLite database file (backup) or upload one to restore.")
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            st.download_button("Download DB Backup", data=f, file_name="noodh_sales.db", mime="application/octet-stream")
    uploaded_db = st.file_uploader("Upload DB to restore (will replace current DB)", type=["db","sqlite","sqlite3"])
    if uploaded_db is not None:
        # caution with restore
        st.warning("Restoring will replace your current database. This operation is irreversible unless you have a backup.")
        if st.button("Confirm Restore"):
            with open(DB_PATH, "wb") as f:
                f.write(uploaded_db.read())
            st.success("Database restored. Reload the app to see updated data.")

# --- Settings ---
elif menu == "Settings":
    st.header("Settings & Help")
    st.markdown("""
    - **How to run on Android**
        1. Install Termux or Pydroid 3.
        2. Install Python and required packages (see README below).
        3. Run `streamlit run app.py` and open `http://localhost:8501` in your mobile browser.
    - **Camera notes**
        - Live camera scanning works best in Chrome on Android when using `streamlit-webrtc`.
        - If live camera doesn't work, use image upload (works reliably).
    """)
    st.write("App directory:")
    for f in os.listdir("."):
        st.write(f"- {f}")
