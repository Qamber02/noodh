# app.py — NOODH Admin POS (component scanner + retail/wholesale + Sell-1)
import os
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta

from pyzbar import pyzbar
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="NOODH Admin POS", layout="wide")

DB_PATH = "noodh.db"

st.markdown(
    """
<style>
:root{--card-bg:#ffffff;--muted:#fafafa;--border:#e5e7eb;--radius:14px;}
.block-container {max-width: 1100px;}
h1, h2, h3 { font-weight: 700; }
.card {border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; background: var(--card-bg);}
.card + .card {margin-top: 16px;}
.stButton>button { border-radius: 10px; padding: 0.5rem 1rem; }
.kpi {padding:12px;border:1px solid var(--border);border-radius:12px;background:var(--muted);text-align:center}
.badge {padding:4px 8px;border:1px solid var(--border);border-radius:999px;background:#f8fafc;font-size:12px}
.row {display:flex; gap:12px; align-items:center; flex-wrap:wrap}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------- DB (cached resource) ----------------------
@st.cache_resource
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # users
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'admin',
        created_at TEXT NOT NULL
    )""")

    # products (legacy price kept; new retail/wholesale)
    c.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        barcode TEXT UNIQUE NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL DEFAULT 0
    )""")

    # sales (store unit_price + channel)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        total_price_pkr REAL NOT NULL,
        sale_time TEXT NOT NULL,
        user_id INTEGER,
        FOREIGN KEY(product_id) REFERENCES products(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    # settings
    c.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")

    # ----- Safe migrations -----
    def colnames(table):
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}

    pcols = colnames("products")
    if "retail_price" not in pcols:
        c.execute("ALTER TABLE products ADD COLUMN retail_price REAL")
    if "wholesale_price" not in pcols:
        c.execute("ALTER TABLE products ADD COLUMN wholesale_price REAL")

    scols = colnames("sales")
    if "unit_price" not in scols:
        c.execute("ALTER TABLE sales ADD COLUMN unit_price REAL")
    if "channel" not in scols:
        c.execute("ALTER TABLE sales ADD COLUMN channel TEXT")  # 'retail' | 'wholesale'

    # backfill retail/wholesale from legacy price if null
    c.execute("UPDATE products SET retail_price = COALESCE(retail_price, price)")
    c.execute("UPDATE products SET wholesale_price = COALESCE(wholesale_price, price)")

    conn.commit()

# ---------------------- CACHE HELPERS ----------------------
@st.cache_data
def cached_get_products(search: str = "") -> pd.DataFrame:
    conn = get_connection()
    if search:
        df = pd.read_sql_query(
            "SELECT * FROM products WHERE name LIKE ? OR barcode LIKE ? ORDER BY id DESC",
            conn,
            params=(f"%{search}%", f"%{search}%"),
        )
    else:
        df = pd.read_sql_query("SELECT * FROM products ORDER BY id DESC", conn)
    return df

def invalidate_caches():
    try:
        st.cache_data.clear()
    except Exception:
        pass

# ---------------------- SETTINGS HELPERS ----------------------
def get_setting(key: str, default=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    if not row:
        return default
    return row["value"]

def set_setting(key: str, value):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()

def get_setting_bool(key: str, default=False):
    v = get_setting(key, None)
    if v is None:
        return default
    return str(v) == "1"

# ---------------------- AUTH (bcrypt + PBKDF2 fallback) ----------------------
try:
    import bcrypt  # type: ignore

    def _hash_password(password: str) -> (str, str):
        salt = bcrypt.gensalt().hex()
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        return hashed, salt

    def _verify_password(password: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False

except Exception:
    # Fallback (PBKDF2)
    def _hash_password(password: str) -> (str, str):
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000)
        return dk.hex(), salt

    def _verify_password(password: str, hashed: str) -> bool:
        try:
            if len(hashed) == 64:
                test = hashlib.pbkdf2_hmac(
                    "sha256",
                    password.encode("utf-8"),
                    bytes.fromhex(hashed[:32]) if len(hashed) >= 32 else b"",
                    200_000,
                )
                return test.hex() == hashed
            return False
        except Exception:
            return False

def create_user(username: str, password: str, role: str = "admin"):
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password required."
    if role not in ("admin", "staff"):
        role = "staff"
    try:
        hashed, salt = _hash_password(password)
    except Exception:
        return False, "Unable to hash password (missing dependency)."
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, hashed, salt, role, datetime.utcnow().isoformat()),
        )
        conn.commit()
        invalidate_caches()
        return True, "User created."
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def authenticate(username: str, password: str):
    username = username.strip().lower()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, password_hash, salt, role FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if not row:
        return None, None
    uid, stored_hash, salt, role = row["id"], row["password_hash"], row["salt"], row["role"]
    ok = _verify_password(password, stored_hash)
    if ok:
        return uid, role
    return None, None

def set_password(user_id: int, new_password: str):
    try:
        hashed, salt = _hash_password(new_password)
    except Exception:
        return False
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash=?, salt=? WHERE id=?", (hashed, salt, user_id))
    conn.commit()
    invalidate_caches()
    return True

def list_users() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT id, username, role, created_at FROM users ORDER BY id ASC", conn)
    return df

def delete_user(user_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    invalidate_caches()

def users_count() -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    return cur.fetchone()[0]

# ---------------------- PRODUCTS & SALES ----------------------
def get_products(search: str = "") -> pd.DataFrame:
    return cached_get_products(search)

def get_product_by_barcode(code: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, stock,
               COALESCE(retail_price, price) AS retail_price,
               COALESCE(wholesale_price, price) AS wholesale_price
        FROM products WHERE barcode=?
    """, (code,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "stock": int(row["stock"]),
        "retail_price": float(row["retail_price"] or 0),
        "wholesale_price": float(row["wholesale_price"] or 0),
    }

def add_product(name: str, barcode: str, price: float, stock: int,
                retail_price: float = None, wholesale_price: float = None):
    if not name.strip() or not barcode.strip():
        return False, "All fields are required."
    retail_price = float(retail_price if retail_price is not None else price)
    wholesale_price = float(wholesale_price if wholesale_price is not None else price)
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO products (name, barcode, price, stock, retail_price, wholesale_price)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name.strip(), barcode.strip(), float(price), int(stock),
              retail_price, wholesale_price))
        conn.commit()
        invalidate_caches()
        return True, "Product added."
    except sqlite3.IntegrityError:
        return False, "Barcode already exists."

def update_product(pid: int, name: str, retail_price: float, wholesale_price: float, stock: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE products
        SET name=?, retail_price=?, wholesale_price=?, price=?, stock=?
        WHERE id=?
    """, (name, float(retail_price), float(wholesale_price), float(retail_price), int(stock), pid))
    conn.commit()
    invalidate_caches()

def delete_product(pid: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM products WHERE id=?", (pid,))
    conn.commit()
    invalidate_caches()

def log_sale(product_id: int, qty: int, user_id: int, channel: str = "retail"):
    channel = "wholesale" if channel == "wholesale" else "retail"
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT name, stock,
               COALESCE(retail_price, price) AS retail_price,
               COALESCE(wholesale_price, price) AS wholesale_price
        FROM products WHERE id=?
    """, (product_id,))
    row = c.fetchone()
    if not row:
        return False, "Product not found."
    name = row["name"]
    stock = int(row["stock"])
    unit_price = float(row["wholesale_price"] if channel == "wholesale" else row["retail_price"])

    if qty > stock:
        return False, "Not enough stock."

    total = unit_price * int(qty)
    c.execute("UPDATE products SET stock=? WHERE id=?", (stock - qty, product_id))
    c.execute("""
        INSERT INTO sales (product_id, quantity, unit_price, total_price_pkr, channel, sale_time, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (product_id, int(qty), unit_price, total, channel, datetime.utcnow().isoformat(), user_id))
    conn.commit()
    invalidate_caches()
    return True, (name, qty, unit_price, total, channel)

def get_sales_df(start=None, end=None, product_id=None) -> pd.DataFrame:
    q = """
    SELECT s.id, p.name AS product_name, p.barcode,
           s.quantity, s.unit_price, s.total_price_pkr, s.channel, s.sale_time
    FROM sales s JOIN products p ON s.product_id=p.id
    WHERE 1=1
    """
    params = []
    if start:
        q += " AND date(s.sale_time) >= date(?)"
        params.append(start)
    if end:
        q += " AND date(s.sale_time) <= date(?)"
        params.append(end)
    if product_id:
        q += " AND s.product_id = ?"
        params.append(product_id)
    q += " ORDER BY s.sale_time DESC"
    conn = get_connection()
    return pd.read_sql_query(q, conn, params=params)

# ---------------------- IMAGE UTILS ----------------------
def scan_image_bgr(img_bgr) -> list:
    """Return list of decoded strings from a BGR image."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        codes = pyzbar.decode(gray)
        return [c.data.decode("utf-8").strip() for c in codes if c.data]
    except Exception:
        return []

def scan_image_bytes(image_bytes: bytes) -> list:
    file_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        return []
    return scan_image_bgr(img)

# ---------------------- SCANNER TRANSFORMER ----------------------
class BarcodeScanner(VideoTransformerBase):
    """Fast, debounced preview scanner; also keeps last frame for photo capture."""
    def __init__(self):
        self.last_data = None
        self.last_when = None
        self._frame_i = 0
        self._last_fired_data = None
        self._last_fire_time = datetime.min
        self.last_frame = None  # store last BGR frame

        self.SKIP_FRAMES = 2      # decode every 3rd frame
        self.DEBOUNCE_SECS = 2.0  # avoid duplicate fires

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img  # keep latest
        self._frame_i += 1

        if self._frame_i % (self.SKIP_FRAMES + 1) == 0:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                barcodes = pyzbar.decode(gray)
            except Exception:
                barcodes = []

            for bc in barcodes:
                x, y, w, h = bc.rect
                data = bc.data.decode("utf-8").strip()
                now = datetime.utcnow()

                # Debounce identical codes
                if data == self._last_fired_data and (now - self._last_fire_time).total_seconds() < self.DEBOUNCE_SECS:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(img, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    continue

                self.last_data = data
                self.last_when = now
                self._last_fired_data = data
                self._last_fire_time = now

                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 3)
                cv2.putText(img, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 3)
        return img

# ---------------------- REUSABLE SCANNER COMPONENT ----------------------
def scanner_component(key_prefix="scan"):
    """
    Unified scanner UI with tabs:
    - Live Scan (front/back camera)
    - Photo Mode (front/back; Capture & Scan current frame)
    - Upload Image
    Returns dict: {"code": str|None, "source": "live"|"photo"|"upload"|None}
    """
    st.markdown("### 📷 Scanner")
    tabs = st.tabs(["🎦 Live Scan", "📸 Photo", "🖼️ Upload"])
    result = {"code": None, "source": None}

    # Shared camera choice
    with tabs[0]:
        st.markdown("<div class='row'><span class='badge'>Choose camera</span></div>", unsafe_allow_html=True)
        cam_side = st.radio("Camera", ["Back", "Front"], horizontal=True, key=f"{key_prefix}_live_cam")
        facing = "environment" if cam_side == "Back" else "user"

        webrtc_ctx_live = webrtc_streamer(
            key=f"{key_prefix}_live_stream_{facing}",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=BarcodeScanner,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"facingMode": {"exact": facing}}, "audio": False},
            async_transform=True,
            desired_playing_state=True,
        )
        st.caption("Align the barcode inside the view. The border turns green/yellow when detected.")
        live_code = None
        if webrtc_ctx_live and webrtc_ctx_live.video_transformer:
            vt = webrtc_ctx_live.video_transformer
            if getattr(vt, "last_data", None) and getattr(vt, "last_when", None) and datetime.utcnow() - vt.last_when <= timedelta(seconds=3.0):
                live_code = vt.last_data
        col1, col2 = st.columns([2,1])
        with col1:
            st.info("Waiting for scan…") if not live_code else st.success(f"Scanned: {live_code}")
        with col2:
            if st.button("Use this code", disabled=not live_code, key=f"{key_prefix}_use_live"):
                result["code"] = live_code
                result["source"] = "live"
                st.session_state[f"{key_prefix}_picked_code"] = live_code

    with tabs[1]:
        st.markdown("<div class='row'><span class='badge'>Photo mode</span></div>", unsafe_allow_html=True)
        cam_side_p = st.radio("Camera", ["Back", "Front"], horizontal=True, key=f"{key_prefix}_photo_cam")
        facing_p = "environment" if cam_side_p == "Back" else "user"

        st.caption("Tap **Capture & Scan** to decode the current frame.")
        webrtc_ctx_photo = webrtc_streamer(
            key=f"{key_prefix}_photo_stream_{facing_p}",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=BarcodeScanner,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"facingMode": {"exact": facing_p}}, "audio": False},
            async_transform=True,
            desired_playing_state=True,
        )

        photo_feedback = st.empty()
        if st.button("📸 Capture & Scan", key=f"{key_prefix}_capture"):
            code = None
            if webrtc_ctx_photo and webrtc_ctx_photo.video_transformer and getattr(webrtc_ctx_photo.video_transformer, "last_frame", None) is not None:
                frame = webrtc_ctx_photo.video_transformer.last_frame
                found = scan_image_bgr(frame)
                code = found[0] if found else None
            if code:
                photo_feedback.success(f"Scanned: {code}")
                result["code"] = code
                result["source"] = "photo"
                st.session_state[f"{key_prefix}_picked_code"] = code
            else:
                photo_feedback.warning("No barcode found in the captured frame.")

    with tabs[2]:
        st.markdown("<div class='row'><span class='badge'>Upload an image that contains a barcode</span></div>", unsafe_allow_html=True)
        up = st.file_uploader("Upload image", type=["png","jpg","jpeg","webp"], key=f"{key_prefix}_uploader")
        if up is not None:
            codes = scan_image_bytes(up.read())
            if codes:
                st.success(f"Detected: {codes[0]}")
                if st.button("Use this code", key=f"{key_prefix}_use_upload"):
                    result["code"] = codes[0]
                    result["source"] = "upload"
                    st.session_state[f"{key_prefix}_picked_code"] = codes[0]
            else:
                st.error("No barcode found in the uploaded image.")

    # If user already picked in any tab, persist it
    picked = st.session_state.get(f"{key_prefix}_picked_code")
    if picked and not result["code"]:
        result["code"] = picked
        result["source"] = st.session_state.get(f"{key_prefix}_picked_source")

    return result

# ---------------------- UI: Auth / Bootstrap ----------------------
def bootstrap_admin_if_empty():
    if users_count() > 0:
        return False
    st.header("👑 NOODH Admin — Initial Setup")
    st.write("No users found. Create the first **admin** account to continue.")
    with st.form("bootstrap_admin"):
        u = st.text_input("Admin username").strip().lower()
        p1 = st.text_input("Admin password", type="password")
        p2 = st.text_input("Confirm password", type="password")
        ok = st.form_submit_button("Create Admin")
        if ok:
            if not u or not p1:
                st.error("Username and password required.")
            elif p1 != p2:
                st.error("Passwords do not match.")
            else:
                created, msg = create_user(u, p1, role="admin")
                if created:
                    st.success("Admin created. Please log in.")
                    st.rerun()
                else:
                    st.error(msg)
    return True

def auth_gate():
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.role = None

    if st.session_state.user_id:
        return True

    tab_login, tab_register = st.tabs(["🔐 Login", "🆕 Register (Admin only)"])
    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Username").strip().lower()
            p = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                uid, role = authenticate(u, p)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = u
                    st.session_state.role = role
                    st.success("Logged in.")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    with tab_register:
        st.info("Only admins can create users. Log in as admin first, then use User Management.")
    return False

# ---------------------- ADMIN PAGES ----------------------
def view_user_management():
    if st.session_state.role != "admin":
        st.error("Admin only.")
        return
    st.subheader("👥 User Management")
    df = list_users()
    st.dataframe(df, use_container_width=True)

    st.markdown("### Create user")
    with st.form("create_user"):
        c1, c2, c3 = st.columns([3, 3, 2])
        u = c1.text_input("Username").strip().lower()
        p = c2.text_input("Password", type="password")
        role = c3.selectbox("Role", ["staff", "admin"], index=0)
        go = st.form_submit_button("Create")
        if go:
            ok, msg = create_user(u, p, role)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

    st.markdown("### Reset password")
    if not df.empty:
        sel = st.selectbox("Select user", [f"{r['id']} — {r['username']} ({r['role']})" for _, r in df.iterrows()])
        uid = int(sel.split(" — ")[0])
        newp = st.text_input("New password", type="password")
        if st.button("Set new password", disabled=(not newp)):
            set_password(uid, newp)
            st.success("Password updated.")

        st.markdown("### Delete user")
        only_admins = df[df["role"] == "admin"]
        if uid == st.session_state.user_id and len(only_admins) == 1:
            st.warning("You are the only admin. You cannot delete this account.")
        else:
            confirm = st.checkbox("I understand this will permanently remove the user.")
            if st.button("Delete user", disabled=not confirm):
                delete_user(uid)
                st.success("User deleted.")
                if uid == st.session_state.user_id:
                    st.session_state.user_id = None
                    st.session_state.username = None
                    st.session_state.role = None
                st.rerun()

# ---------------------- BUSINESS PAGES ----------------------
def view_products():
    st.subheader("📦 Product Catalog")
    cA, cB, cC = st.columns([3, 2, 2])
    search = cA.text_input("Search by name or barcode", placeholder="Type here…", key="products_search")
    low_stock_thr = int(get_setting("low_stock_threshold", 3) or 3)
    low_stock_thr = cB.number_input("Low-stock threshold", min_value=0, value=low_stock_thr, step=1)
    if cC.button("Save threshold"):
        set_setting("low_stock_threshold", low_stock_thr)
        st.success("Threshold saved.")

    df = get_products(search)
    if not df.empty:
        # Ensure both price columns visible
        if "retail_price" not in df.columns or "wholesale_price" not in df.columns:
            conn = get_connection()
            df = pd.read_sql_query("SELECT * FROM products ORDER BY id DESC", conn)

        df["Low stock?"] = (df["stock"].astype(int) <= low_stock_thr).map({True: "⚠️", False: ""})
        st.dataframe(df[["id","name","barcode","retail_price","wholesale_price","stock","Low stock?"]], use_container_width=True)
        low = df[df["stock"].astype(int) <= low_stock_thr]
        if not low.empty:
            st.warning(f"Low-stock items: {', '.join(low['name'].tolist())}")
    else:
        st.info("No products found.")

    st.markdown("### Add new product")
    with st.form("add_product"):
        c1, c2, c3, c4, c5 = st.columns([3, 3, 2, 2, 2])
        name = c1.text_input("Name")
        barcode = c2.text_input("Barcode")
        retail_price = c3.number_input("Retail Price (PKR)", min_value=0.0, step=0.5, format="%.2f")
        wholesale_price = c4.number_input("Wholesale Price (PKR)", min_value=0.0, step=0.5, format="%.2f")
        stock = c5.number_input("Stock", min_value=0, step=1)
        add = st.form_submit_button("Add")
        if add:
            ok, msg = add_product(name, barcode, retail_price, stock,
                                  retail_price=retail_price, wholesale_price=wholesale_price)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

    st.markdown("### Edit / delete / adjust stock")
    all_df = get_products()
    if not all_df.empty:
        options = [f"{r['id']} — {r['name']} ({r['barcode']})" for _, r in all_df.iterrows()]
        sel = st.selectbox("Select product", options, key="select_product_edit")
        pid = int(sel.split(" — ")[0])
        row = all_df[all_df["id"] == pid].iloc[0]

        with st.form("edit_prod"):
            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
            new_name = c1.text_input("Name", value=row["name"])
            c2.text_input("Barcode (immutable)", value=row["barcode"], disabled=True)
            live = get_product_by_barcode(row["barcode"])
            rp = live["retail_price"] if live else float(row.get("retail_price", row["price"]))
            wp = live["wholesale_price"] if live else float(row.get("wholesale_price", row["price"]))
            new_retail = c3.number_input("Retail (PKR)", value=float(rp), step=0.5, format="%.2f")
            new_wholesale = c4.number_input("Wholesale (PKR)", value=float(wp), step=0.5, format="%.2f")
            new_stock = c5.number_input("Stock", value=int(row["stock"]), step=1)
            save = st.form_submit_button("Save changes")
            if save:
                update_product(pid, new_name, new_retail, new_wholesale, new_stock)
                st.success("Updated.")
                st.rerun()

        st.markdown("**Quick stock adjust**")
        colA, colB, colC = st.columns([2, 2, 3])
        delta = colA.number_input("Adjust by (±)", value=0, step=1, key="quick_adj_delta")
        if colB.button("Apply"):
            if delta != 0:
                update_product(pid, row["name"], float(rp), float(wp), int(row["stock"]) + int(delta))
                st.success(("Increased" if delta > 0 else "Decreased") + " stock.")
                st.rerun()

        st.warning("Deleting a product removes it from the catalog (sales history remains).")
        confirm = colC.checkbox("Confirm delete", key="confirm_delete_prod")
        if colC.button("Delete selected product", disabled=not confirm):
            delete_product(pid)
            st.success("Deleted.")
            st.rerun()
    else:
        st.info("No products yet.")

def view_sales():
    st.subheader("💰 Sales")

    # Settings row
    ctop1, ctop2, ctop3 = st.columns([2, 2, 3])
    with ctop1:
        auto_sell = st.toggle("⚡ Auto-sell 1 on live scan", value=get_setting_bool("auto_sell_one_on_scan", False))
        if st.button("Save", key="save_auto_sell"):
            set_setting("auto_sell_one_on_scan", "1" if auto_sell else "0")
            st.success("Saved")
    with ctop2:
        st.caption("Use photo or upload if camera is not available.")
    with ctop3:
        pass

    # --- Scanner component
    scan_out = scanner_component(key_prefix="sales_scan")
    code_to_use = scan_out["code"]

    # Manual input fallback
    st.markdown("#### Or enter barcode manually")
    manual = st.text_input("Barcode", placeholder="Type or paste code…", key="manual_barcode_input")
    if not code_to_use and manual:
        code_to_use = manual.strip()

    if not code_to_use:
        st.info("Use the scanner above or enter a barcode.")
        return

    # Lookup
    prod = get_product_by_barcode(code_to_use)
    if not prod:
        st.warning(f"⚠️ Code not found in products: {code_to_use}")
        if st.button("Add new product with this barcode", key=f"addprod_{code_to_use}"):
            st.session_state.prefill_barcode = code_to_use
            st.info("Go to Products → Add new product (barcode prefilled).")
        return

    pid = prod["id"]
    pname = prod["name"]
    stock = prod["stock"]
    retail_price = prod["retail_price"]
    wholesale_price = prod["wholesale_price"]

    st.markdown(
        f"<div class='card'><b>Product:</b> {pname} &nbsp;&nbsp; "
        f"<span class='badge'>Stock: {stock}</span> &nbsp; "
        f"<span class='badge'>Retail: {retail_price:.2f} PKR</span> &nbsp; "
        f"<span class='badge'>Wholesale: {wholesale_price:.2f} PKR</span></div>",
        unsafe_allow_html=True,
    )

    # sell controls
    colA, colB, colC, colD = st.columns([2, 2, 2, 3])
    channel = colA.radio("Price type", ["retail", "wholesale"], horizontal=True, key=f"ch_{pid}_{code_to_use}")
    unit_price = retail_price if channel == "retail" else wholesale_price
    qty = colB.number_input("Quantity", min_value=1, max_value=max(1, stock), step=1, value=1, key=f"qty_{pid}_{code_to_use}")
    colC.metric("Unit (PKR)", f"{unit_price:.2f}")
    colD.metric("Total (PKR)", f"{unit_price * qty:,.2f}")

    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("Sell 1", key=f"sell1_{pid}_{code_to_use}", disabled=(stock < 1)):
        ok, info = log_sale(pid, 1, st.session_state.user_id, channel)
        if ok:
            name, q, up, total, ch = info
            st.success(f"Sold 1 ({ch}) — {total:.2f} PKR")
            st.experimental_rerun()
        else:
            st.error(info)

    if c2.button("Confirm Sale", key=f"sale_{pid}_{code_to_use}", disabled=(qty < 1 or qty > stock)):
        ok, info = log_sale(pid, int(qty), st.session_state.user_id, channel)
        if ok:
            name, q, up, total, ch = info
            st.success(f"Sale logged — {q} @ {up:.2f} ({ch}) = {total:.2f} PKR")
            st.session_state["manual_barcode_input"] = ""
            st.experimental_rerun()
        else:
            st.error(info)

    # Auto-sell when a *new* live code appears (debounced in transformer)
    if scan_out["source"] == "live" and auto_sell and stock >= 1:
        # minimal dedupe per run
        fire_key = f"autosell_fired_{code_to_use}_{datetime.utcnow().strftime('%H%M%S')}"
        if not st.session_state.get(fire_key):
            st.session_state[fire_key] = True
            ok, info = log_sale(pid, 1, st.session_state.user_id, channel="retail")
            if ok:
                name, q, up, total, ch = info
                st.success(f"Auto-sold 1 — {total:.2f} PKR")
                st.experimental_rerun()
            else:
                st.error(info)

def view_dashboard():
    st.subheader("📊 Dashboard")
    today = datetime.utcnow().date()
    colA, colB = st.columns(2)
    start = colA.date_input("Start date", value=today - timedelta(days=30))
    end = colB.date_input("End date", value=today)

    products_df = get_products()
    sel = st.selectbox("Filter by product", ["All"] + [f"{r['id']} — {r['name']}" for _, r in products_df.iterrows()], key="dashboard_product_filter")
    pid = None if sel == "All" else int(sel.split(" — ")[0])

    df_sales = get_sales_df(start.isoformat(), end.isoformat(), pid)
    st.write(f"Total sales records: {len(df_sales)}")

    if df_sales.empty:
        st.info("No sales in this range.")
        return

    st.dataframe(df_sales, use_container_width=True)

    total_rev = float(df_sales["total_price_pkr"].sum())
    orders = int(len(df_sales))
    avg_order = total_rev / orders if orders else 0.0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='kpi'><div><b>Total Revenue (PKR)</b></div><div style='font-size:22px'>{total_rev:,.2f}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><div><b>Orders</b></div><div style='font-size:22px'>{orders}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi'><div><b>Avg. Order (PKR)</b></div><div style='font-size:22px'>{avg_order:,.2f}</div></div>", unsafe_allow_html=True)

    daily = df_sales.copy()
    daily["date_only"] = pd.to_datetime(daily["sale_time"]).dt.date
    daily = daily.groupby("date_only")["total_price_pkr"].sum().reset_index().sort_values("date_only")
    if not daily.empty:
        st.markdown("**Daily Sales**")
        st.line_chart(daily.set_index("date_only")["total_price_pkr"])

    top = df_sales.groupby("product_name")["total_price_pkr"].sum().sort_values(ascending=False).head(7)
    if not top.empty:
        st.markdown("**Top Products**")
        st.bar_chart(top)

    by_channel = df_sales.groupby("channel")["total_price_pkr"].sum().sort_values(ascending=False)
    if not by_channel.empty:
        st.markdown("**Revenue by Channel**")
        st.bar_chart(by_channel)

    csv = df_sales.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", data=csv, file_name=f"sales_{start}_{end}.csv", mime="text/csv")

def view_backup():
    st.subheader("🗂️ Backup & Restore")
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            st.download_button("Download DB Backup", data=f, file_name="noodh.db", mime="application/octet-stream")
    up = st.file_uploader("Upload DB to restore (replaces current DB)", type=["db", "sqlite", "sqlite3"])
    if up is not None:
        warn = st.checkbox("I understand this will overwrite the current database.")
        if st.button("Confirm Restore", disabled=not warn):
            try:
                with open(DB_PATH, "wb") as f:
                    f.write(up.read())
                st.success("Database restored. Reloading…")
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")

# ---------------------- MAIN ----------------------
def main():
    init_db()

    if bootstrap_admin_if_empty():
        st.stop()

    st.header("NOODH — Admin POS")
    st.caption("Secure admin console · Dual pricing · Component scanner")

    if not auth_gate():
        st.stop()

    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.username} ({st.session_state.role})")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()
        st.markdown("---")
        menu_items = ["Scan / Sales", "Products", "Dashboard", "Backup"]
        if st.session_state.role == "admin":
            menu_items.insert(1, "User Management")
        menu = st.radio("Menu", menu_items, index=0)

    if menu == "User Management":
        view_user_management()
    elif menu == "Products":
        view_products()
    elif menu == "Dashboard":
        view_dashboard()
    elif menu == "Backup":
        view_backup()
    else:
        view_sales()

if __name__ == "__main__":
    main()
