# app.py — NOODH Admin POS (fixed + ready for Streamlit Cloud)
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
.block-container {max-width: 1100px;}
h1, h2, h3 { font-weight: 700; }
.card {border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px; background: #ffffff;}
.card + .card {margin-top: 16px;}
.stButton>button { border-radius: 10px; padding: 0.5rem 1rem; }
.kpi {padding:12px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;text-align:center}
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
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'admin',
        created_at TEXT NOT NULL
    )"""
    )
    # products
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        barcode TEXT UNIQUE NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL DEFAULT 0
    )"""
    )
    # sales
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        total_price_pkr REAL NOT NULL,
        sale_time TEXT NOT NULL,
        user_id INTEGER,
        FOREIGN KEY(product_id) REFERENCES products(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )"""
    )
    # settings
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )"""
    )
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
        # Not a full pbkdf2 verification here (we store pbkdf2 hex digest in hashed in fallback scenario)
        try:
            # If hashed length equals 64 (sha256 hex), compare
            if len(hashed) == 64:
                test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(hashed[:32]) if len(hashed) >= 32 else b"", 200_000)
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
    # Use _hash_password abstraction
    try:
        hashed, salt = _hash_password(password)
    except Exception:
        # last-resort: store plain (NOT recommended) — but keep safe path
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
    cur.execute("SELECT id, name, price, stock FROM products WHERE barcode=?", (code,))
    row = cur.fetchone()
    return (row["id"], row["name"], row["price"], row["stock"]) if row else None


def add_product(name: str, barcode: str, price: float, stock: int):
    if not name.strip() or not barcode.strip():
        return False, "All fields are required."
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("INSERT INTO products (name, barcode, price, stock) VALUES (?, ?, ?, ?)", (name.strip(), barcode.strip(), float(price), int(stock)))
        conn.commit()
        invalidate_caches()
        return True, "Product added."
    except sqlite3.IntegrityError:
        return False, "Barcode already exists."


def update_product(pid: int, name: str, price: float, stock: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE products SET name=?, price=?, stock=? WHERE id=?", (name, float(price), int(stock), pid))
    conn.commit()
    invalidate_caches()


def delete_product(pid: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM products WHERE id=?", (pid,))
    conn.commit()
    invalidate_caches()


def log_sale(product_id: int, qty: int, user_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name, price, stock FROM products WHERE id=?", (product_id,))
    row = c.fetchone()
    if not row:
        return False, "Product not found."
    name, price, stock = row["name"], row["price"], row["stock"]
    if qty > stock:
        return False, "Not enough stock."
    total = float(price) * int(qty)
    c.execute("UPDATE products SET stock=? WHERE id=?", (stock - qty, product_id))
    c.execute("INSERT INTO sales (product_id, quantity, total_price_pkr, sale_time, user_id) VALUES (?, ?, ?, ?, ?)", (product_id, int(qty), total, datetime.utcnow().isoformat(), user_id))
    conn.commit()
    invalidate_caches()
    return True, (name, qty, price, total)


def get_sales_df(start=None, end=None, product_id=None) -> pd.DataFrame:
    q = """SELECT s.id, p.name AS product_name, p.barcode, s.quantity, s.total_price_pkr, s.sale_time
           FROM sales s JOIN products p ON s.product_id=p.id WHERE 1=1"""
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
    df = pd.read_sql_query(q, conn, params=params)
    return df


# ---------------------- SCANNER (VideoTransformerBase) ----------------------
class Scanner(VideoTransformerBase):
    def __init__(self):
        self.last_data = None
        self.last_when = None
        self.ok = False
        self._last_scan_time = datetime.utcnow()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # scan box sized relative to frame
        box = int(min(h, w) * 0.62)
        x1, y1 = (w - box) // 2, (h - box) // 2
        x2, y2 = x1 + box, y1 + box
        roi = img[y1:y2, x1:x2]

        # throttle decoding to every 0.5s for optimization
        now = datetime.utcnow()
        if (now - self._last_scan_time).total_seconds() >= 0.5:
            barcodes = pyzbar.decode(roi)
            self.ok = bool(barcodes)
            if self.ok:
                # pick first barcode value
                try:
                    self.last_data = barcodes[0].data.decode("utf-8")
                except Exception:
                    self.last_data = str(barcodes[0].data)
                self.last_when = now
            self._last_scan_time = now

        # border color
        color = (0, 200, 0) if self.ok else (0, 0, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # corner accents
        L = int(box * 0.12)
        for (xa, ya) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            cv2.line(img, (xa, ya), (xa + (L if xa == x1 else -L), ya), color, 4)
            cv2.line(img, (xa, ya), (xa, ya + (L if ya == y1 else -L)), color, 4)

        # label
        label = "Detected" if self.ok else "Align barcode in the box"
        cv2.putText(img, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # draw bounding boxes for detected barcodes
        barcodes = pyzbar.decode(roi)
        for bc in barcodes:
            pts = bc.polygon
            if pts and len(pts) > 1:
                pts = np.array([(p.x + x1, p.y + y1) for p in pts], dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

        return img


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
        df["Low stock?"] = (df["stock"].astype(int) <= low_stock_thr).map({True: "⚠️", False: ""})
        st.dataframe(df, use_container_width=True)
        low = df[df["stock"].astype(int) <= low_stock_thr]
        if not low.empty:
            st.warning(f"Low-stock items: {', '.join(low['name'].tolist())}")
    else:
        st.info("No products found.")

    st.markdown("### Add new product")
    with st.form("add_product"):
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        name = c1.text_input("Name")
        barcode = c2.text_input("Barcode")
        price = c3.number_input("Price (PKR)", min_value=0.0, step=0.5, format="%.2f")
        stock = c4.number_input("Stock", min_value=0, step=1)
        add = st.form_submit_button("Add")
        if add:
            ok, msg = add_product(name, barcode, price, stock)
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
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            new_name = c1.text_input("Name", value=row["name"])
            c2.text_input("Barcode (immutable)", value=row["barcode"], disabled=True)
            new_price = c3.number_input("Price (PKR)", value=float(row["price"]), step=0.5, format="%.2f")
            new_stock = c4.number_input("Stock", value=int(row["stock"]), step=1)
            save = st.form_submit_button("Save changes")
            if save:
                update_product(pid, new_name, new_price, new_stock)
                st.success("Updated.")
                st.rerun()

        st.markdown("**Quick stock adjust**")
        colA, colB, colC = st.columns([2, 2, 3])
        delta = colA.number_input("Adjust by (±)", value=0, step=1, key="quick_adj_delta")
        if colB.button("Apply"):
            if delta != 0:
                update_product(pid, row["name"], float(row["price"]), int(row["stock"]) + int(delta))
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
    st.subheader("💰 Sales (Scan or search)")

    # start the camera scanner (uses the Scanner VideoTransformer)
    webrtc_ctx = webrtc_streamer(
        key="scanner",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=Scanner,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": {"exact": "environment"}}, "audio": False},
        async_transform=True,
        desired_playing_state=True,
    )
    st.caption("Tip: Align the barcode inside the box. Border turns green when detected.")

    # read a recent scanned value (only if it happened in the last 3 seconds)
    scanned = None
    if webrtc_ctx and webrtc_ctx.video_transformer:
        vt = webrtc_ctx.video_transformer
        if getattr(vt, "last_data", None) and getattr(vt, "last_when", None) and datetime.utcnow() - vt.last_when <= timedelta(seconds=3.0):
            scanned = vt.last_data

    # UI: scanned vs manual input
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Last scanned**")
        if scanned:
            st.success(scanned)
        else:
            st.info("Waiting for scan…")

    with c2:
        manual = st.text_input("Or enter barcode manually", placeholder="Type or paste code…", key="manual_barcode_input")

    # Choose which code to use (scanned preferred)
    code_to_use = scanned or (manual.strip() if manual else None)

    if not code_to_use:
        st.info("Scan a barcode with your camera or enter it manually to load product.")
        return

    # Look up product safely
    prod = None
    try:
        prod = get_product_by_barcode(code_to_use)
    except Exception as e:
        st.error(f"Lookup failed: {e}")
        return

    if not prod:
        st.warning("⚠️ Code not found in products.")
        # show quick add product hint
        if st.button("Add new product with this barcode", key=f"addprod_{code_to_use}"):
            st.session_state.prefill_barcode = code_to_use
            st.info("Go to Products → Add new product (barcode prefilled).")
        return

    pid, pname, price, stock = prod
    st.markdown(f"**Product:** {pname} · **Price:** {price:.2f} PKR · **Stock:** {stock}")

    # ensure stock is an int >= 0
    try:
        max_qty = max(1, int(stock))
    except Exception:
        max_qty = 1

    # unique keys ensure no Streamlit collisions when repeated scans occur
    qty_key = f"qty_{pid}_{code_to_use}"
    sale_key = f"sale_{pid}_{code_to_use}"

    qty = st.number_input("Quantity", min_value=1, max_value=max_qty, step=1, value=1, key=qty_key)

    if st.button("Confirm Sale", key=sale_key):
        ok, info = log_sale(pid, int(qty), st.session_state.user_id)
        if ok:
            name, q, unit, total = info
            st.success(f"Sale logged — total {total:.2f} PKR")
            st.info(f"🧾 Receipt: {name} × {q} @ {unit:.2f} = {total:.2f} PKR")
            # clear manual input to avoid double-selling accidentally
            try:
                st.session_state["manual_barcode_input"] = ""
            except Exception:
                pass
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
    st.caption("Secure admin console · Mobile camera scanning · Streamlit")

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
