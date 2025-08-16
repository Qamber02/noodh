# app.py — Enhanced NOODH Admin POS with Returns and Enhanced Scanner
import os
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# Import our enhanced scanner
from enhanced_scanner import create_enhanced_scanner_interface, create_scanner_settings_panel

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="NOODH Admin POS", layout="wide")

DB_PATH = "noodh.db"

st.markdown(
    """
<style>
:root{--card-bg:#ffffff;--muted:#fafafa;--border:#e5e7eb;--radius:14px;}
.block-container {max-width: 1200px;}
h1, h2, h3 { font-weight: 700; }
.card {border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; background: var(--card-bg); margin-bottom: 16px;}
.stButton>button { border-radius: 10px; padding: 0.5rem 1rem; }
.kpi {padding:12px;border:1px solid var(--border);border-radius:12px;background:var(--muted);text-align:center}
.badge {padding:4px 8px;border:1px solid var(--border);border-radius:999px;background:#f8fafc;font-size:12px}
.row {display:flex; gap:12px; align-items:center; flex-wrap:wrap}
.success-box {background: #f0f9ff; border: 1px solid #7dd3fc; border-radius: 8px; padding: 12px; margin: 8px 0;}
.warning-box {background: #fffbeb; border: 1px solid #fbbf24; border-radius: 8px; padding: 12px; margin: 8px 0;}
.error-box {background: #fef2f2; border: 1px solid #f87171; border-radius: 8px; padding: 12px; margin: 8px 0;}
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

    # products
    c.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        barcode TEXT UNIQUE NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL DEFAULT 0
    )""")

    # sales (enhanced with return support)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        total_price_pkr REAL NOT NULL,
        sale_time TEXT NOT NULL,
        user_id INTEGER,
        transaction_type TEXT DEFAULT 'sale',
        original_sale_id INTEGER,
        return_reason TEXT,
        FOREIGN KEY(product_id) REFERENCES products(id),
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(original_sale_id) REFERENCES sales(id)
    )""")

    # settings
    c.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")

    # Safe migrations
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
        c.execute("ALTER TABLE sales ADD COLUMN channel TEXT DEFAULT 'retail'")
    if "transaction_type" not in scols:
        c.execute("ALTER TABLE sales ADD COLUMN transaction_type TEXT DEFAULT 'sale'")
    if "original_sale_id" not in scols:
        c.execute("ALTER TABLE sales ADD COLUMN original_sale_id INTEGER")
    if "return_reason" not in scols:
        c.execute("ALTER TABLE sales ADD COLUMN return_reason TEXT")

    # Backfill retail/wholesale from legacy price if null
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

# ---------------------- AUTH (simplified for demo) ----------------------
try:
    import bcrypt

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

def log_sale(product_id: int, qty: int, user_id: int, channel: str = "retail", 
             transaction_type: str = "sale", original_sale_id: int = None, return_reason: str = None):
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

    # For sales, check stock; for returns, add back to stock
    if transaction_type == "sale":
        if qty > stock:
            return False, "Not enough stock."
        new_stock = stock - qty
        total = unit_price * qty
    else:  # return
        new_stock = stock + qty
        total = -(unit_price * qty)  # Negative for returns

    # Update stock
    c.execute("UPDATE products SET stock=? WHERE id=?", (new_stock, product_id))
    
    # Log transaction
    c.execute("""
        INSERT INTO sales (product_id, quantity, unit_price, total_price_pkr, channel, 
                          sale_time, user_id, transaction_type, original_sale_id, return_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (product_id, int(qty), unit_price, total, channel, datetime.utcnow().isoformat(), 
          user_id, transaction_type, original_sale_id, return_reason))
    
    conn.commit()
    invalidate_caches()
    return True, (name, qty, unit_price, total, channel, transaction_type)

def get_sales_for_returns(days_back: int = 30) -> pd.DataFrame:
    """Get recent sales that can be returned"""
    conn = get_connection()
    cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    df = pd.read_sql_query("""
        SELECT s.id, p.name AS product_name, p.barcode, s.quantity, 
               s.unit_price, s.total_price_pkr, s.channel, s.sale_time,
               u.username
        FROM sales s 
        JOIN products p ON s.product_id = p.id
        LEFT JOIN users u ON s.user_id = u.id
        WHERE s.transaction_type = 'sale' 
        AND s.sale_time > ?
        ORDER BY s.sale_time DESC
    """, conn, params=[cutoff_date])
    
    return df

def get_sales_df(start=None, end=None, product_id=None, include_returns=True) -> pd.DataFrame:
    q = """
    SELECT s.id, p.name AS product_name, p.barcode,
           s.quantity, s.unit_price, s.total_price_pkr, s.channel, s.sale_time,
           s.transaction_type, s.return_reason,
           u.username
    FROM sales s 
    JOIN products p ON s.product_id=p.id
    LEFT JOIN users u ON s.user_id = u.id
    WHERE 1=1
    """
    params = []
    
    if not include_returns:
        q += " AND s.transaction_type = 'sale'"
    
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

# ---------------------- UI COMPONENTS ----------------------
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

    st.header("🔐 Login Required")
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
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    return False

# ---------------------- MAIN PAGES ----------------------
def view_sales_and_returns():
    st.subheader("💰 Sales & Returns")
    
    # Create tabs for sales and returns
    sales_tab, returns_tab = st.tabs(["🛒 Sales", "↩️ Returns"])
    
    with sales_tab:
        st.markdown("#### Enhanced Product Scanner")
        
        # Enhanced scanner interface with all new features
        scan_result = create_enhanced_scanner_interface(key_prefix="sales_scanner", show_advanced=True)
        
        if scan_result["code"]:
            code = scan_result["code"]
            prod = get_product_by_barcode(code)
            
            if not prod:
                st.warning(f"⚠️ Product not found: {code}")
                
                # Enhanced add new product experience
                st.markdown("""
                <div class='warning-box'>
                    <h4>🆕 Product Not Found</h4>
                    <p>The scanned barcode <code>{}</code> is not in your inventory.</p>
                    <p>Would you like to add this as a new product?</p>
                </div>
                """.format(code), unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➕ Add New Product", key="add_new_from_scan", type="primary"):
                        # Store the barcode for use in products page
                        st.session_state["prefill_barcode"] = code
                        st.success("✅ Barcode saved! Switch to Products tab to add this item.")
                
                with col2:
                    if st.button("🔍 Search Again", key="search_again"):
                        st.rerun()
                return
            
            # Enhanced product information display
            confidence_indicator = "🟢" if scan_result["confidence"] > 0.8 else "🟡" if scan_result["confidence"] > 0.6 else "🔴"
            scan_type_badge = f"<span class='badge'>{scan_result['type'] or 'Unknown'}</span>" if scan_result.get('type') else ""
            
            st.markdown(f"""
            <div class='success-box'>
                <h4>📦 {prod['name']} {confidence_indicator}</h4>
                <div class='row'>
                    <span class='badge'>Stock: {prod['stock']}</span>
                    <span class='badge'>Retail: {prod['retail_price']:.2f} PKR</span>
                    <span class='badge'>Wholesale: {prod['wholesale_price']:.2f} PKR</span>
                    <span class='badge'>Barcode: {code}</span>
                    {scan_type_badge}
                    <span class='badge'>Confidence: {scan_result['confidence']:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sales controls with enhanced UI
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                channel = st.radio("Price Type", ["retail", "wholesale"], key=f"channel_{code}")
            
            with col2:
                quantity = st.number_input(
                    "Quantity", 
                    min_value=1, 
                    max_value=max(1, prod['stock']), 
                    value=1, 
                    step=1,
                    key=f"qty_{code}",
                    help=f"Available stock: {prod['stock']}"
                )
            
            unit_price = prod['retail_price'] if channel == 'retail' else prod['wholesale_price']
            total_price = unit_price * quantity
            
            with col3:
                st.metric("Unit Price", f"{unit_price:.2f} PKR", help=f"{channel.title()} price")
            
            with col4:
                st.metric("Total", f"{total_price:.2f} PKR", help="Quantity × Unit Price")
            
            # Enhanced sale buttons with better UX
            st.markdown("#### 💳 Process Sale")
            col_a, col_b, col_c = st.columns([2, 2, 2])
            
            with col_a:
                quick_sell_disabled = prod['stock'] < 1
                quick_button_type = "secondary" if quick_sell_disabled else "primary"
                if st.button("🚀 Quick Sell (1)", 
                           disabled=quick_sell_disabled, 
                           key=f"quick_{code}",
                           type=quick_button_type,
                           help="Instantly sell 1 unit"):
                    success, result = log_sale(prod['id'], 1, st.session_state.user_id, channel)
                    if success:
                        name, qty, unit_p, total, ch, trans_type = result
                        st.success(f"✅ Quick sale completed!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
            
            with col_b:
                custom_sell_disabled = quantity > prod['stock']
                custom_button_type = "secondary" if custom_sell_disabled else "primary"
                if st.button("💰 Process Sale", 
                           disabled=custom_sell_disabled, 
                           key=f"sale_{code}",
                           type=custom_button_type,
                           help=f"Sell {quantity} unit(s)"):
                    success, result = log_sale(prod['id'], quantity, st.session_state.user_id, channel)
                    if success:
                        name, qty, unit_p, total, ch, trans_type = result
                        st.success(f"✅ Sale completed: {qty}x {name} = {total:.2f} PKR ({ch})")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
            
            with col_c:
                if st.button("🔄 Scan Another", key=f"scan_another_{code}", help="Clear and scan next item"):
                    # Clear the scanner state
                    if f"sales_scanner_enhanced_scanner" in st.session_state:
                        del st.session_state[f"sales_scanner_enhanced_scanner"]
                    st.rerun()
        else:
            # Enhanced empty state with tips
            st.markdown("""
            <div class='card'>
                <h4>👆 Ready to Scan</h4>
                <p>Use the enhanced scanner above to:</p>
                <ul>
                    <li>📱 <strong>Smart Camera:</strong> Scan with AI-powered detection and visual guides</li>
                    <li>🖼️ <strong>AI Upload:</strong> Upload images with advanced processing</li>
                    <li>⌨️ <strong>Quick Entry:</strong> Type barcodes manually or use keyboard wedge</li>
                </ul>
                <p><em>The scanner now features improved accuracy, visual guides, and real-time confidence scoring!</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    with returns_tab:
        st.markdown("#### Process Returns")
        
        # Return options
        return_method = st.radio(
            "Return Method", 
            ["Scan Product", "Select from Recent Sales"], 
            horizontal=True
        )
        
        if return_method == "Scan Product":
            st.markdown("**Scan the product to return:**")
            return_scan = create_enhanced_scanner_interface(key_prefix="return_scanner", show_advanced=False)
            
            if return_scan["code"]:
                return_code = return_scan["code"]
                return_prod = get_product_by_barcode(return_code)
                
                if not return_prod:
                    st.error(f"Product not found: {return_code}")
                    return
                
                confidence_indicator = "🟢" if return_scan["confidence"] > 0.8 else "🟡" if return_scan["confidence"] > 0.6 else "🔴"
                
                st.markdown(f"""
                <div class='warning-box'>
                    <h4>🔄 Processing Return {confidence_indicator}</h4>
                    <p><strong>Product:</strong> {return_prod['name']}</p>
                    <p><strong>Current Stock:</strong> {return_prod['stock']}</p>
                    <p><strong>Scan Confidence:</strong> {return_scan['confidence']:.0%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Return form
                with st.form("process_return"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        return_qty = st.number_input("Return Quantity", min_value=1, value=1, step=1)
                        return_channel = st.radio("Original Price Type", ["retail", "wholesale"])
                    
                    with col2:
                        return_reason = st.selectbox(
                            "Return Reason",
                            ["Defective", "Wrong Item", "Customer Changed Mind", "Damaged in Transit", "Other"]
                        )
                        other_reason = st.text_input("Other Reason (if selected)")
                    
                    final_reason = other_reason if return_reason == "Other" else return_reason
                    
                    if st.form_submit_button("🔄 Process Return", type="primary"):
                        success, result = log_sale(
                            return_prod['id'], 
                            return_qty, 
                            st.session_state.user_id, 
                            return_channel,
                            transaction_type="return",
                            return_reason=final_reason
                        )
                        
                        if success:
                            name, qty, unit_p, total, ch, trans_type = result
                            st.success(f"✅ Return processed: {qty}x {name} = {abs(total):.2f} PKR refund")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
        
        else:  # Select from recent sales
            st.markdown("**Recent Sales (Last 30 days):**")
            recent_sales = get_sales_for_returns()
            
            if recent_sales.empty:
                st.info("No recent sales found for returns.")
                return
            
            # Enhanced display of recent sales
            st.dataframe(
                recent_sales[['id', 'product_name', 'quantity', 'total_price_pkr', 'channel', 'sale_time', 'username']], 
                use_container_width=True,
                hide_index=True
            )
            
            # Select sale for return
            sale_options = [
                f"#{row['id']} - {row['product_name']} (Qty: {row['quantity']}, {row['total_price_pkr']:.2f} PKR)"
                for _, row in recent_sales.iterrows()
            ]
            
            if sale_options:
                selected_sale = st.selectbox("Select Sale to Return:", sale_options)
                sale_id = int(selected_sale.split('#')[1].split(' -')[0])
                sale_row = recent_sales[recent_sales['id'] == sale_id].iloc[0]
                
                with st.form("return_from_sale"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Product:** {sale_row['product_name']}")
                        st.write(f"**Original Quantity:** {sale_row['quantity']}")
                        st.write(f"**Original Price:** {sale_row['total_price_pkr']:.2f} PKR")
                        
                        return_qty = st.number_input(
                            "Return Quantity", 
                            min_value=1, 
                            max_value=int(sale_row['quantity']), 
                            value=1, 
                            step=1
                        )
                    
                    with col2:
                        return_reason = st.selectbox(
                            "Return Reason",
                            ["Defective", "Wrong Item", "Customer Changed Mind", "Damaged in Transit", "Other"]
                        )
                        other_reason = st.text_input("Other Reason (if selected)")
                    
                    final_reason = other_reason if return_reason == "Other" else return_reason
                    
                    if st.form_submit_button("🔄 Process Return from Sale", type="primary"):
                        # Get product info
                        return_prod = get_product_by_barcode(sale_row['barcode'])
                        if return_prod:
                            success, result = log_sale(
                                return_prod['id'], 
                                return_qty, 
                                st.session_state.user_id, 
                                sale_row['channel'],
                                transaction_type="return",
                                original_sale_id=sale_id,
                                return_reason=final_reason
                            )
                            
                            if success:
                                name, qty, unit_p, total, ch, trans_type = result
                                st.success(f"✅ Return processed: {qty}x {name} = {abs(total):.2f} PKR refund")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ {result}")

def view_products():
    st.subheader("📦 Product Management")
    
    # Search and filters
    col1, col2, col3 = st.columns([4, 2, 2])
    with col1:
        search = st.text_input("🔍 Search Products", placeholder="Name or barcode...")
    with col2:
        low_stock_threshold = st.number_input("Low Stock Alert", min_value=0, value=5, step=1)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("💾 Save Threshold"):
            set_setting("low_stock_threshold", low_stock_threshold)
            st.success("Saved!")
    
    # Get products
    df = get_products(search)
    
    if not df.empty:
        # Add low stock indicator
        df['Status'] = df['stock'].apply(
            lambda x: "🔴 Low Stock" if x <= low_stock_threshold else "✅ In Stock"
        )
        
        # Enhanced products table display
        st.dataframe(
            df[['id', 'name', 'barcode', 'retail_price', 'wholesale_price', 'stock', 'Status']], 
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": "ID",
                "name": "Product Name",
                "barcode": "Barcode",
                "retail_price": st.column_config.NumberColumn("Retail Price (PKR)", format="%.2f"),
                "wholesale_price": st.column_config.NumberColumn("Wholesale Price (PKR)", format="%.2f"),
                "stock": "Stock",
                "Status": "Status"
            }
        )
        
        # Show low stock alerts
        low_stock_items = df[df['stock'] <= low_stock_threshold]
        if not low_stock_items.empty:
            st.warning(f"⚠️ Low stock alert: {', '.join(low_stock_items['name'].tolist())}")
    else:
        st.info("No products found.")
    
    # Enhanced add new product section
    st.markdown("### ➕ Add New Product")
    
    # Check if barcode was scanned from sales page
    prefill_barcode = st.session_state.get("prefill_barcode", "")
    if prefill_barcode:
        st.markdown(f"""
        <div class='success-box'>
            <h4>📱 Scanned Barcode Ready</h4>
            <p>Adding product for scanned barcode: <code>{prefill_barcode}</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced scanner for adding products
    st.markdown("#### 📱 Scan New Product Barcode")
    add_product_scan = create_enhanced_scanner_interface(key_prefix="add_product_scanner", show_advanced=False)
    
    # Use scanned barcode if available
    if add_product_scan["code"] and not prefill_barcode:
        prefill_barcode = add_product_scan["code"]
        st.session_state["prefill_barcode"] = prefill_barcode
        st.rerun()
    
    with st.form("add_product_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input("Product Name", help="Enter a descriptive product name")
            barcode = st.text_input("Barcode", value=prefill_barcode, help="Scan or enter barcode manually")
            initial_stock = st.number_input("Initial Stock", min_value=0, value=0, step=1, help="Starting inventory quantity")
        
        with col2:
            retail_price = st.number_input("Retail Price (PKR)", min_value=0.0, step=0.50, format="%.2f", help="Customer selling price")
            wholesale_price = st.number_input("Wholesale Price (PKR)", min_value=0.0, step=0.50, format="%.2f", help="Bulk/trade price")
            st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing
        
        # Enhanced form submission
        col_submit1, col_submit2 = st.columns(2)
        with col_submit1:
            submit_add = st.form_submit_button("➕ Add Product", type="primary")
        with col_submit2:
            clear_form = st.form_submit_button("🗑️ Clear Form")
        
        if submit_add:
            if product_name and barcode:
                success, message = add_product(
                    product_name, barcode, retail_price, initial_stock,
                    retail_price, wholesale_price
                )
                if success:
                    st.success(f"✅ {message}")
                    # Clear prefilled barcode
                    if "prefill_barcode" in st.session_state:
                        del st.session_state["prefill_barcode"]
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
            else:
                st.error("❌ Please fill in all required fields.")
        
        if clear_form:
            if "prefill_barcode" in st.session_state:
                del st.session_state["prefill_barcode"]
            st.rerun()

def view_dashboard():
    st.subheader("📊 Sales Dashboard")
    
    # Enhanced date range selector
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    today = datetime.now().date()
    
    with col1:
        start_date = st.date_input("Start Date", value=today - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=today)
    with col3:
        include_returns = st.checkbox("Include Returns", value=True)
    with col4:
        auto_refresh = st.checkbox("Auto Refresh", value=False, help="Refresh data automatically")
    
    # Get sales data
    sales_df = get_sales_df(
        start_date.isoformat(), 
        end_date.isoformat(), 
        include_returns=include_returns
    )
    
    if sales_df.empty:
        st.info("No transactions found for the selected period.")
        return
    
    # Calculate enhanced metrics
    sales_only = sales_df[sales_df['transaction_type'] == 'sale']
    returns_only = sales_df[sales_df['transaction_type'] == 'return']
    
    total_sales = sales_only['total_price_pkr'].sum() if not sales_only.empty else 0
    total_returns = abs(returns_only['total_price_pkr'].sum()) if not returns_only.empty else 0
    net_revenue = total_sales - total_returns
    
    total_orders = len(sales_only)
    total_return_orders = len(returns_only)
    total_items_sold = sales_only['quantity'].sum() if not sales_only.empty else 0
    
    # Enhanced KPI display
    st.markdown("#### 📈 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.metric("Total Sales", f"{total_sales:,.2f} PKR", help="Total sales revenue")
    with kpi2:
        st.metric("Total Returns", f"{total_returns:,.2f} PKR", help="Total returns amount")
    with kpi3:
        st.metric("Net Revenue", f"{net_revenue:,.2f} PKR", help="Sales minus returns")
    with kpi4:
        return_rate = (total_return_orders / total_orders * 100) if total_orders > 0 else 0
        st.metric("Return Rate", f"{return_rate:.1f}%", help="Percentage of orders returned")
    with kpi5:
        avg_order_value = (total_sales / total_orders) if total_orders > 0 else 0
        st.metric("Avg Order Value", f"{avg_order_value:.2f} PKR", help="Average sale amount")
    
    # Additional metrics row
    kpi6, kpi7, kpi8, kpi9 = st.columns(4)
    
    with kpi6:
        st.metric("Total Orders", f"{total_orders:,}", help="Number of sales transactions")
    with kpi7:
        st.metric("Items Sold", f"{total_items_sold:,}", help="Total quantity of items sold")
    with kpi8:
        days_in_period = (end_date - start_date).days + 1
        daily_avg = net_revenue / days_in_period if days_in_period > 0 else 0
        st.metric("Daily Average", f"{daily_avg:.2f} PKR", help="Average daily revenue")
    with kpi9:
        retail_sales = sales_only[sales_only['channel'] == 'retail']['total_price_pkr'].sum() if not sales_only.empty else 0
        wholesale_sales = sales_only[sales_only['channel'] == 'wholesale']['total_price_pkr'].sum() if not sales_only.empty else 0
        retail_percentage = (retail_sales / total_sales * 100) if total_sales > 0 else 0
        st.metric("Retail %", f"{retail_percentage:.1f}%", help="Percentage of retail vs wholesale sales")
    
    # Enhanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Daily Sales Trend")
        if not sales_only.empty:
            daily_sales = sales_only.copy()
            daily_sales['date'] = pd.to_datetime(daily_sales['sale_time']).dt.date
            daily_summary = daily_sales.groupby('date')['total_price_pkr'].sum().reset_index()
            
            # Add returns to the chart if included
            if include_returns and not returns_only.empty:
                daily_returns = returns_only.copy()
                daily_returns['date'] = pd.to_datetime(daily_returns['sale_time']).dt.date
                daily_returns_summary = daily_returns.groupby('date')['total_price_pkr'].sum().reset_index()
                daily_returns_summary['total_price_pkr'] = abs(daily_returns_summary['total_price_pkr'])
                
                # Merge sales and returns
                chart_data = daily_summary.merge(daily_returns_summary, on='date', how='outer', suffixes=('_sales', '_returns')).fillna(0)
                chart_data['net'] = chart_data['total_price_pkr_sales'] - chart_data['total_price_pkr_returns']
                st.line_chart(chart_data.set_index('date')[['total_price_pkr_sales', 'total_price_pkr_returns', 'net']])
            else:
                st.line_chart(daily_summary.set_index('date')['total_price_pkr'])
        else:
            st.info("No sales data to display")
    
    with col2:
        st.markdown("#### 🏆 Top Products")
        if not sales_only.empty:
            top_products = sales_only.groupby('product_name')['total_price_pkr'].sum().sort_values(ascending=False).head(10)
            st.bar_chart(top_products)
        else:
            st.info("No sales data to display")
    
    # Channel analysis
    if not sales_only.empty:
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 💰 Sales by Channel")
            channel_sales = sales_only.groupby('channel')['total_price_pkr'].sum()
            st.bar_chart(channel_sales)
        
        with col4:
            st.markdown("#### 📊 Hourly Sales Pattern")
            hourly_sales = sales_only.copy()
            hourly_sales['hour'] = pd.to_datetime(hourly_sales['sale_time']).dt.hour
            hourly_summary = hourly_sales.groupby('hour')['total_price_pkr'].sum()
            st.line_chart(hourly_summary)
    
    # Enhanced transactions table
    st.markdown("#### 📋 Transaction Details")
    
    # Enhanced filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        transaction_filter = st.selectbox(
            "Filter by Type", 
            ["All", "Sales Only", "Returns Only"]
        )
    with filter_col2:
        product_filter = st.selectbox(
            "Filter by Product", 
            ["All Products"] + list(sales_df['product_name'].unique())
        )
    with filter_col3:
        channel_filter = st.selectbox(
            "Filter by Channel",
            ["All Channels", "Retail", "Wholesale"]
        )
    
    # Apply filters
    filtered_df = sales_df.copy()
    if transaction_filter == "Sales Only":
        filtered_df = filtered_df[filtered_df['transaction_type'] == 'sale']
    elif transaction_filter == "Returns Only":
        filtered_df = filtered_df[filtered_df['transaction_type'] == 'return']
    
    if product_filter != "All Products":
        filtered_df = filtered_df[filtered_df['product_name'] == product_filter]
    
    if channel_filter != "All Channels":
        filtered_df = filtered_df[filtered_df['channel'] == channel_filter.lower()]
    
    # Display filtered data with enhanced formatting
    if not filtered_df.empty:
        display_df = filtered_df[['id', 'product_name', 'quantity', 'unit_price', 'total_price_pkr', 
                                'channel', 'transaction_type', 'sale_time', 'username', 'return_reason']]
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": "Transaction ID",
                "product_name": "Product",
                "quantity": "Qty",
                "unit_price": st.column_config.NumberColumn("Unit Price", format="%.2f PKR"),
                "total_price_pkr": st.column_config.NumberColumn("Total", format="%.2f PKR"),
                "channel": "Channel",
                "transaction_type": "Type",
                "sale_time": st.column_config.DatetimeColumn("Date/Time"),
                "username": "User",
                "return_reason": "Return Reason"
            }
        )
        
        # Enhanced export functionality
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export to CSV",
                data=csv_data,
                file_name=f"transactions_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )
        
        with col_export2:
            # Summary export
            summary_data = {
                'Total Sales': [f"{total_sales:.2f} PKR"],
                'Total Returns': [f"{total_returns:.2f} PKR"],
                'Net Revenue': [f"{net_revenue:.2f} PKR"],
                'Total Orders': [total_orders],
                'Return Rate': [f"{return_rate:.1f}%"]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Export Summary",
                data=summary_csv,
                file_name=f"summary_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )
        
        with col_export3:
            if st.button("🔄 Refresh Data", help="Refresh dashboard data"):
                invalidate_caches()
                st.rerun()
    else:
        st.info("No transactions match the selected filters.")

# ---------------------- MAIN APPLICATION ----------------------
def main():
    init_db()
    
    # Bootstrap admin if no users exist
    if bootstrap_admin_if_empty():
        st.stop()
    
    # Authentication gate
    if not auth_gate():
        st.stop()
    
    # Enhanced header
    st.title("🏪 NOODH POS System")
    st.caption(f"Welcome back, **{st.session_state.username}** ({st.session_state.role}) • Enhanced with AI Scanner")
    
    # Sidebar navigation with enhanced UI
    with st.sidebar:
        st.markdown(f"""
        <div class='card'>
            <h4>👤 User Session</h4>
            <p><strong>{st.session_state.username}</strong></p>
            <p><em>{st.session_state.role.title()}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Enhanced navigation menu
        st.markdown("### 🧭 Navigation")
        menu_options = ["🛒 Sales & Returns", "📦 Products", "📊 Dashboard"]
        
        # Add admin-only options
        if st.session_state.role == "admin":
            menu_options.append("⚙️ Settings")
        
        selected_page = st.radio("", menu_options, index=0)
    
    # Enhanced scanner settings in sidebar
    st.sidebar.markdown("---")
    create_scanner_settings_panel()
    
    # Route to selected page
    if selected_page == "🛒 Sales & Returns":
        view_sales_and_returns()
    elif selected_page == "📦 Products":
        view_products()
    elif selected_page == "📊 Dashboard":
        view_dashboard()
    elif selected_page == "⚙️ Settings" and st.session_state.role == "admin":
        st.subheader("⚙️ System Settings")
        
        # Enhanced settings page
        settings_tabs = st.tabs(["👥 User Management", "🔧 System Config", "📱 Scanner Settings", "💾 Data Management"])
        
        with settings_tabs[0]:
            st.markdown("#### User Management")
            st.info("User management features - Create/edit users, roles, permissions")
        
        with settings_tabs[1]:
            st.markdown("#### System Configuration")
            st.info("System settings - Business info, tax rates, receipt templates")
        
        with settings_tabs[2]:
            st.markdown("#### Scanner Configuration")
            st.info("Advanced scanner settings and calibration")
        
        with settings_tabs[3]:
            st.markdown("#### Data Management")
            st.info("Backup, restore, and data export/import features")

if __name__ == "__main__":
    main()
