import streamlit as st
import sqlite3
import os
import hashlib
from datetime import datetime
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "data.db"
APP_URL = "https://neardoer.streamlit.app"

# =========================
# Password hashing helpers
# =========================
def _pbkdf2_hash(password: str, salt: bytes = None) -> str:
    """
    Returns 'salt_hex$hash_hex'. Uses PBKDF2-HMAC-SHA256, 100k iterations.
    No external dependency required.
    """
    if salt is None:
        salt = os.urandom(16)
    pwd = password.encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", pwd, salt, 100_000)
    return salt.hex() + "$" + dk.hex()

def _pbkdf2_verify(password: str, salted_hash: str) -> bool:
    try:
        salt_hex, hash_hex = salted_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        pwd_hash = _pbkdf2_hash(password, salt).split("$", 1)[1]
        return hashlib.compare_digest(pwd_hash, hash_hex)
    except Exception:
        return False

# =========================
# DB helpers + migration
# =========================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def _column_exists(cur, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return col in cols

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Poster','Helper')),
            zip TEXT,
            skills TEXT
        )
    """)

    # tasks
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT,
            price TEXT,
            zip TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open' CHECK(status IN ('Open','Accepted','Completed')),
            posted_by INTEGER,
            accepted_by INTEGER,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(posted_by) REFERENCES users(id),
            FOREIGN KEY(accepted_by) REFERENCES users(id)
        )
    """)

    # testimonials
    cur.execute("""
        CREATE TABLE IF NOT EXISTS testimonials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            quote TEXT NOT NULL,
            created_at TEXT
        )
    """)

    # ---- migrations on users: add email/password_hash/created_at if missing
    if not _column_exists(cur, "users", "email"):
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if not _column_exists(cur, "users", "password_hash"):
        cur.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if not _column_exists(cur, "users", "created_at"):
        cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")

    # unique email index (if exists)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")

    conn.commit()
    conn.close()

# =========================
# Data functions
# =========================
def get_user_by_email(email: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, role, zip, skills, email, password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return row

def create_account(name: str, email: str, password: str, role: str, zip_code: str, skills: str = "") -> int:
    now = datetime.utcnow().isoformat()
    pwd_hash = _pbkdf2_hash(password)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (name, role, zip, skills, email, password_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name.strip(), role.strip(), zip_code.strip(), skills.strip(), email.strip().lower(), pwd_hash, now))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid

def authenticate(email: str, password: str):
    row = get_user_by_email(email.strip().lower())
    if not row:
        return None
    uid, name, role, zip_code, skills, email_db, pwd_hash = row
    if pwd_hash and _pbkdf2_verify(password, pwd_hash):
        return {"id": uid, "name": name, "role": role, "zip": zip_code, "skills": skills, "email": email_db}
    return None

def get_or_create_user(name, role, zip_code, skills=""):
    """Kept for legacy flows if needed (not used by auth)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name=? AND role=? AND zip=?", (name, role, zip_code))
    row = cur.fetchone()
    if row:
        user_id = row[0]
    else:
        cur.execute("INSERT INTO users (name, role, zip, skills, created_at) VALUES (?,?,?,?,?)",
                    (name, role, zip_code, skills, datetime.utcnow().isoformat()))
        conn.commit()
        user_id = cur.lastrowid
    conn.close()
    return user_id

def create_task(title, description, category, price, zip_code, posted_by):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tasks (title, description, category, price, zip, status, posted_by, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'Open', ?, ?, ?)
    """, (title, description, category, price, zip_code, posted_by, now, now))
    conn.commit()
    conn.close()

def accept_task(task_id, helper_id):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE tasks SET status='Accepted', accepted_by=?, updated_at=? WHERE id=? AND status='Open'",
        (helper_id, now, task_id),
    )
    conn.commit()
    conn.close()

def complete_task(task_id):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE tasks SET status='Completed', updated_at=? WHERE id=? AND status='Accepted'",
        (now, task_id),
    )
    conn.commit()
    conn.close()

def add_testimonial(name, role, quote):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO testimonials (name, role, quote, created_at) VALUES (?,?,?,?)",
                (name, role, quote, now))
    conn.commit()
    conn.close()

def fetch_testimonials(limit=6):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name, role, quote FROM testimonials ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_stats():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users"); users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tasks"); posted = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tasks WHERE status='Accepted'"); accepted = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tasks WHERE status='Completed'"); completed = cur.fetchone()[0]
    conn.close()
    return users, posted, accepted, completed

# =========================
# AI Task Matching
# =========================
def rank_tasks_by_match(tasks: List[tuple], helper_keywords: str):
    if not tasks:
        return []
    docs = [f"{t[1]} {t[2]} {t[3]}" for t in tasks]
    query = helper_keywords or ""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform([*docs, query])
    sims = cosine_similarity(X[:-1], X[-1])
    scores = [float(s[0]) for s in sims]
    with_scores = list(zip(tasks, scores))
    with_scores.sort(key=lambda x: x[1], reverse=True)
    return with_scores

# =========================
# UI helpers
# =========================
def status_badge(status: str) -> str:
    status = (status or "").strip()
    if status == "Completed":
        return '<span class="pill pill-green">Completed</span>'
    if status == "Accepted":
        return '<span class="pill pill-amber">Accepted</span>'
    return '<span class="pill pill-slate">Open</span>'

# =========================
# Page setup + styles
# =========================
st.set_page_config(page_title="NearDoer", page_icon="🧰", layout="wide")

st.markdown("""
<style>
:root{
  --bg1:#0b1022;
  --bg2:#3b0764;
  --panel:#121a2e;
  --muted:#cbd5e1;
  --text:#eef2ff;
  --accent:#7c3aed;
  --accent2:#22c55e;
  --accent3:#f59e0b;
}
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 20% -10%, #1e1b4b 0%, transparent 60%),
              linear-gradient(180deg, var(--bg1), var(--bg2)) !important;
}
[data-testid="stSidebar"]{background:#0f172a !important;}
.block-container{max-width: 1150px;}
h1,h2,h3,h4 { color: var(--text) !important; letter-spacing:.2px }
p, label, span, li, div{ color: var(--muted); }

.card{
  background: var(--panel);
  padding:14px; border-radius:12px; margin:8px 0;
  box-shadow: 0 6px 24px rgba(0,0,0,.35);
}

.stat{
  background: linear-gradient(180deg,#111827,#0b1327);
  border:1px solid rgba(255,255,255,.06);
  border-radius:14px; padding:12px 16px; min-width:150px;
}
.stat b{ color: var(--text); }

.stTextInput>div>div>input,
.stSelectbox>div>div,
textarea, .stTextArea textarea {
  background:#0f172a !important;
  color: var(--muted) !important;
  border:1px solid #243046 !important;
  border-radius:10px !important;
}
.stSelectbox input{ caret-color: transparent !important; }

/* Dropdown menu visibility (Role, Category, etc.) */
div[data-baseweb="select"] div[role="listbox"] {
  background-color: #111827 !important;
  border: 1px solid #374151 !important;
  color: #ffffff !important;
}
div[data-baseweb="select"] div[role="option"] {
  background-color: #111827 !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}
div[data-baseweb="select"] div[role="option"] span {
  color: #ffffff !important;
  font-weight: 700 !important;
}
div[data-baseweb="select"] div[role="option"][aria-selected="true"] {
  background-color: #1f2937 !important;
  color: #22c55e !important;
}
div[data-baseweb="select"] div[role="option"]:hover {
  background-color: #374151 !important;
  color: #ffffff !important;
}
div[data-baseweb="select"] > div {
  background-color: #0f172a !important;
  color: #f9fafb !important;
  border: 1px solid #243046 !important;
}

/* Pills */
.pill{
  display:inline-block; padding:3px 10px; font-size:12px; font-weight:700;
  border-radius:999px; letter-spacing:.2px;
}
.pill-green{ background: rgba(34,197,94,.18); color:#86efac; border:1px solid rgba(34,197,94,.35);}
.pill-amber{ background: rgba(245,158,11,.18); color:#fbbf24; border:1px solid rgba(245,158,11,.35);}
.pill-slate{ background: rgba(148,163,184,.18); color:#cbd5e1; border:1px solid rgba(148,163,184,.35);}

.stButton button{
  background: linear-gradient(90deg,#2563eb,#7c3aed);
  border:0; color:white; font-weight:700; border-radius:10px; padding:8px 14px;
  box-shadow: 0 8px 24px rgba(124,58,237,.35);
}
.stButton button:hover{ filter:brightness(1.1); }

.streamlit-expanderHeader{ color:var(--text) !important; }

.section-title{ font-size:34px; font-weight:800; color:var(--text); margin:6px 0 10px 0; display:flex; gap:10px; align-items:center; }
.section-emoji{ font-size:26px; }
.hero{
  background: linear-gradient(90deg,#2563eb44,#7c3aed44);
  border:1px solid rgba(255,255,255,.08);
  padding:16px; border-radius:16px; margin:6px 0 14px 0;
}
.hero h1{ margin:0; font-size:28px; }
.hero p{ margin:2px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# =========================
# App
# =========================
init_db()

# ---------- AUTH ----------
st.sidebar.header("Account")
if "user" not in st.session_state:
    st.session_state["user"] = None

if st.session_state["user"] is None:
    tab_login, tab_signup = st.sidebar.tabs(["Log in", "Sign up"])

    with tab_login:
        lemail = st.text_input("Email", key="login_email")
        lpwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Log in", use_container_width=True):
            if not lemail or not lpwd:
                st.sidebar.error("Enter email and password.")
            else:
                u = authenticate(lemail, lpwd)
                if u:
                    st.session_state["user"] = u
                    st.sidebar.success(f"Welcome back, {u['name']}!")
                else:
                    st.sidebar.error("Invalid email or password.")

    with tab_signup:
        s_name = st.text_input("Full name", key="su_name")
        s_email = st.text_input("Email", key="su_email")
        s_pwd = st.text_input("Password", type="password", key="su_pwd")
        s_role = st.selectbox("Role", ["Poster","Helper"], key="su_role")
        s_zip  = st.text_input("ZIP code", key="su_zip")
        s_skills = st.text_input("Your skills (comma-separated, helpers only)", key="su_skills")
        if st.button("Create account", use_container_width=True):
            if not (s_name and s_email and s_pwd and s_role and s_zip):
                st.sidebar.error("Please fill all required fields.")
            elif get_user_by_email(s_email.strip().lower()):
                st.sidebar.error("An account with this email already exists.")
            else:
                uid = create_account(s_name, s_email, s_pwd, s_role, s_zip, s_skills if s_role=="Helper" else "")
                # auto-login
                st.session_state["user"] = {
                    "id": uid, "name": s_name, "role": s_role, "zip": s_zip,
                    "skills": s_skills if s_role=="Helper" else "", "email": s_email.strip().lower()
                }
                st.sidebar.success("Account created. You're logged in!")

else:
    u = st.session_state["user"]
    st.sidebar.write(f"**Logged in as:** {u['name']}  \n**Role:** {u['role']}  \n**ZIP:** {u['zip']}")
    if u.get("email"):
        st.sidebar.write(f"**Email:** {u['email']}")
    if st.sidebar.button("Log out", use_container_width=True):
        st.session_state["user"] = None
        st.rerun()

# Require auth to proceed
if st.session_state["user"] is None:
    st.stop()

user = st.session_state["user"]

# ---------- HEADER + STATS ----------
u_count, p_count, a_count, c_count = get_stats()
st.markdown(f"""
<div class="hero">
  <h1>🧰 NearDoer — Get Small Things Done Fast</h1>
  <p>Post · AI ranks · Accept · Done</p>
</div>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 18px 0;">
  <div class="stat"><b>👤 Users</b><br><span>{u_count}</span></div>
  <div class="stat"><b>📝 Tasks</b><br><span>{p_count}</span></div>
  <div class="stat"><b>🤝 Accepted</b><br><span>{a_count}</span></div>
  <div class="stat"><b>✅ Completed</b><br><span>{c_count}</span></div>
</div>
""", unsafe_allow_html=True)

# ---------- TESTIMONIALS ----------
st.markdown('<div class="section-title"><span class="section-emoji">💬</span><span>What people are saying</span></div>', unsafe_allow_html=True)
rows = fetch_testimonials()
if rows:
    for (nm, rl, qt) in rows:
        st.markdown(f"<div class='card'><b style='color:#e2e8f0'>{nm}</b> · {rl}<br>{qt}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='card'>No testimonials yet. Be the first!</div>", unsafe_allow_html=True)

with st.expander("Leave a testimonial"):
    tn = st.text_input("Your name")
    tr = st.selectbox("I used NearDoer as…", ["Poster","Helper"])
    tq = st.text_area("Your experience")
    if st.button("Submit testimonial"):
        if tn and tq:
            add_testimonial(tn, tr, tq)
            st.success("✅ Thanks! Your testimonial is live.")

# ---------- MAIN BODY ----------
col1, col2 = st.columns([1,1])

# Poster view
if user["role"] == "Poster":
    with col1:
        st.markdown('<div class="section-title"><span class="section-emoji">📝</span><span>Post a Task</span></div>', unsafe_allow_html=True)
        with st.form("post_task", clear_on_submit=True):
            t = st.text_input("Title")
            d = st.text_area("Description")
            cat = st.selectbox("Category", ["Cleaning","Errands","Assembly","Yardwork","Tech Help","Other"])
            pr = st.text_input("Price")
            zp = st.text_input("ZIP", value=user["zip"])
            posted = st.form_submit_button("Post Task")
            if posted and t and d and zp:
                create_task(t, d, cat, pr, zp, user["id"])
                st.success("Task posted!")

    with col2:
        st.markdown('<div class="section-title"><span class="section-emoji">🗂️</span><span>Your Tasks</span></div>', unsafe_allow_html=True)
        conn = get_conn(); cur = conn.cursor()
        cur.execute("SELECT id,title,description,status FROM tasks WHERE posted_by=? ORDER BY id DESC", (user["id"],))
        rows = cur.fetchall(); conn.close()
        if not rows:
            st.markdown("<div class='card'>No tasks yet.</div>", unsafe_allow_html=True)
        for tid, tit, desc, stt in rows:
            st.markdown(
                f"<div class='card'><b style='color:#e2e8f0'>{tit}</b> · {status_badge(stt)}<br>{desc}</div>",
                unsafe_allow_html=True
            )
            if stt == "Accepted":
                if st.button("Mark Completed", key=f"c{tid}"):
                    complete_task(tid)
                    st.success("✅ Completed! Refresh to see it in stats.")

# Helper view
else:
    with col1:
        st.markdown('<div class="section-title"><span class="section-emoji">🔎</span><span>Find Tasks</span></div>', unsafe_allow_html=True)
        filt = st.text_input("Filter by ZIP", value=user["zip"])
        catpick = st.selectbox("Category", ["All","Cleaning","Errands","Assembly","Yardwork","Tech Help","Other"])
        conn = get_conn(); cur = conn.cursor()
        if catpick == "All":
            cur.execute("SELECT * FROM tasks WHERE status='Open' AND zip=? ORDER BY id DESC", (filt,))
        else:
            cur.execute("SELECT * FROM tasks WHERE status='Open' AND zip=? AND category=? ORDER BY id DESC", (filt, catpick))
        open_tasks = cur.fetchall(); conn.close()

        ranked = rank_tasks_by_match(open_tasks, user.get("skills","")) if user.get("skills") else [(t, 0.0) for t in open_tasks]
        if not ranked:
            st.markdown("<div class='card'>No open tasks in this ZIP.</div>", unsafe_allow_html=True)
        for (task, sc) in ranked:
            tid, tit, desc, cat, pr, zp, stt, pid, aid, _, _ = task
            price_html = f" · 💵 {pr}" if pr else ""
            st.markdown(
                f"<div class='card'><b style='color:#e2e8f0'>{tit}</b> "
                f"(AI match {sc:.2f}){price_html}<br>{desc}</div>",
                unsafe_allow_html=True
            )
            if st.button("Accept Task", key=f"a{tid}"):
                accept_task(tid, user["id"])
                st.success("Accepted! Ask the poster to mark it completed when done.")

    with col2:
        st.markdown('<div class="section-title"><span class="section-emoji">📑</span><span>Your Accepted Tasks</span></div>', unsafe_allow_html=True)
        conn = get_conn(); cur = conn.cursor()
        cur.execute("SELECT title,description,status FROM tasks WHERE accepted_by=? ORDER BY id DESC", (user["id"],))
        rows = cur.fetchall(); conn.close()
        if not rows:
            st.markdown("<div class='card'>You haven’t accepted any tasks yet.</div>", unsafe_allow_html=True)
        for (tit, desc, stt) in rows:
            st.markdown(
                f"<div class='card'><b style='color:#e2e8f0'>{tit}</b> · {status_badge(stt)}<br>{desc}</div>",
                unsafe_allow_html=True
            )
