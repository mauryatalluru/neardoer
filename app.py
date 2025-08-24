import streamlit as st
import sqlite3
from datetime import datetime
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "data.db"
APP_URL = "https://neardoer.streamlit.app"  # change if your URL is different

# -------------------------
# Database helpers
# -------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Poster','Helper')),
            zip TEXT
        )
        """
    )

    cur.execute(
        """
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
        """
    )

    # Testimonials
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS testimonials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            quote TEXT NOT NULL,
            created_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()

@st.cache_data(show_spinner=False)
def fetch_tasks(status: str = None, zip_filter: str = None, category: str = None):
    conn = get_conn()
    cur = conn.cursor()

    query = "SELECT id, title, description, category, price, zip, status, posted_by, accepted_by, created_at, updated_at FROM tasks WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if zip_filter:
        query += " AND zip = ?"
        params.append(zip_filter)
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY id DESC"

    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return rows

def create_user(name: str, role: str, zip_code: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name, role, zip) VALUES (?, ?, ?)", (name, role, zip_code))
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id

def get_or_create_user(name: str, role: str, zip_code: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name=? AND role=? AND zip=?", (name.strip(), role.strip(), zip_code.strip()))
    row = cur.fetchone()
    if row:
        user_id = row[0]
    else:
        cur.execute("INSERT INTO users (name, role, zip) VALUES (?, ?, ?)", (name.strip(), role.strip(), zip_code.strip()))
        conn.commit()
        user_id = cur.lastrowid
    conn.close()
    return user_id

def create_task(title: str, description: str, category: str, price: str, zip_code: str, posted_by: int):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tasks (title, description, category, price, zip, status, posted_by, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'Open', ?, ?, ?)
        """,
        (title, description, category, price, zip_code, posted_by, now, now),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return task_id

def accept_task(task_id: int, helper_id: int):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE tasks SET status='Accepted', accepted_by=?, updated_at=? WHERE id=? AND status='Open'",
        (helper_id, now, task_id),
    )
    conn.commit()
    conn.close()

def complete_task(task_id: int):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE tasks SET status='Completed', updated_at=? WHERE id=? AND status='Accepted'",
        (now, task_id),
    )
    conn.commit()
    conn.close()

# -------------------------
# Testimonials + Stats
# -------------------------

def add_testimonial(name: str, role: str, quote: str):
    now = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO testimonials (name, role, quote, created_at) VALUES (?, ?, ?, ?)", (name.strip(), role.strip(), quote.strip(), now))
    conn.commit()
    conn.close()

@st.cache_data(show_spinner=False)
def fetch_testimonials(limit: int = 6):
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

# -------------------------
# AI-style matching
# -------------------------

def rank_tasks_by_match(tasks: List[tuple], helper_keywords: str):
    if not tasks: return []
    docs = []
    for t in tasks:
        _, title, desc, category, _, _, _, _, _, _, _ = t
        docs.append(" ".join([str(title or ""), str(desc or ""), str(category or "")]))

    query = helper_keywords or ""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform([*docs, query])
    sims = cosine_similarity(X[:-1], X[-1])
    scores = [float(s[0]) for s in sims]

    if all(s == 0.0 for s in scores) and query.strip():
        q_terms = {w.strip().lower() for w in query.split() if len(w.strip()) > 1}
        new_scores = []
        for text in docs:
            t_terms = {w.strip().lower() for w in text.split() if len(w.strip()) > 1}
            new_scores.append(float(len(q_terms & t_terms)))
        scores = new_scores

    with_scores = list(zip(tasks, scores))
    with_scores.sort(key=lambda x: x[1], reverse=True)
    return with_scores

# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="NearDoer", page_icon="🧰", layout="wide")

# Header
st.markdown("""
<style>
.block-container {max-width: 1000px !important;}
.hero {
  background: linear-gradient(90deg,#2563EB, #7C3AED);
  color: white; padding: 14px; border-radius: 12px;
  margin: 6px 0 12px 0; display:flex; align-items:center; gap:12px;
}
.hero .logo {width:40px; height:40px; border-radius:50%; background:#fff2; display:flex; align-items:center; justify-content:center; font-weight:800;}
.card {border:1px solid #eaeef4; border-radius:10px; padding:10px 12px; margin-bottom:8px; background:#fff;}
.empty {padding:8px; background:#f8fafc; border:1px dashed #cfd8e3; border-radius:10px; color:#475569; font-size:13px}
</style>
<div class="hero"><div class="logo">ND</div><div><b>NearDoer</b> — get small things done fast<br><span style="font-size:13px">Post · AI ranks · Accept · Done</span></div></div>
""", unsafe_allow_html=True)

init_db()

# Stats
u,p,a,c = get_stats()
st.markdown(f"""
<div style="display:flex; gap:8px; margin-bottom:10px;">
  <div class="card"><b>👤 Users</b><br>{u}</div>
  <div class="card"><b>📝 Tasks</b><br>{p}</div>
  <div class="card"><b>🤝 Accepted</b><br>{a}</div>
  <div class="card"><b>✅ Completed</b><br>{c}</div>
</div>
""", unsafe_allow_html=True)

st.link_button("🔗 Share NearDoer", f"https://wa.me/?text=Try%20NearDoer%20at%20{APP_URL}", use_container_width=True)
st.link_button("🐦 Share on X", f"https://twitter.com/intent/tweet?text=Try%20NearDoer%20%F0%9F%A7%B0%20{APP_URL}", use_container_width=True)

# Testimonials
st.subheader("What people are saying")
rows = fetch_testimonials()
if not rows:
    st.markdown('<div class="empty">No testimonials yet. Be the first!</div>', unsafe_allow_html=True)
else:
    for (nm, rl, qt) in rows:
        st.markdown(f'<div class="card"><b>{nm}</b> · {rl}<br>{qt}</div>', unsafe_allow_html=True)

with st.expander("Leave a testimonial"):
    tn = st.text_input("Your name")
    tr = st.selectbox("I used NearDoer as…", ["Poster","Helper"])
    tq = st.text_area("Your experience")
    if st.button("Submit testimonial"):
        if tn and tq:
            add_testimonial(tn,tr,tq); fetch_testimonials.clear(); st.success("Thanks! Your testimonial is live.")

# Sidebar profile
st.sidebar.header("Profile")
name = st.sidebar.text_input("Name")
role = st.sidebar.selectbox("Role", ["Poster","Helper"])
zip_code = st.sidebar.text_input("ZIP code")

if st.sidebar.button("Save / Switch Profile"):
    if name and zip_code:
        uid = get_or_create_user(name, role, zip_code)
        st.session_state["user"] = {"id": uid, "name": name, "role": role, "zip": zip_code}
        fetch_tasks.clear()
        st.sidebar.success("Profile saved.")

user = st.session_state.get("user")
if not user: st.stop()

col1,col2 = st.columns([1,1])

# Poster
if user["role"]=="Poster":
    with col1:
        st.subheader("Post a Task")
        with st.form("post_task", clear_on_submit=True):
            t = st.text_input("Title")
            d = st.text_area("Description")
            cat = st.selectbox("Category", ["Cleaning","Errands","Assembly","Yardwork","Tech Help","Other"])
            pr = st.text_input("Price")
            zp = st.text_input("ZIP", value=user["zip"])
            if st.form_submit_button("Post Task") and t and d and zp:
                create_task(t,d,cat,pr,zp,user["id"]); fetch_tasks.clear(); st.success("Task posted!")

    with col2:
        st.subheader("Your Tasks")
        tasks = fetch_tasks()
        for task in tasks:
            tid,tit,desc,cat,pr,zp,stt,pid,aid,_,_ = task
            if pid==user["id"]:
                st.markdown(f"**{tit}** ({stt}) - {desc}")
                if stt=="Accepted" and st.button("Mark Completed", key=f"c{tid}"):
                    complete_task(tid); fetch_tasks.clear(); st.success("Completed!")

# Helper
else:
    with col1:
        st.subheader("Find Tasks")
        filt = st.text_input("Filter by ZIP", value=user["zip"])
        cat = st.selectbox("Category", ["All","Cleaning","Errands","Assembly","Yardwork","Tech Help","Other"])
        skills = st.text_input("Your skills")
        open_tasks = fetch_tasks(status="Open", zip_filter=filt, category=cat)
        ranked = rank_tasks_by_match(open_tasks, skills) if skills else [(t,0.0) for t in open_tasks]
        for (task,sc) in ranked:
            tid,tit,desc,cat,pr,zp,stt,pid,aid,_,_ = task
            st.markdown(f"**{tit}** (AI Match {sc:.2f}) - {desc}")
            if st.button("Accept Task", key=f"a{tid}"):
                accept_task(tid,user["id"]); fetch_tasks.clear(); st.success("Accepted!")

    with col2:
        st.subheader("Your Accepted Tasks")
        conn=get_conn(); cur=conn.cursor()
        cur.execute("SELECT title,description,status FROM tasks WHERE accepted_by=?",(user["id"],))
        rows=cur.fetchall(); conn.close()
        for (tit,desc,stt) in rows:
            st.markdown(f"**{tit}** ({stt}) - {desc}")
