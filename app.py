import streamlit as st
import sqlite3
from datetime import datetime
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "data.db"

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
    """Return an existing user id for (name,role,zip) or create one."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM users WHERE name=? AND role=? AND zip=?",
        (name.strip(), role.strip(), zip_code.strip())
    )
    row = cur.fetchone()
    if row:
        user_id = row[0]
    else:
        cur.execute(
            "INSERT INTO users (name, role, zip) VALUES (?, ?, ?)",
            (name.strip(), role.strip(), zip_code.strip())
        )
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
# AI-style matching (local)
# -------------------------

def rank_tasks_by_match(tasks: List[tuple], helper_keywords: str):
    """
    Return list of (task_row, score_float) sorted by score desc.
    More forgiving: uses unigrams+bigrams and a simple fallback if TF-IDF yields 0.0 everywhere.
    """
    if not tasks:
        return []

    # Build documents from task fields
    docs = []
    for t in tasks:
        _, title, desc, category, _, _, _, _, _, _, _ = t
        text = " ".join([
            str(title or ""),
            str(desc or ""),
            str(category or "")
        ])
        docs.append(text)

    query = helper_keywords or ""
    # TF-IDF with unigrams+bigrams, english stop words
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform([*docs, query])
    task_vecs = X[:-1]
    helper_vec = X[-1]
    sims = cosine_similarity(task_vecs, helper_vec)
    scores = [float(s[0]) for s in sims]

    # Fallback: if all zero, do a simple keyword overlap score
    if all(s == 0.0 for s in scores) and query.strip():
        q_terms = {w.strip().lower() for w in query.replace(",", " ").split() if len(w.strip()) > 1}
        new_scores = []
        for text in docs:
            t_terms = {w.strip().lower() for w in text.replace(",", " ").split() if len(w.strip()) > 1}
            overlap = len(q_terms & t_terms)
            new_scores.append(float(overlap))
        scores = new_scores

    with_scores = list(zip(tasks, scores))
    with_scores.sort(key=lambda x: x[1], reverse=True)
    return with_scores


# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="NearDoer", page_icon="🧰", layout="wide")

# --- Global CSS + Header with Logo ---
st.markdown("""
<style>
/* Page width + fonts */
.block-container {max-width: 1000px !important;}
h1, h2, h3, h4 {letter-spacing: 0.2px}

/* Top hero banner */
.hero {
  background: linear-gradient(90deg,#2563EB, #7C3AED);
  color: white; padding: 16px 18px; border-radius: 16px;
  margin: 8px 0 14px 0; box-shadow: 0 8px 28px rgba(37,99,235,0.25);
  display:flex; align-items:center; gap:14px;
}
.hero .logo {
  width: 40px; height: 40px; border-radius: 999px; background: #fff2;
  display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:16px; border: 2px solid rgba(255,255,255,0.35);
}
.hero .title {font-size: 22px; font-weight: 800; line-height: 1.2; letter-spacing:0.2px}
.hero .subtitle {opacity: 0.95; font-size: 14px; margin-top: 4px}

/* Re-usable card + badges */
.card {
  border: 1px solid #eaeef4; border-radius: 16px; padding: 14px 16px;
  background: #fff; box-shadow: 0 6px 18px rgba(2,6,23,0.05); margin-bottom: 10px;
}
.card h4 {margin: 0 0 6px 0;}
.badge {display:inline-block; padding:2px 10px; border-radius: 999px; font-size:12px; font-weight:600;}
.badge.assembly{background:#EEF2FF;color:#4F46E5}
.badge.cleaning{background:#ECFDF5;color:#047857}
.badge.errands{background:#FFF7ED;color:#EA580C}
.badge.yard{background:#ECFEFF;color:#0E7490}
.badge.tech{background:#F0FDFA;color:#0F766E}
.badge.other{background:#F1F5F9;color:#334155}
.meta {font-size:12px; color:#475569}
.price {font-size:12px; color:#334155}
.zip {font-size:12px; color:#64748B}
.small {font-size:12px; color:#64748B}
.empty {padding:10px 12px; background:#F8FAFC; border:1px dashed #cfd8e3; border-radius:12px; color:#475569; font-size:13px}
</style>

<!-- Header with a simple "ND" logo -->
<div class="hero">
  <div class="logo">ND</div>
  <div>
    <div class="title">NearDoer — Get Small things done, Fast.</div>
    <div class="subtitle">Post a task · AI ranks matches · Helpers Accept · Done.</div>
  </div>
</div>
""", unsafe_allow_html=True)

init_db()

# Sidebar: user session
st.sidebar.header("Your Profile")
name = st.sidebar.text_input("Name", placeholder="e.g., Alex")
role = st.sidebar.selectbox("Role", ["Poster", "Helper"])
zip_code = st.sidebar.text_input("ZIP code", placeholder="e.g., 60616")

if st.sidebar.button("Save / Switch Profile", use_container_width=True):
    if not name or not zip_code:
        st.sidebar.error("Please enter both Name and ZIP code.")
    else:
        user_id = get_or_create_user(name, role, zip_code)
        st.session_state["user"] = {"id": user_id, "name": name, "role": role, "zip": zip_code}
        st.sidebar.success(f"Profile ready as {role} (ID {user_id}).")
        fetch_tasks.clear()

user = st.session_state.get("user")

if not user:
    st.info("Create your profile in the left sidebar to begin.")
    st.stop()

st.toast("Tip: Press R to refresh if a list looks stale.", icon="ℹ️")

col1, col2 = st.columns([1, 1])

# -------------------------
# Poster view
# -------------------------
if user["role"] == "Poster":
    with col1:
        st.subheader("Post a Task")
        with st.form("post_task_form", clear_on_submit=True):
            title = st.text_input("Title", placeholder="Assemble IKEA shelf")
            desc = st.text_area("Description", placeholder="Need help assembling a KALLAX shelf. Bring a screwdriver.")
            category = st.selectbox("Category", ["Cleaning", "Errands", "Assembly", "Yardwork", "Tech Help", "Other"])
            price = st.text_input("Price (optional)", placeholder="$20 flat")
            zip_in = st.text_input("ZIP", value=user["zip"])  # allow override
            submitted = st.form_submit_button("Post Task")
            if submitted:
                if not title or not desc or not zip_in:
                    st.warning("Title, description, and ZIP are required.")
                else:
                    tid = create_task(title.strip(), desc.strip(), category, price.strip(), zip_in.strip(), user["id"])
                    st.success(f"Task posted (ID {tid}).")
                    fetch_tasks.clear()  # invalidate cache

    with col2:
        st.subheader("Your Tasks")
        my_open = []
        my_accepted = []
        my_completed = []
        tasks = fetch_tasks()
        for t in tasks:
            if t[7] == user["id"]:  # posted_by
                if t[6] == "Open":
                    my_open.append(t)
                elif t[6] == "Accepted":
                    my_accepted.append(t)
                else:
                    my_completed.append(t)

        def render_task(t):
            tid, title, desc, category, price, zipv, status, posted_by, accepted_by, created_at, updated_at = t
            cat_map = {
                "Assembly":"assembly","Cleaning":"cleaning","Errands":"errands",
                "Yardwork":"yard","Tech Help":"tech","Other":"other"
            }
            cat_class = cat_map.get(category or "Other","other")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h4>🧰 {title}</h4>", unsafe_allow_html=True)
            st.markdown(
                f'<span class="badge {cat_class}">{category}</span> &nbsp; '
                f'<span class="zip">ZIP {zipv}</span>',
                unsafe_allow_html=True
            )
            st.write(desc)
            meta = []
            if price: meta.append(f'<span class="price">💵 {price}</span>')
            meta.append(f'<span class="meta">📌 {status}</span>')
            if status == "Accepted" and accepted_by:
                meta.append(f'<span class="meta">🤝 Helper ID {accepted_by}</span>')
            st.markdown(" &nbsp; ".join(meta), unsafe_allow_html=True)

            if status == "Accepted":
                if st.button("Mark Completed", key=f"complete_{tid}"):
                    complete_task(tid)
                    fetch_tasks.clear()
                    st.success("Marked completed.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("**Open**")
        if not my_open:
            st.markdown('<div class="empty">No open tasks yet. Post one on the left!</div>', unsafe_allow_html=True)
        for t in my_open:
            render_task(t)

        st.markdown("**Accepted**")
        if not my_accepted:
            st.markdown('<div class="empty">No accepted tasks yet.</div>', unsafe_allow_html=True)
        for t in my_accepted:
            render_task(t)

        st.markdown("**Completed**")
        if not my_completed:
            st.markdown('<div class="empty">No completed tasks yet.</div>', unsafe_allow_html=True)
        for t in my_completed:
            render_task(t)

# -------------------------
# Helper view
# -------------------------
else:
    with col1:
        st.subheader("Find Tasks Near You")
        filt_zip = st.text_input("Filter by ZIP", value=user["zip"])  # keep simple: exact ZIP match
        category = st.selectbox("Category", ["All", "Cleaning", "Errands", "Assembly", "Yardwork", "Tech Help", "Other"])
        skills = st.text_input("Your skills / interests (comma-separated)", placeholder="cleaning, furniture assembly, laptop setup")

        open_tasks = fetch_tasks(status="Open", zip_filter=filt_zip, category=category)

        if skills.strip():
            ranked = rank_tasks_by_match(open_tasks, skills)
        else:
            ranked = [(t, 0.0) for t in open_tasks]

        st.caption("Tasks ranked by AI Match (0.00–1.00)")
        if not ranked:
            st.markdown('<div class="empty">No open tasks for this ZIP/category yet.</div>', unsafe_allow_html=True)

        for (t, score) in ranked:
            tid, title, desc, category, price, zipv, status, posted_by, accepted_by, created_at, updated_at = t
            cat_map = {
                "Assembly":"assembly","Cleaning":"cleaning","Errands":"errands",
                "Yardwork":"yard","Tech Help":"tech","Other":"other"
            }
            cat_class = cat_map.get(category or "Other","other")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h4>🧰 {title}</h4>", unsafe_allow_html=True)
            st.markdown(
                f'<span class="badge {cat_class}">{category}</span> &nbsp; '
                f'<span class="zip">ZIP {zipv}</span>',
                unsafe_allow_html=True
            )
            st.write(desc)
            row = st.columns([1,1,1])
            with row[0]:
                if price:
                    st.markdown(f'<span class="price">💵 {price}</span>', unsafe_allow_html=True)
            with row[1]:
                st.markdown(f'<span class="small">🤖 AI Match: <b>{score:.2f}</b></span>', unsafe_allow_html=True)
            with row[2]:
                if st.button("Accept Task", key=f"accept_{tid}"):
                    accept_task(tid, user["id"])
                    fetch_tasks.clear()
                    st.success("Accepted! Check your Accepted list.")
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Your Accepted Tasks")
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, description, category, price, zip, status FROM tasks WHERE accepted_by=? ORDER BY id DESC",
            (user["id"],),
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            st.markdown('<div class="empty">You haven’t accepted any tasks yet.</div>', unsafe_allow_html=True)

        for r in rows:
            tid, title, desc, category, price, zipv, status = r
            cat_map = {
                "Assembly":"assembly","Cleaning":"cleaning","Errands":"errands",
                "Yardwork":"yard","Tech Help":"tech","Other":"other"
            }
            cat_class = cat_map.get(category or "Other","other")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h4>🧰 {title}</h4>", unsafe_allow_html=True)
            st.markdown(
                f'<span class="badge {cat_class}">{category}</span> &nbsp; '
                f'<span class="zip">ZIP {zipv}</span>',
                unsafe_allow_html=True
            )
            st.write(desc)
            st.markdown(f'<span class="meta">📌 {status}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
