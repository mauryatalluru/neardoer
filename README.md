# NearDoer 🧰

**NearDoer** is a lightweight neighborhood task runner app where people can **post small tasks** (like errands, cleaning, or furniture assembly) and **nearby helpers accept them**.  
It uses a simple **AI matcher** (TF-IDF + cosine similarity) to rank tasks against a helper’s skills — all running locally with no external API costs.  

Built with **Streamlit**, **SQLite**, and **scikit-learn**.  
Deployed free on **Streamlit Community Cloud**.  

---

## 🚀 Demo
👉 Live app: [https://neardoer.streamlit.app](https://neardoer.streamlit.app)  
👉 Walkthrough: Post → Accept → Complete

---

## ✨ Features
- Post tasks with: **title, description, category, price, and ZIP code**.
- Browse open tasks by ZIP and category.
- AI Match score ranks tasks based on helper’s skills.
- Accept → status moves to **Accepted** → Poster can mark as **Completed**.
- Lightweight user profiles (name + role, no passwords).
- Uses SQLite (`data.db`) for storage — created automatically.

---

## 🛠️ Tech Stack
- **Frontend + Backend**: [Streamlit](https://streamlit.io/)  
- **Database**: SQLite (file-based, zero setup)  
- **AI Matching**: [scikit-learn](https://scikit-learn.org/) (TF-IDF + cosine similarity)  
- **Deployment**: GitHub + [Streamlit Cloud](https://streamlit.io/cloud)  

---

## 📂 Project Structure
