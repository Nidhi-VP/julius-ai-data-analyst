# MYAI — Julius AI-Style Data Analyst Web Application

A security-first, high-performance AI Data Analyst web application built with Python and Streamlit. Inspired by Julius AI, MYAI allows users to upload CSV or Excel files, inspect schemas, run read-only DuckDB SQL queries, chat with an AI analyst that generates sandboxed code or SQL, build interactive multi-chart dashboards, and export datasets or offline HTML dashboards—all styled with a refined Studio design theme.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10+
- Virtual Environment

### 2. Setup Virtual Environment & Dependencies
```bash
# Clone/Navigate to workspace
cd c:\Users\Nidhi\data

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables (.env)
Create or edit your `.env` file (refer to `.env.example`):
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Default Demo Credentials:**
- **Username:** `demo`
- **Password:** `password123`

---

## 🧪 Running Automated Tests

The application includes automated test coverage verifying security isolation, SQL parser validation, and end-to-end agent chart generation.

```bash
# Run pytest suite
$env:PYTHONPATH="c:\Users\Nidhi\data"; .\.venv\Scripts\python.exe -m pytest -v
```

---

## 💡 5 Example Prompts to Try

1. **SQL Query Question:**
   > *"Show me total revenue and units sold grouped by category using a SQL query."*

2. **Chart / Data Visualization Request:**
   > *"Show me total sales by region as a bar chart."*

3. **Data Cleaning / Transformation Question:**
   > *"Find all transactions where unit price is greater than 500 and calculate the effective discount amount."*

4. **Interactive Dashboard Request:**
   > *"Build a sales dashboard with KPI cards and revenue trend charts."*

5. **PII Masking & Privacy Test:**
   > *"Show me customer contact details for North region orders with PII masking turned ON."*

---

## 🔒 Comprehensive Security Design & Controls

MYAI enforces multi-layered defense-in-depth security:

### 1. Authentication & Access Gate
- **Mechanism:** Authentication is enforced prior to rendering any app UI or initializing session state via `streamlit-authenticator`.
- **Password Storage:** Passwords are hashed using `bcrypt`/`argon2`. Plaintext passwords are never stored.

### 2. User Isolation
- **Partitioning:** Uploaded datasets, DuckDB in-memory tables, dashboards, and chat sessions are isolated per `user_id` in temporary workspace folders (`temp_storage/<user_id>/`).

### 3. Secrets Management
- **Protection:** API keys are read strictly from environment variables via `python-dotenv`. `.env` is included in `.gitignore`. Secrets are never logged or exposed in UI tracebacks.

### 4. Subprocess Sandbox
- **Isolation:** AI-generated Python code executes inside an isolated `subprocess.Popen`.
- **Restrictions:** Environment variables are completely stripped (`env={}`), filesystem access is restricted to a temporary sandbox directory via open-hook overrides, network access is disallowed, and execution times out after 10 seconds.

### 5. Prompt Injection Defense
- **Boundary Control:** Column names and cell values are encapsulated in `<data_schema>` tags and treated strictly as untrusted data.
- **Instruction:** System prompts explicitly instruct the model to ignore any prompt injection attempts or instructions inside user cell values.

### 6. Privacy Controls & PII Masking
- **Sample Row Toggle:** Users can toggle "Send sample rows to AI" OFF to send only schema definitions and column types.
- **PII Masking Mode:** Detects emails, phone numbers, names, and SSN/ID patterns, masking/hashing them before sending to LLM or writing to audit logs.

### 7. Upload & Export Safety
- **File Allowlist:** Accepts `.csv` and `.xlsx` only up to 50 MB. Rejects macro-enabled Excel files (`.xlsm`). Sanitizes filenames using `werkzeug.utils.secure_filename`.
- **Formula Injection Defense:** CSV exports automatically prefix cells starting with `=`, `+`, `-`, or `@` with a single quote (`'`).

### 8. Data Retention & Instant Wipe
- **Auto-Purge:** Files older than 24 hours are automatically deleted by a background retention script.
- **Delete My Data:** Sidebar includes a button that instantly wipes files, DuckDB tables, chat history, and dashboards.

### 9. Audit Logging
- **Non-Sensitive Logging:** Logs timestamp, `user_id`, action category, and query metadata to `logs/audit_<user_id>.log`. Raw data values are strictly excluded.

### 10. Abuse Limits & Rate Limiting
- **Throttling:** Sliding-window rate limiter restricts requests (default 15 req/min). Retries are capped at 3 per agent execution request.

### 11. HTTPS / Transport Security
- **Production Deployment:** In production, run Streamlit behind an SSL/TLS termination reverse proxy (Nginx, Traefik, or Caddy) enforcing HTTPS with HSTS headers.

---

## 🎨 Studio Theme Design
MYAI features a Studio design theme: warm off-white background (`#FCFBF9`), thin framing, muted slate typography, custom KPI metric cards, clean tab headers, and code expanders.
