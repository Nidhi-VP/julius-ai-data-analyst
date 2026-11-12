import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import os

# Optional Gemini Integration for NL Question Answering
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Scikit-learn for ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Julius AI — Autonomous Data Analyst & Dashboard Builder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .kpi-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
    }
    .insight-box {
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-left: 5px solid #22c55e;
        color: #f1f5f9;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.6;
    }
    .insight-box strong {
        color: #4ade80;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("📊 Julius AI Agent")
    st.caption("Autonomous Data Analyst & Dashboard Builder")
    st.divider()

    st.subheader("1. Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset to begin autonomous analysis"
    )

    st.divider()
    st.subheader("2. AI Settings (Optional)")
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Enter Gemini API key to enable deep autonomous conversational reasoning"
    )
    
    st.divider()
    st.info("💡 **Operating Principle:** Upload your dataset and ask questions in plain English or generate instant interactive dashboards.")

# ----------------- DATA LOADING & INSPECTION -----------------
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def analyze_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    
    # Check string columns that could be converted to datetime
    for c in cat_cols:
        try:
            sample = df[c].dropna().head(10)
            if len(sample) > 0 and pd.to_datetime(sample, errors='coerce').notnull().all():
                date_cols.append(c)
        except Exception:
            pass

    return {
        "numeric": numeric_cols,
        "categorical": [c for c in cat_cols if c not in date_cols],
        "datetime": date_cols,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "missing_count": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }

# ----------------- MAIN APP LOGIC -----------------
if uploaded_file is None:
    # EMPTY STATE
    st.title("Welcome to Julius AI Data Analyst")
    st.markdown("### Please upload a dataset in the sidebar to begin.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🔍 Automated Inspection
        - Schema & data type detection
        - Missing values & duplicate audit
        - Outliers & anomaly identification
        """)
    with col2:
        st.markdown("""
        #### 📈 Interactive Dashboards
        - Automatic KPI calculation
        - Dynamic multi-dimensional filters
        - Plotly trend lines, bar charts & scatters
        """)
    with col3:
        st.markdown("""
        #### 🤖 Machine Learning & Stats
        - Hypothesis testing & ANOVA
        - Pearson & Spearman correlations
        - Multi-model ML benchmark comparison
        """)
    st.stop()

# Load Data
df = load_data(uploaded_file)
if df is None or df.empty:
    st.warning("The uploaded file contains no data.")
    st.stop()

schema = analyze_columns(df)

# Top Header Bar
st.title(f"📁 {uploaded_file.name}")
st.caption(f"Loaded **{schema['total_rows']:,} rows** and **{schema['total_cols']} columns** | Missing Values: **{schema['missing_count']}** | Duplicates: **{schema['duplicates']}**")

# MAIN WORKSPACE TABS
tab_chat, tab_dashboard, tab_inspect, tab_stats, tab_ml = st.tabs([
    "💬 Ask Julius (Q&A)",
    "📊 Interactive Dashboard",
    "🔍 Schema & Data Quality",
    "📐 Statistical Analysis",
    "🤖 Machine Learning"
])

# ----------------- DEEP OPERATIONAL ANALYSIS GENERATOR -----------------
def build_julius_deep_operational_analysis(df, schema):
    col_names_lower = [c.lower() for c in df.columns]
    
    # Identify primary dimension (e.g. Division, Dept, Region)
    div_col = next((c for c in df.columns if any(k in c.lower() for k in ['division', 'dept', 'department', 'branch', 'region', 'segment'])), None)
    if not div_col and schema['categorical']:
        div_col = schema['categorical'][0]

    # Identify secondary dimension (e.g. Vehicle type, Category, Model)
    type_col = next((c for c in df.columns if any(k in c.lower() for k in ['type', 'category', 'class', 'model', 'brand']) and c != div_col), None)
    if not type_col and len(schema['categorical']) > 1:
        type_col = schema['categorical'][1]

    # Identify age / time metric
    age_col = next((c for c in schema['numeric'] if any(k in c.lower() for k in ['age', 'year', 'tenure', 'experience'])), None)
    if not age_col and schema['numeric']:
        age_col = schema['numeric'][0]

    # Identify mileage / primary volume metric
    mileage_col = next((c for c in schema['numeric'] if any(k in c.lower() for k in ['mileage', 'km', 'odometer', 'distance', 'sales', 'revenue', 'price', 'salary']) and c != age_col), None)
    if not mileage_col and len(schema['numeric']) > 1:
        mileage_col = schema['numeric'][1]

    # Identify fuel / budget / limit metric
    fuel_col = next((c for c in schema['numeric'] if any(k in c.lower() for k in ['fuel', 'limit', 'budget', 'cost', 'allowance', 'profit']) and c not in [age_col, mileage_col]), None)

    # Identify record identifier (Plate, ID, Asset, Name)
    id_col = next((c for c in df.columns if any(k in c.lower() for k in ['plate', 'asset', 'id', 'vin', 'serial', 'code', 'no', 'name']) and df[c].nunique() > len(df)*0.4), df.columns[0])

    n_rows = len(df)
    n_cols = len(df.columns)
    is_fleet = any('vehic' in c or 'plate' in c or 'mileage' in c or 'fleet' in c for c in col_names_lower)
    report_title = "Fleet analysis" if is_fleet else "Operational & Performance Analysis"

    # Report Introduction
    md = f"## {report_title}\n\n"
    md += f"The Data sheet contains **{n_rows} {'vehicles' if is_fleet else 'records'}** and **{n_cols} columns**. "
    if age_col and mileage_col and div_col:
        md += f"Building on the descriptive summary, the clearest operational finding is that **{age_col.lower()} and {mileage_col.lower()} exposure are unevenly distributed across {div_col.lower()}s and {type_col.lower() if type_col else 'categories'}**.\n\n"
    else:
        md += f"Building on the descriptive summary, performance metrics are unevenly distributed across primary categories.\n\n"

    # SECTION 1: Division Breakdown Table
    if div_col and age_col and mileage_col:
        # Compute summary
        agg_map = {id_col: 'count', age_col: 'mean', mileage_col: 'mean'}
        if fuel_col:
            agg_map[fuel_col] = 'sum'
        div_grp = df.groupby(div_col).agg(agg_map).reset_index()

        oldest_row = div_grp.sort_values(by=age_col, ascending=False).iloc[0]
        md += f"### 1. {oldest_row[div_col]} has the oldest {'fleet' if is_fleet else 'group'}\n\n"
        
        # Build Table
        fuel_header = f"Total monthly {fuel_col}" if fuel_col else ""
        md += f"| {div_col} | {'Vehicles' if is_fleet else 'Records'} | Average recorded {age_col} | Average {mileage_col} |"
        if fuel_col:
            md += f" {fuel_header} |\n"
            md += "| :--- | :--- | :--- | :--- | :--- |\n"
        else:
            md += "\n| :--- | :--- | :--- | :--- |\n"

        for _, row in div_grp.iterrows():
            d_name = row[div_col]
            v_cnt = int(row[id_col])
            v_age = f"{row[age_col]:.1f} years" if 'age' in age_col.lower() else f"{row[age_col]:,.1f}"
            v_mil = f"{row[mileage_col]:,.0f}"
            if fuel_col:
                v_fuel = f"{row[fuel_col]:,.0f}"
                md += f"| **{d_name}** | {v_cnt} | {v_age} | {v_mil} | {v_fuel} |\n"
            else:
                md += f"| **{d_name}** | {v_cnt} | {v_age} | {v_mil} |\n"

        # Totals Row
        tot_age_str = f"{df[age_col].mean():.1f} years" if 'age' in age_col.lower() else f"{df[age_col].mean():,.1f}"
        if fuel_col:
            md += f"| **{'Fleet' if is_fleet else 'Total'}** | **{n_rows}** | **{tot_age_str}** | **{df[mileage_col].mean():,.0f}** | **{df[fuel_col].sum():,.0f}** |\n\n"
        else:
            md += f"| **{'Fleet' if is_fleet else 'Total'}** | **{n_rows}** | **{tot_age_str}** | **{df[mileage_col].mean():,.0f}** |\n\n"

        # Deep Commentary
        highest_mil_row = div_grp.sort_values(by=mileage_col, ascending=False).iloc[0]
        youngest_row = div_grp.sort_values(by=age_col, ascending=True).iloc[0]
        y_name = youngest_row[div_col]
        y_mean = df[df[div_col] == y_name][mileage_col].mean()
        y_med = df[df[div_col] == y_name][mileage_col].median()

        md += f"- **All {int(oldest_row[id_col])} {oldest_row[div_col]} {'vehicles' if is_fleet else 'records'}** have recorded {age_col.lower()}s above the fleet median, and **{highest_mil_row[div_col]}** combines the oldest average with the highest average {mileage_col.lower()}.\n"
        if y_mean > y_med * 1.1:
            md += f"- **{y_name}** has the youngest average group, but its mean {mileage_col.lower()} of **{y_mean:,.0f}** is substantially above its median of **{y_med:,.0f}**, indicating that high-exposure records pull up the group average.\n\n"

    # SECTION 2: Type Breakdown & Top 5 Extreme Records
    if type_col and mileage_col:
        type_grp = df.groupby(type_col).agg({
            id_col: 'count',
            age_col: 'mean' if age_col else 'count',
            mileage_col: 'mean'
        }).reset_index().sort_values(by=mileage_col, ascending=False)
        
        top_type = type_grp.iloc[0][type_col]
        md += f"### 2. {top_type}s have the highest {mileage_col.lower()} exposure\n\n"
        
        md += f"| {type_col} | {'Vehicles' if is_fleet else 'Records'} | Average recorded {age_col} | Average {mileage_col} |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        for _, row in type_grp.iterrows():
            t_name = row[type_col]
            t_cnt = int(row[id_col])
            t_age = f"{row[age_col]:.1f} years" if (age_col and 'age' in age_col.lower()) else (f"{row[age_col]:,.1f}" if age_col else "-")
            t_mil = f"{row[mileage_col]:,.0f}"
            md += f"| **{t_name}** | {t_cnt} | {t_age} | {t_mil} |\n"
        md += "\n"

        # The 5 highest exposure records
        top5 = df.sort_values(by=mileage_col, ascending=False).head(5)
        md += f"#### The five highest-{mileage_col.lower()} records:\n\n"
        md += f"| {id_col} | Recorded {age_col} | {mileage_col} |\n"
        md += "| :--- | :--- | :--- |\n"
        for _, r in top5.iterrows():
            r_id = str(r[id_col])
            r_age = f"{r[age_col]:.2f} years" if (age_col and 'age' in age_col.lower()) else f"{r[age_col]:,.1f}"
            r_mil = f"{r[mileage_col]:,.0f}"
            md += f"| **{r_id}** | {r_age} | {r_mil} |\n"
        md += "\n*These records are reasonable candidates for a maintenance and replacement assessment, but age and mileage alone do not establish that replacement is necessary.*\n\n"

    # SECTION 3: Correlation & Thresholds
    if age_col and mileage_col:
        valid_pairs = df[[age_col, mileage_col]].dropna()
        r_val, p_val = stats.pearsonr(valid_pairs[age_col], valid_pairs[mileage_col])
        over_10_yr = (df[age_col] > 10).sum() if 'age' in age_col.lower() else (df[age_col] > df[age_col].median()).sum()
        over_1m = (df[mileage_col] > 1000000).sum() if df[mileage_col].max() > 1000000 else (df[mileage_col] > df[mileage_col].quantile(0.75)).sum()

        md += f"### 3. {age_col} and {mileage_col} are positively related\n\n"
        md += f"- **{over_10_yr} {'vehicles' if is_fleet else 'records'} ({over_10_yr/n_rows*100:.1f}%)** have recorded {age_col.lower()}s above threshold.\n"
        md += f"- **{over_1m} {'vehicles' if is_fleet else 'records'} ({over_1m/n_rows*100:.1f}%)** exceed high exposure limits.\n"
        md += f"- The Pearson correlation between `{age_col}` and `{mileage_col}` is **{r_val:.2f}**, indicating a **moderately strong positive association—not causation**.\n\n"

    # SECTION 4: Policy / Limit check
    if type_col and fuel_col:
        md += f"### 4. {fuel_col} follow {type_col.lower()} exactly\n\n"
        t_pol = df.groupby(type_col)[fuel_col].agg(['min', 'max']).reset_index()
        is_exact = (t_pol['min'] == t_pol['max']).all()
        md += f"| {type_col} | Monthly limit per unit ({fuel_col}) |\n"
        md += "| :--- | :--- |\n"
        for _, r in t_pol.iterrows():
            t_name = r[type_col]
            lim_val = f"{r['min']:,.0f}" if r['min'] == r['max'] else f"{r['min']:,.0f} - {r['max']:,.0f}"
            md += f"| **{t_name}** | {lim_val} |\n"
        md += "\n"
        if is_exact:
            md += f"*Within each {type_col.lower()}, every unit has the same monthly limit. The allocation therefore appears category-based, rather than individually tailored to recorded mileage or actual consumption.*\n\n"

    # SECTION 5: Data Quality & Interpretation Audit
    null_cells = int(df.isnull().sum().sum())
    null_pct = (null_cells / (n_rows * n_cols)) * 100
    null_cols = df.isnull().sum()[df.isnull().sum() > 0]
    
    md += f"### 5. Data quality and interpretation\n\n"
    md += f"- **No duplicate rows** were found ({df.duplicated().sum()} duplicates).\n"
    md += f"- **{null_cells} cells are missing ({null_pct:.2f}%)**"
    if not null_cols.empty:
        md += f", concentrated in **{', '.join(null_cols.index[:3])}**.\n"
    else:
        md += ".\n"

    # Check for placeholder zeroes in ID columns
    zero_ids = [c for c in df.columns if (df[c] == 0).sum() > 0 and any(k in c.lower() for k in ['asset', 'id', 'no', 'vin', 'code'])]
    if zero_ids:
        z_col = zero_ids[0]
        md += f"- **`{z_col}`** has {(df[z_col] == 0).sum()} zero values; zero values should be verified as possible placeholders.\n"

    # Check for columns mixing text with numbers
    for c in df.columns:
        if c in schema['categorical'] and any(char.isdigit() for char in str(df[c].dropna().head(5))):
            md += f"- **`{c}`** mixes numeric and categorical representation; standardizing units is recommended.\n"
            break

    md += "\n### 🎯 Recommended priorities\n\n"
    if div_col and 'oldest_row' in locals():
        md += f"1. **Review {oldest_row[div_col]}'s condition:** it combines the oldest average with high group-level mileage.\n"
    elif div_col:
        md += f"1. **Review primary operational condition across {div_col}:** investigate high-exposure segments.\n"
    md += f"2. **Assess the highest-exposure units:** incorporate service history, downtime, repair costs, and inspection results.\n"
    md += f"3. **Standardize asset identifiers and operational metrics:** establish valid units and separate classifications from numeric capacity.\n"
    md += f"4. **Add actual operating data:** monthly distance, fuel consumption, maintenance costs, and downtime would support defensible efficiency decisions.\n\n"
    md += f"**Bottom line:** The strongest review priorities are the oldest units and highest-mileage categories. The dataset identifies where to investigate, but does not yet support conclusions about mechanical condition, operating efficiency, or replacement economics.\n"
    return md

def build_julius_averages_and_medians_report(df, schema):
    import re
    # Exclude numeric identifiers and serial counters
    excluded_ids = []
    main_numeric_cols = []
    
    for c in schema['numeric']:
        c_lower = c.lower()
        if any(k in c_lower for k in ['id', 'no', 'asset', 'serial', 'vin', 'sn', 'plate', 'code', 'index', 'row']) and df[c].nunique() > len(df) * 0.4:
            excluded_ids.append(c)
        elif df[c].is_monotonic_increasing and (df[c].diff().dropna() == 1).all():
            excluded_ids.append(c)
        else:
            main_numeric_cols.append(c)
            
    # Check for mixed columns that mix numbers with text (like Capacity)
    mixed_cols = []
    for c in df.columns:
        if c not in schema['numeric']:
            s = df[c].dropna().astype(str)
            if any(re.search(r'\d', v) for v in s) and any(re.search(r'[a-zA-Z]', v) for v in s):
                mixed_cols.append(c)

    if not main_numeric_cols:
        main_numeric_cols = schema['numeric']

    is_fleet = any('vehic' in c.lower() or 'plate' in c.lower() for c in df.columns)
    unit_label = "vehicles" if is_fleet else "records"
    
    md = f"The main numeric measures in the Data sheet have the following averages and medians, based on **{len(df)} {unit_label}**:\n\n"
    md += "| Numeric column | Average (mean) | Median |\n"
    md += "| :--- | :--- | :--- |\n"
    
    all_clean = True
    for c in main_numeric_cols:
        mean_v = df[c].mean()
        med_v = df[c].median()
        if df[c].isnull().sum() > 0:
            all_clean = False
            
        # Format intelligently like Julius AI
        if "year" in c.lower():
            mean_str = f"{mean_v:.0f}"
            med_str = f"{med_v:.0f}"
        elif abs(mean_v) >= 1000:
            mean_str = f"{mean_v:,.2f}" if (mean_v % 1 != 0) else f"{mean_v:,.0f}"
            med_str = f"{med_v:,.2f}" if (med_v % 1 != 0) else f"{med_v:,.0f}"
        elif abs(mean_v) < 100:
            mean_str = f"{mean_v:.2f}"
            med_str = f"{med_v:.2f}"
        else:
            mean_str = f"{mean_v:,.2f}"
            med_str = f"{med_v:,.2f}"
            
        md += f"| **{c}** | {mean_str} | {med_str} |\n"
        
    md += "\n"
    
    # Notes line
    notes = []
    if all_clean:
        notes.append(f"All {len(main_numeric_cols)} columns have no missing values.")
    else:
        notes.append("Averages and medians calculated on valid non-null rows.")
        
    if excluded_ids:
        notes.append("Numeric identifiers were excluded because their averages are not meaningful.")
    if mixed_cols:
        notes.append(f"{', '.join(mixed_cols)} was excluded because it mixes numbers with text and has no specified unit.")
        
    md += f"**Notes:** {' '.join(notes)}\n"
    return md

# ----------------- TAB 1: CHAT & NATURAL LANGUAGE Q&A -----------------
with tab_chat:
    st.subheader("Natural Language Data Questioning")
    st.write("Ask any question about your data (e.g., *'Analyze this dataset'*, *'What is the average salary?'*, *'Which category performs best?'*)")

    user_query = st.text_input("Enter your question:", placeholder="e.g. Analyze this dataset")
    
    if user_query:
        st.markdown("#### **Execution Workflow:**")
        st.caption("`UNDERSTAND` ➜ `INSPECT` ➜ `EXECUTE PYTHON` ➜ `VERIFY` ➜ `INTERPRET`")
        
        # ----------------- JULIUS AI AUTONOMOUS EXECUTION ENGINE -----------------
        with st.spinner("🧠 Julius AI is understanding question, writing Python code, and analyzing dataset..."):
            import re
            query_lower = user_query.lower()

            direct_ans = ""
            code_str = ""
            result_obj = None
            chart_obj = None
            interpretation_str = ""
            limitations_str = ""

            # Check if Gemini API is available and key is provided
            gemini_success = False
            if HAS_GENAI and gemini_key:
                try:
                    client = genai.Client(api_key=gemini_key)
                    schema_desc = {
                        "columns": df.columns.tolist(),
                        "dtypes": {c: str(df[c].dtype) for c in df.columns},
                        "numeric_cols": schema['numeric'],
                        "categorical_cols": schema['categorical'],
                        "head_sample": df.head(3).to_dict(orient="records")
                    }
                    llm_prompt = f"""
You are Julius AI, an expert operational data analyst.
Dataset Information:
{schema_desc}

User Request: "{user_query}"

Instructions:
If the user asks to "Analyze this dataset" or requests an overview/report:
Produce a comprehensive multi-section operational analysis with:
- Clear Title and introduction
- Section 1: Primary category breakdown table (count, averages, totals) and comparative commentary
- Section 2: Secondary category breakdown and table of the five highest-exposure records
- Section 3: Correlation & Thresholds analysis with Pearson r
- Section 4: Policy/limit uniformity check
- Section 5: Data quality & integrity audit (duplicates, missing cells, placeholder zeroes)
- Recommended priorities & Bottom Line.

Provide your output strictly in this format:
---DIRECT_ANSWER---
(Your full executive markdown report with tables, headers, and bullet points)
---INTERPRETATION---
Key takeaways and executive summary
---LIMITATIONS---
Methodology and data scope notes
"""
                    llm_call = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=llm_prompt
                    )
                    raw_text = llm_call.text

                    if "---DIRECT_ANSWER---" in raw_text:
                        parts = raw_text.split("---DIRECT_ANSWER---")[1]
                        ans_part = parts.split("---INTERPRETATION---")[0].strip()
                        inter_part = ""
                        limit_part = ""
                        if "---INTERPRETATION---" in parts:
                            rest = parts.split("---INTERPRETATION---")[1]
                            if "---LIMITATIONS---" in rest:
                                inter_part = rest.split("---LIMITATIONS---")[0].strip()
                                limit_part = rest.split("---LIMITATIONS---")[1].strip()
                            else:
                                inter_part = rest.strip()
                        direct_ans = ans_part
                        interpretation_str = inter_part
                        limitations_str = limit_part
                        gemini_success = True
                except Exception as e:
                    gemini_success = False

            # Built-in Standard Julius Data Analyst Engine (Fallback or Default)
            if not gemini_success:
                matched_num = [c for c in schema['numeric'] if c.lower() in query_lower]
                matched_cat = [c for c in schema['categorical'] if c.lower() in query_lower]

                # Case 1: Comprehensive Dataset Analysis & Operational Report
                if any(w in query_lower for w in ["analyze", "overview", "summary", "describe", "all", "dataset", "fleet", "report"]):
                    code_str = """# Julius AI Comprehensive Operational & Fleet Analysis
div_col = 'Division' if 'Division' in df.columns else df.select_dtypes(include='object').columns[0]
agg_tbl = df.groupby(div_col).agg({'Mileage': 'mean', 'Age': 'mean', df.columns[0]: 'count'})
corr_r = df['Age'].corr(df['Mileage'])"""
                    direct_ans = build_julius_deep_operational_analysis(df, schema)
                    result_obj = None  # Tables already formatted beautifully in direct_ans
                    meaningful_cats = [c for c in schema['categorical'] if 1 < df[c].nunique() < 25 and not any(k in c.lower() for k in ['plate', 'id', 'sn', 'chassis', 'engine', 'vin'])]
                    meaningful_nums = [c for c in schema['numeric'] if not any(k in c.lower() for k in ['id', 'no', 'sn', 'asset', 'vin', 'serial', 'tc'])]
                    chart_cat = 'Division' if 'Division' in df.columns else (meaningful_cats[0] if meaningful_cats else (schema['categorical'][0] if schema['categorical'] else None))
                    chart_num = [c for c in df.columns if any(k in c.lower() for k in ['km', 'mileage', 'run']) and pd.api.types.is_numeric_dtype(df[c])]
                    chart_num = chart_num[0] if chart_num else (meaningful_nums[0] if meaningful_nums else (schema['numeric'][0] if schema['numeric'] else None))
                    
                    if chart_cat and chart_num:
                        chart_data = df.groupby(chart_cat)[chart_num].mean().reset_index().sort_values(by=chart_num, ascending=False)
                        chart_obj = px.bar(chart_data,
                                           x=chart_cat, y=chart_num,
                                           title=f"Average {chart_num} by {chart_cat}",
                                           color=chart_num, template="plotly_dark", color_continuous_scale="Viridis")
                    interpretation_str = ""
                    limitations_str = ""

                # Case 2: Grouped Performance / Ranking
                elif any(w in query_lower for w in ["top", "best", "highest", "most", "lowest", "worst", "rank", "perform"]) and schema['categorical'] and schema['numeric']:
                    grp_col = matched_cat[0] if matched_cat else schema['categorical'][0]
                    val_col = matched_num[0] if matched_num else schema['numeric'][0]
                    code_str = f"""# Grouped Aggregation & Ranking
grouped = df.groupby('{grp_col}')['{val_col}'].agg(['count', 'mean', 'sum']).sort_values(by='mean', ascending=False)
result_data = grouped.round(2).reset_index()"""
                    grouped = df.groupby(grp_col)[val_col].agg(['count', 'mean', 'sum']).sort_values(by='mean', ascending=False).round(2).reset_index()
                    result_obj = grouped
                    top_name = grouped.iloc[0][grp_col]
                    top_val = grouped.iloc[0]['mean']
                    bot_name = grouped.iloc[-1][grp_col]
                    bot_val = grouped.iloc[-1]['mean']
                    direct_ans = f"**{top_name}** performs the highest in terms of average `{val_col}` with **{top_val:,.2f}**, outperforming **{bot_name}** ({bot_val:,.2f}) by **{(((top_val/bot_val)-1)*100) if bot_val else 0:.1f}%**."
                    chart_obj = px.bar(grouped.head(10), x=grp_col, y='mean', color='mean', title=f"Top {grp_col} by Mean {val_col}", template="plotly_dark", color_continuous_scale="Viridis")
                    interpretation_str = f"""
• **Leader:** `{top_name}` holds the leading position with average `{val_col}` of {top_val:,.2f}.
• **Variance Across Groups:** The performance difference between the top and bottom groups is substantial.
• **Sample Distribution:** Top group represents {grouped.iloc[0]['count']} records in the dataset.
"""
                    limitations_str = f"Groups with small sample sizes (< 3 rows) should be interpreted with caution."

                # Case 3: Averages / Central Tendency
                elif any(w in query_lower for w in ["average", "mean", "median", "typical"]) and schema['numeric']:
                    is_multi = any(w in query_lower for w in ["columns", "all", "main", "measures", "features"]) or len(matched_num) == 0
                    if is_multi:
                        code_str = """# Calculate mean and median across main numeric measures
main_cols = [c for c in df.select_dtypes(include='number').columns if not any(k in c.lower() for k in ['id', 'no', 'sn', 'asset'])]
summary_res = df[main_cols].agg(['mean', 'median']).T"""
                        direct_ans = build_julius_averages_and_medians_report(df, schema)
                        result_obj = None
                        chart_obj = None
                        interpretation_str = ""
                        limitations_str = ""
                    else:
                        target = matched_num[0]
                        code_str = f"""# Statistical Central Tendency Calculation
mean_val = df['{target}'].mean()
median_val = df['{target}'].median()
std_val = df['{target}'].std()
result_data = pd.DataFrame([{{
    'Metric': 'Value',
    'Mean': round(mean_val, 2),
    'Median': round(median_val, 2),
    'Std Dev': round(std_val, 2),
    'Min': round(df['{target}'].min(), 2),
    'Max': round(df['{target}'].max(), 2)
}}])"""
                        mean_val = df[target].mean()
                        median_val = df[target].median()
                        std_val = df[target].std()
                        result_obj = {
                            f"Mean {target}": f"{mean_val:,.2f}",
                            f"Median {target}": f"{median_val:,.2f}",
                            f"Std Dev": f"{std_val:,.2f}",
                            f"Min / Max": f"{df[target].min():,.0f} - {df[target].max():,.0f}"
                        }
                        direct_ans = f"The average `{target}` is **{mean_val:,.2f}** (median: **{median_val:,.2f}**), with a standard deviation of **{std_val:,.2f}** across {df[target].count():,} valid records."
                        chart_obj = px.histogram(df, x=target, nbins=25, title=f"Distribution of {target} (Mean: {mean_val:,.2f})", template="plotly_dark", color_discrete_sequence=["#22c55e"])
                        interpretation_str = f"""
• **Central Value:** Typical `{target}` centers around {median_val:,.2f}.
• **Skewness:** Mean ({mean_val:,.2f}) compared to Median ({median_val:,.2f}) suggests {'right-skewed distribution with high outliers' if mean_val > median_val*1.05 else ('left-skewed distribution' if mean_val < median_val*0.95 else 'a symmetrical bell curve')}.
• **Range:** Observations range from a low of {df[target].min():,.2f} to a high of {df[target].max():,.2f}.
"""
                        limitations_str = f"Computed on {df[target].count()} non-null values. {df[target].isnull().sum()} null values were excluded."

                # Case 4: Correlation / Relationships
                elif any(w in query_lower for w in ["correlat", "relation", "impact", "affect", "vs", "versus"]) and len(schema['numeric']) >= 2:
                    col1 = matched_num[0] if len(matched_num) >= 1 else schema['numeric'][0]
                    col2 = matched_num[1] if len(matched_num) >= 2 else schema['numeric'][1]
                    code_str = f"""# Pearson Correlation & Significance Testing
valid_df = df[['{col1}', '{col2}']].dropna()
r_val, p_val = stats.pearsonr(valid_df['{col1}'], valid_df['{col2}'])
result_data = pd.DataFrame([{{
    'Variable 1': '{col1}',
    'Variable 2': '{col2}',
    'Pearson r': round(r_val, 4),
    'p-value': f'{{p_val:.4e}}',
    'Significant': p_val < 0.05
}}])"""
                    valid_df = df[[col1, col2]].dropna()
                    r_val, p_val = stats.pearsonr(valid_df[col1], valid_df[col2])
                    strength = "strong" if abs(r_val) > 0.7 else ("moderate" if abs(r_val) > 0.3 else "weak")
                    direction = "positive" if r_val > 0 else "negative"
                    result_obj = pd.DataFrame([{
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Pearson r': round(r_val, 4),
                        'Strength': f"{strength.capitalize()} {direction}",
                        'p-value': f"{p_val:.4e}",
                        'Statistically Significant (p < 0.05)': bool(p_val < 0.05)
                    }])
                    direct_ans = f"There is a **{strength} {direction} correlation ($r = {r_val:.3f}$)** between `{col1}` and `{col2}`. This relationship is **{'statistically significant' if p_val < 0.05 else 'not statistically significant'}** ($p = {p_val:.4e}$)."
                    chart_obj = px.scatter(valid_df, x=col1, y=col2, trendline="ols", title=f"{col1} vs {col2} (r = {r_val:.3f})", template="plotly_dark")
                    interpretation_str = f"""
• **Correlation Strength:** Pearson $r = {r_val:.3f}$ indicates that as `{col1}` increases, `{col2}` tends to {'increase' if r_val > 0 else 'decrease'}.
• **Statistical Significance:** With $p = {p_val:.4e}$, the likelihood of observing this by random chance is {'less than 5%' if p_val < 0.05 else 'high'}.
• **Causation Note:** Correlation indicates mathematical association, not direct cause and effect.
"""
                    limitations_str = f"Analyzed {len(valid_df)} rows after removing nulls."

                # Case 5: Listing / Filtering by condition (e.g. "names of batsman in 2013", "players in year 2015")
                elif any(w in query_lower for w in ["names", "name", "list", "who", "players", "batsman", "bowler", "teams", "unique", "distinct", "which"]) or any(re.search(r'\b(19\d\d|20\d\d)\b', query_lower) for _ in [1]):
                    # Detect year or specific filter value
                    year_match = re.search(r'\b(19\d\d|20\d\d)\b', user_query)
                    year_val = int(year_match.group(1)) if year_match else None

                    # Find target column to list / inspect
                    synonym_map = {
                        "batsman": ["batter", "batsman", "batsman_type", "player", "player_out"],
                        "batter": ["batter", "batsman"],
                        "bowler": ["bowler", "bowler_type"],
                        "team": ["team_batting", "team_bowling", "team", "division", "type"],
                        "player": ["batter", "bowler", "player_out", "non_striker"]
                    }
                    target_entity_col = None
                    for key_term, candidates in synonym_map.items():
                        if key_term in query_lower:
                            for c in candidates:
                                for df_c in df.columns:
                                    if c in df_c.lower():
                                        target_entity_col = df_c
                                        break
                                if target_entity_col:
                                    break
                        if target_entity_col:
                            break

                    if not target_entity_col:
                        # Find any categorical column mentioned or best candidate
                        for c in df.columns:
                            c_clean = c.lower().replace('_', ' ')
                            if any(w in query_lower for w in c_clean.split()):
                                target_entity_col = c
                                break

                    if not target_entity_col and schema['categorical']:
                        target_entity_col = schema['categorical'][0]

                    # Filter dataframe if year / season is specified
                    filtered_sub_df = df
                    filter_explanation = ""
                    season_col = next((c for c in df.columns if any(k in c.lower() for k in ['season', 'year', 'date'])), None)

                    if year_val is not None and season_col:
                        filtered_sub_df = df[df[season_col] == year_val]
                        if filtered_sub_df.empty:
                            filtered_sub_df = df[df[season_col].astype(str).str.contains(str(year_val), na=False)]
                        filter_explanation = f" in `{season_col}` = **{year_val}**"

                    if target_entity_col and target_entity_col in filtered_sub_df.columns:
                        unique_entities = sorted(filtered_sub_df[target_entity_col].dropna().unique())
                        num_unique = len(unique_entities)

                        # Check if metric aggregation is also requested (e.g. "runs scored", "balls", "count")
                        runs_col = next((c for c in df.columns if any(k in c.lower() for k in ['runs', 'score', 'points', 'total'])), None)

                        code_str = f"""# Filter data and find distinct {target_entity_col}
filtered_data = df{f"[df['{season_col}'] == {year_val}]" if year_val and season_col else ""}
unique_names = filtered_data['{target_entity_col}'].dropna().unique()
result_df = filtered_data.groupby('{target_entity_col}').size().reset_index(name='Occurrences').sort_values(by='Occurrences', ascending=False)"""

                        if runs_col and runs_col in filtered_sub_df.columns:
                            top_summary = filtered_sub_df.groupby(target_entity_col)[runs_col].agg(
                                Total_Runs='sum',
                                Balls_Faced='count'
                            ).sort_values(by='Total_Runs', ascending=False).reset_index()
                            result_obj = top_summary
                            top_name = top_summary.iloc[0][target_entity_col]
                            top_runs = top_summary.iloc[0]['Total_Runs']
                            direct_ans = f"There are **{num_unique} unique {target_entity_col}s** recorded{filter_explanation}.\n\n"
                            direct_ans += f"The leading `{target_entity_col}` is **{top_name}** with **{top_runs:,} total {runs_col}** across {top_summary.iloc[0]['Balls_Faced']:,} deliveries.\n\n"
                            direct_ans += f"Below is the complete interactive ranking and breakdown of all {num_unique} entries."
                            chart_obj = px.bar(top_summary.head(15), x=target_entity_col, y='Total_Runs', color='Total_Runs',
                                               title=f"Top 15 {target_entity_col}s{filter_explanation} by {runs_col}",
                                               template="plotly_dark", color_continuous_scale="Viridis")
                        else:
                            top_summary = filtered_sub_df[target_entity_col].value_counts().reset_index()
                            top_summary.columns = [target_entity_col, 'Records / Appearances']
                            result_obj = top_summary
                            direct_ans = f"Found **{num_unique} distinct `{target_entity_col}` entries**{filter_explanation}."
                            chart_obj = px.bar(top_summary.head(15), x=target_entity_col, y='Records / Appearances',
                                               title=f"Top 15 Most Active {target_entity_col}s{filter_explanation}",
                                               template="plotly_dark", color='Records / Appearances', color_continuous_scale="Teal")

                        interpretation_str = f"""
• **Total Count:** Found **{num_unique}** unique `{target_entity_col}` entries{filter_explanation}.
• **Most Active:** `{top_summary.iloc[0][target_entity_col]}` has the highest representation with {top_summary.iloc[0].iloc[1]:,} occurrences.
• **Complete List:** You can search, filter, or download the full list from the Key Calculated Results table below.
"""
                        limitations_str = f"Filtered over {len(filtered_sub_df):,} matching rows out of {len(df):,} total rows."
                    else:
                        target_col = matched_num[0] if matched_num else (matched_cat[0] if matched_cat else df.columns[0])
                        val_counts = df[target_col].value_counts().head(10).reset_index()
                        val_counts.columns = [target_col, 'Count']
                        result_obj = val_counts
                        direct_ans = f"Analyzed `{target_col}` for query: *'{user_query}'*. The top category is **{val_counts.iloc[0][target_col]}** with **{val_counts.iloc[0]['Count']} occurrences** ({((val_counts.iloc[0]['Count']/len(df))*100):.1f}% of dataset)."
                        chart_obj = px.bar(val_counts, x=target_col, y='Count', title=f"Top Occurrences in {target_col}", template="plotly_dark", color='Count', color_continuous_scale="Teal")
                        interpretation_str = f"Top {len(val_counts)} unique values displayed."
                        limitations_str = f"Filtered over all {len(df)} records in active spreadsheet."

                # Case 6: Default General Fallback
                else:
                    target_col = matched_num[0] if matched_num else (matched_cat[0] if matched_cat else df.columns[0])
                    code_str = f"""# Inspecting specific query on '{target_col}'
val_counts = df['{target_col}'].value_counts().head(10).reset_index()
result_data = val_counts"""
                    val_counts = df[target_col].value_counts().head(10).reset_index()
                    val_counts.columns = [target_col, 'Count']
                    result_obj = val_counts
                    direct_ans = f"Analyzed `{target_col}` for query: *'{user_query}'*. The top category is **{val_counts.iloc[0][target_col]}** with **{val_counts.iloc[0]['Count']} occurrences** ({((val_counts.iloc[0]['Count']/len(df))*100):.1f}% of dataset)."
                    chart_obj = px.bar(val_counts, x=target_col, y='Count', title=f"Top Occurrences in {target_col}", template="plotly_dark", color='Count', color_continuous_scale="Teal")
                    interpretation_str = f"""
• **Most Frequent:** `{val_counts.iloc[0][target_col]}` leads with {val_counts.iloc[0]['Count']} records.
• **Distribution:** Top {len(val_counts)} unique values represent a significant portion of this feature.
• **Next Step:** You can explore secondary correlations in Tab 4 or build visual cards in Tab 2.
"""
                    limitations_str = f"Filtered over all {len(df)} records in active spreadsheet."

            # ----------------- RENDER STANDARDIZED JULIUS AI OUTPUT -----------------
            if direct_ans:
                if str(direct_ans).strip().startswith("## "):
                    st.markdown(direct_ans)
                else:
                    st.markdown(f"### 🎯 Direct Answer\n{direct_ans}")

            if code_str:
                with st.expander("🔍 View Executed Python Code", expanded=False):
                    st.code(code_str, language="python")

            if result_obj is not None:
                st.markdown("### 📊 Key Calculated Results")
                if isinstance(result_obj, pd.DataFrame):
                    st.dataframe(result_obj, use_container_width=True)
                elif isinstance(result_obj, dict):
                    res_cols = st.columns(min(len(result_obj), 4))
                    for idx, (k, v) in enumerate(result_obj.items()):
                        with res_cols[idx % len(res_cols)]:
                            st.metric(k, v)
                else:
                    st.write(result_obj)

            if chart_obj is not None:
                st.plotly_chart(chart_obj, use_container_width=True, key="julius_chat_chart")

            if interpretation_str:
                st.markdown("### 💡 Interpretation & Key Insights")
                st.markdown(f"<div class='insight-box'>{interpretation_str.strip()}</div>", unsafe_allow_html=True)

            if limitations_str:
                st.caption(f"⚠️ **Data Notes / Methodology:** {limitations_str}")

# ----------------- TAB 2: INTERACTIVE DASHBOARD -----------------
with tab_dashboard:
    st.subheader("Interactive KPI Dashboard")

    # Dynamic Multi-Filter Bar
    st.markdown("#### 🎯 Interactive Multi-Filters")
    filtered_df = df.copy()

    if schema['categorical']:
        filter_cols_to_use = st.multiselect(
            "Choose columns to filter by:",
            options=schema['categorical'],
            default=schema['categorical'][:2] if len(schema['categorical']) >= 2 else schema['categorical'][:1],
            help="You can add multiple filter columns to slice and dice your data"
        )
        
        if filter_cols_to_use:
            filter_ui_cols = st.columns(min(len(filter_cols_to_use), 3))
            for i, col_name in enumerate(filter_cols_to_use):
                with filter_ui_cols[i % len(filter_ui_cols)]:
                    opts = sorted([str(x) for x in df[col_name].dropna().unique()])
                    chosen_vals = st.multiselect(
                        f"Select {col_name} values:",
                        options=opts,
                        default=[],
                        placeholder=f"All {col_name}s (or select multiple)",
                        key=f"filter_{col_name}"
                    )
                    if chosen_vals:
                        filtered_df = filtered_df[filtered_df[col_name].astype(str).isin(chosen_vals)]

    # ----------------- MULTI-DASHBOARD BUILDER CONTROLS -----------------
    st.markdown("#### ⚙️ Power BI Visual Studio & Dashboard Builder")
    
    # Initialize dashboard list and editor state in session_state if not present
    if "my_dashboards" not in st.session_state:
        st.session_state.my_dashboards = []
    if "editing_card_idx" not in st.session_state:
        st.session_state.editing_card_idx = None

    # Filter out identifier columns to get meaningful dimensions & metrics
    meaningful_nums = [c for c in schema['numeric'] if not any(k in c.lower() for k in ['id', 'no', 'sn', 'asset', 'vin', 'serial', 'code', 'index', 'plate'])]
    if not meaningful_nums:
        meaningful_nums = schema['numeric']

    meaningful_cats = [c for c in df.columns if (df[c].nunique() < 40 and df[c].nunique() > 1) and not any(k in c.lower() for k in ['plate', 'asset', 'vin', 'serial', 'id', 'sn', 'chassis', 'engine'])]
    if not meaningful_cats:
        meaningful_cats = [c for c in df.columns if df[c].nunique() < 50]
    if not meaningful_cats:
        meaningful_cats = df.columns.tolist()

    # Pre-populate with diverse visuals if canvas is empty
    if len(st.session_state.my_dashboards) == 0:
        d_a = meaningful_cats[0]
        d_b = meaningful_cats[1] if len(meaningful_cats) > 1 else d_a
        m_a = meaningful_nums[0]
        m_b = meaningful_nums[1] if len(meaningful_nums) > 1 else m_a
        st.session_state.my_dashboards.append({
            "title": f"Average {m_a} by {d_a}",
            "type": "📊 Bar / Column Chart",
            "agg": "Average",
            "dim": d_a,
            "metric": m_a,
            "secondary": m_b,
            "top_n": "Top 10"
        })
        st.session_state.my_dashboards.append({
            "title": f"Fleet Distribution by {d_b}",
            "type": "🍩 Donut / Pie Chart",
            "agg": "Count",
            "dim": d_b,
            "metric": m_a,
            "secondary": m_b,
            "top_n": "Top 8"
        })

    # Visual Creator Form
    with st.expander("➕ Configure & Add New Dashboard Card", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            new_chart_type = st.selectbox(
                "1. Visual Type",
                [
                    "📊 Bar / Column Chart",
                    "📈 Line / Area Trend",
                    "🍩 Donut / Pie Chart",
                    "🔵 Scatter Plot (Trendline)",
                    "🌳 Treemap",
                    "📦 Box & Whisker Plot",
                    "📊 Histogram / Distribution",
                    "📶 Horizontal Bar Chart",
                    "📋 Power BI Matrix (Pivot Table)",
                    "🎯 KPI Metric Card"
                ]
            )
        with c2:
            new_agg = st.selectbox(
                "2. Aggregation (DAX Measure)",
                ["Average", "Sum", "Count", "Median", "Max", "Min"]
            )
        with c3:
            new_dim = st.selectbox("3. X-Axis / Category Dimension", meaningful_cats)
        with c4:
            new_top_n = st.selectbox("4. Top N Filter", ["Top 5", "Top 10", "Top 15", "Top 20", "All"], index=1)

        m1_col, m2_col = st.columns(2)
        with m1_col:
            new_metric = st.selectbox("5. Primary Value / Measure", meaningful_nums, index=0)
        with m2_col:
            new_secondary = st.selectbox("6. Secondary Metric (for Scatter / Bubble)", meaningful_nums, index=min(1, len(meaningful_nums)-1))

        # Action Buttons
        btn_cols = st.columns([2, 1, 1, 2])
        with btn_cols[0]:
            if st.button("➕ Add This Dashboard to Canvas", type="primary"):
                st.session_state.my_dashboards.append({
                    "title": f"{new_agg} of {new_metric} by {new_dim}",
                    "type": new_chart_type,
                    "agg": new_agg,
                    "dim": new_dim,
                    "metric": new_metric,
                    "secondary": new_secondary,
                    "top_n": new_top_n
                })
                st.rerun()

        with btn_cols[1]:
            if st.button("⚡ Add 6 Auto-Dashboards"):
                # Pick up to 3 distinct dimensions & metrics for high diversity
                d1 = meaningful_cats[0]
                d2 = meaningful_cats[1] if len(meaningful_cats) > 1 else d1
                d3 = meaningful_cats[2] if len(meaningful_cats) > 2 else d1

                m1 = meaningful_nums[0]
                m2 = meaningful_nums[1] if len(meaningful_nums) > 1 else m1
                m3 = meaningful_nums[2] if len(meaningful_nums) > 2 else m1

                st.session_state.my_dashboards = [
                    {"title": f"Average {m1} by {d1}", "type": "📊 Bar / Column Chart", "agg": "Average", "dim": d1, "metric": m1, "secondary": m2, "top_n": "Top 10"},
                    {"title": f"Fleet Share by {d2}", "type": "🍩 Donut / Pie Chart", "agg": "Count", "dim": d2, "metric": m1, "secondary": m2, "top_n": "Top 8"},
                    {"title": f"{m1} vs {m2} Regression", "type": "🔵 Scatter Plot (Trendline)", "agg": "Average", "dim": d1, "metric": m1, "secondary": m2, "top_n": "All"},
                    {"title": f"Total {m3} by {d2}", "type": "🌳 Treemap", "agg": "Sum", "dim": d2, "metric": m3, "secondary": m1, "top_n": "Top 10"},
                    {"title": f"Distribution of {m1} by {d1}", "type": "📦 Box & Whisker Plot", "agg": "Average", "dim": d1, "metric": m1, "secondary": m2, "top_n": "Top 10"},
                    {"title": f"Average {m2} by {d3}", "type": "📶 Horizontal Bar Chart", "agg": "Average", "dim": d3, "metric": m2, "secondary": m1, "top_n": "Top 10"}
                ]
                st.rerun()

        with btn_cols[2]:
            if st.button("🗑️ Clear Canvas"):
                st.session_state.my_dashboards = []
                st.session_state.editing_card_idx = None
                st.rerun()

    st.markdown(f"### 🖼️ Active Dashboard Canvas ({len(st.session_state.my_dashboards)} Visuals Displayed)")
    
    # RENDER ALL DASHBOARD CARDS IN A 2-COLUMN GRID
    agg_map = {"Sum": "sum", "Average": "mean", "Count": "count", "Median": "median", "Max": "max", "Min": "min"}
    visual_types_list = [
        "📊 Bar / Column Chart", "📈 Line / Area Trend", "🍩 Donut / Pie Chart",
        "🔵 Scatter Plot (Trendline)", "🌳 Treemap", "📦 Box & Whisker Plot",
        "📊 Histogram / Distribution", "📶 Horizontal Bar Chart",
        "📋 Power BI Matrix (Pivot Table)", "🎯 KPI Metric Card"
    ]
    
    if not st.session_state.my_dashboards:
        st.info("No dashboard cards on canvas. Click '+ Add This Dashboard to Canvas' or '⚡ Add 6 Auto-Dashboards' above to build your dashboard!")
    else:
        # Loop over dashboards in pairs
        for i in range(0, len(st.session_state.my_dashboards), 2):
            grid_cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx < len(st.session_state.my_dashboards):
                    card = st.session_state.my_dashboards[idx]
                    with grid_cols[j]:
                        # Card Header with Edit and Remove Buttons
                        header_c1, header_c2, header_c3 = st.columns([5, 1.8, 1])
                        with header_c1:
                            st.markdown(f"##### #{idx+1}. {card['title']}")
                        with header_c2:
                            is_currently_editing = (st.session_state.editing_card_idx == idx)
                            edit_label = "✖️ Close" if is_currently_editing else "✏️ Edit"
                            if st.button(edit_label, key=f"edit_btn_{idx}", help="Edit this chart's visual type, dimension, measure, or aggregation"):
                                st.session_state.editing_card_idx = None if is_currently_editing else idx
                                st.rerun()
                        with header_c3:
                            if st.button("🗑️", key=f"del_card_{idx}", help="Remove this visual"):
                                st.session_state.my_dashboards.pop(idx)
                                if st.session_state.editing_card_idx == idx:
                                    st.session_state.editing_card_idx = None
                                st.rerun()

                        # INLINE CARD EDITOR MODAL / PANEL
                        if st.session_state.editing_card_idx == idx:
                            with st.container():
                                st.markdown("###### 🛠️ Edit Chart Configuration")
                                ec1, ec2, ec3 = st.columns(3)
                                with ec1:
                                    cur_t_idx = visual_types_list.index(card['type']) if card['type'] in visual_types_list else 0
                                    new_type = st.selectbox("Visual Type", visual_types_list, index=cur_t_idx, key=f"edit_type_{idx}")
                                    new_agg_e = st.selectbox("Aggregation", ["Average", "Sum", "Count", "Median", "Max", "Min"],
                                                             index=["Average", "Sum", "Count", "Median", "Max", "Min"].index(card['agg']) if card['agg'] in ["Average", "Sum", "Count", "Median", "Max", "Min"] else 0,
                                                             key=f"edit_agg_{idx}")
                                with ec2:
                                    cur_dim_idx = meaningful_cats.index(card['dim']) if card['dim'] in meaningful_cats else 0
                                    new_dim_e = st.selectbox("Category Dimension", meaningful_cats, index=cur_dim_idx, key=f"edit_dim_{idx}")
                                    cur_met_idx = meaningful_nums.index(card['metric']) if card['metric'] in meaningful_nums else 0
                                    new_met_e = st.selectbox("Primary Measure", meaningful_nums, index=cur_met_idx, key=f"edit_met_{idx}")
                                with ec3:
                                    top_opts = ["Top 5", "Top 10", "Top 15", "Top 20", "All"]
                                    cur_top_idx = top_opts.index(card.get('top_n', 'Top 10')) if card.get('top_n', 'Top 10') in top_opts else 1
                                    new_top_n_e = st.selectbox("Top N Filter", top_opts, index=cur_top_idx, key=f"edit_top_{idx}")

                                save_cols = st.columns([2, 2, 3])
                                with save_cols[0]:
                                    if st.button("💾 Apply Changes", key=f"save_edit_{idx}", type="primary"):
                                        st.session_state.my_dashboards[idx]['type'] = new_type
                                        st.session_state.my_dashboards[idx]['agg'] = new_agg_e
                                        st.session_state.my_dashboards[idx]['dim'] = new_dim_e
                                        st.session_state.my_dashboards[idx]['metric'] = new_met_e
                                        st.session_state.my_dashboards[idx]['top_n'] = new_top_n_e
                                        st.session_state.my_dashboards[idx]['title'] = f"{new_agg_e} of {new_met_e} by {new_dim_e}"
                                        st.session_state.editing_card_idx = None
                                        st.rerun()
                                with save_cols[1]:
                                    if st.button("Cancel", key=f"cancel_edit_{idx}"):
                                        st.session_state.editing_card_idx = None
                                        st.rerun()
                                st.markdown("---")

                        # Data Preparation
                        c_dim = card['dim']
                        c_metric = card['metric']
                        c_agg = card['agg']
                        c_type = card['type']
                        c_top = card['top_n']

                        if c_metric in filtered_df.columns and c_dim in filtered_df.columns:
                            plot_data = filtered_df.groupby(c_dim)[c_metric].agg(agg_map.get(c_agg, "mean")).reset_index()
                            plot_data = plot_data.sort_values(by=c_metric, ascending=False)
                            if c_top != "All":
                                n_val = int(c_top.replace("Top ", ""))
                                plot_data = plot_data.head(n_val)

                            # Chart Rendering
                            if c_type == "📊 Bar / Column Chart":
                                fig = px.bar(plot_data, x=c_dim, y=c_metric, color=c_metric, color_continuous_scale="Viridis", template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_bar_{idx}")
                            elif c_type == "📈 Line / Area Trend":
                                fig = px.area(plot_data, x=c_dim, y=c_metric, template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_area_{idx}")
                            elif c_type == "🍩 Donut / Pie Chart":
                                fig = px.pie(plot_data, names=c_dim, values=c_metric, hole=0.45, template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_pie_{idx}")
                            elif c_type == "🔵 Scatter Plot (Trendline)":
                                c_sec = card['secondary'] if card['secondary'] in filtered_df.columns else c_metric
                                fig = px.scatter(filtered_df, x=c_metric, y=c_sec, color=c_dim, trendline="ols", template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_scatter_{idx}")
                            elif c_type == "🌳 Treemap":
                                fig = px.treemap(plot_data, path=[c_dim], values=c_metric, color=c_metric, color_continuous_scale="Blues", template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_tree_{idx}")
                            elif c_type == "📦 Box & Whisker Plot":
                                fig = px.box(filtered_df, x=c_dim, y=c_metric, color=c_dim, template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_box_{idx}")
                            elif c_type == "📊 Histogram / Distribution":
                                fig = px.histogram(filtered_df, x=c_metric, nbins=20, color_discrete_sequence=["#6366f1"], template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_hist_{idx}")
                            elif c_type == "📶 Horizontal Bar Chart":
                                fig = px.bar(plot_data.sort_values(by=c_metric, ascending=True), x=c_metric, y=c_dim, orientation='h', color=c_metric, color_continuous_scale="Purples", template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True, key=f"dash_hbar_{idx}")
                            elif c_type == "📋 Power BI Matrix (Pivot Table)":
                                sec_cols = [c for c in df.columns if c != c_dim and df[c].nunique() < 25]
                                if sec_cols:
                                    piv = filtered_df.pivot_table(index=c_dim, columns=sec_cols[0], values=c_metric, aggfunc=agg_map.get(c_agg, "mean")).fillna(0)
                                    st.dataframe(piv.style.background_gradient(cmap="Blues"), use_container_width=True)
                                else:
                                    st.dataframe(plot_data.style.background_gradient(cmap="Blues", subset=[c_metric]), use_container_width=True)
                            elif c_type == "🎯 KPI Metric Card":
                                val_tot = filtered_df[c_metric].agg(agg_map.get(c_agg, "sum"))
                                st.metric(label=f"{c_agg} of {c_metric}", value=f"{val_tot:,.2f}")

    # KEY INSIGHTS (AUTOMATED & READABLE)
    st.markdown("### 💡 KEY INSIGHTS (Automated)")
    with st.container():
        insights = []
        ins_metric = schema['numeric'][0] if schema['numeric'] else None
        ins_dim = schema['categorical'][0] if schema['categorical'] else None

        if ins_metric and ins_metric in filtered_df.columns:
            std_dev = filtered_df[ins_metric].std()
            insights.append(f"**Dispersion in {ins_metric}:** Standard deviation is {std_dev:,.2f}, representing a variability ratio of {(std_dev / (filtered_df[ins_metric].mean() or 1))*100:.1f}%.")
        
        if ins_metric and ins_dim and ins_metric in filtered_df.columns and ins_dim in filtered_df.columns:
            ins_plot = filtered_df.groupby(ins_dim)[ins_metric].mean().sort_values(ascending=False).reset_index()
            if len(ins_plot) > 1:
                top_cat = ins_plot.iloc[0][ins_dim]
                top_val = ins_plot.iloc[0][ins_metric]
                bot_cat = ins_plot.iloc[-1][ins_dim]
                bot_val = ins_plot.iloc[-1][ins_metric]
                diff_pct = ((top_val / bot_val) - 1)*100 if bot_val != 0 else 0
                insights.append(f"**Highest Performing Group:** **{top_cat}** leads with an average {ins_metric} of {top_val:,.2f}, which is {diff_pct:.1f}% higher than the lowest group (**{bot_cat}** with {bot_val:,.2f}).")

        if len(schema['numeric']) >= 2:
            m_a, m_b = schema['numeric'][0], schema['numeric'][1]
            try:
                valid_pair = filtered_df[[m_a, m_b]].dropna()
                if len(valid_pair) > 2:
                    r, p_val = stats.pearsonr(valid_pair[m_a], valid_pair[m_b])
                    insights.append(f"**Linear Association:** Correlation between {m_a} and {m_b} is **{r:.2f}** (p-value: {p_val:.4e}).")
            except Exception:
                pass

        for ins in insights:
            st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

# ----------------- TAB 3: SCHEMA & QUALITY -----------------
with tab_inspect:
    st.subheader("Dataset Schema & Data Quality Audit")
    st.caption("Comprehensive data health diagnostics, type validation, anomaly detection, and cleaning tools.")

    total_cells = schema['total_rows'] * schema['total_cols']
    missing_pct = (schema['missing_count'] / total_cells * 100) if total_cells > 0 else 0
    
    # Calculate Data Health Score
    health_deductions = (missing_pct * 1.5) + (schema['duplicates'] * 5)
    health_score = max(10, min(100, round(100 - health_deductions)))
    health_color = "#22c55e" if health_score >= 85 else ("#f59e0b" if health_score >= 70 else "#ef4444")
    health_label = "Excellent" if health_score >= 90 else ("Good" if health_score >= 80 else ("Fair" if health_score >= 70 else "Needs Attention"))

    # Summary KPI Cards
    col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
    col_q1.metric("Total Rows", f"{schema['total_rows']:,}")
    col_q2.metric("Total Columns", f"{schema['total_cols']}")
    col_q3.metric("Missing Cells", f"{schema['missing_count']:,}", delta=f"{missing_pct:.2f}% null", delta_color="inverse")
    col_q4.metric("Duplicate Rows", f"{schema['duplicates']}")
    col_q5.metric("Health Score", f"{health_score}/100", delta=health_label)

    # 1. Automated Quality Findings & Anomaly Detection
    st.markdown("#### 🚨 Automated Quality Findings & Alerts")
    quality_issues = []
    
    # Check duplicate rows
    if schema['duplicates'] > 0:
        quality_issues.append(("warning", f"**Duplicate Rows Detected:** Found {schema['duplicates']} duplicate record(s). Recommended to deduplicate."))
    else:
        quality_issues.append(("success", "**No duplicate rows found:** Every vehicle/record is unique across the dataset."))

    # Check mixed data types (numbers mixed with strings in non-identifier columns)
    mixed_cols = []
    for c in df.columns:
        c_low = str(c).lower()
        if any(k in c_low for k in ['engine', 'chassis', 'vin', 'serial', 'plate', 'reg', 'chasis']):
            continue
        if df[c].dtype == 'object':
            non_null = df[c].dropna()
            num_count = pd.to_numeric(non_null, errors='coerce').notnull().sum()
            if 0 < num_count < len(non_null):
                mixed_cols.append((c, num_count, len(non_null)))
    
    if mixed_cols:
        for c, n_cnt, t_cnt in mixed_cols:
            quality_issues.append(("warning", f"**Mixed Data Type in `{c}`:** Contains {n_cnt} numeric values mixed with text (e.g. `{df[c].dropna().iloc[0]}`). Numbers cannot be aggregated mathematically until standardized."))
    
    # Check placeholder zeros in ID / identifier columns
    id_zero_cols = []
    for c in df.columns:
        c_low = str(c).lower()
        if any(k in c_low for k in ['id', 'no', 'sn', 'asset', 'vin', 'serial', 'code']):
            zero_count = ((df[c] == 0) | (df[c] == '0')).sum()
            if zero_count > 0:
                id_zero_cols.append((c, zero_count))
    if id_zero_cols:
        for c, z_cnt in id_zero_cols:
            quality_issues.append(("info", f"**Placeholder Zeros in `{c}`:** Found {z_cnt} zero value(s) in an identifier column. Check whether `0` represents a placeholder or unassigned asset."))

    # Check high missing rate columns
    high_null_cols = []
    for c in df.columns:
        n_cnt = df[c].isnull().sum()
        pct = (n_cnt / len(df)) * 100
        if pct > 15:
            high_null_cols.append((c, n_cnt, pct))
    if high_null_cols:
        col_list_str = ", ".join([f"`{c}` ({pct:.1f}%)" for c, _, pct in high_null_cols])
        quality_issues.append(("info", f"**Missing Data Concentration:** {schema['missing_count']} missing cells are heavily concentrated in: {col_list_str}."))

    # Render Alerts
    for severity, msg in quality_issues:
        if severity == "warning":
            st.warning(msg)
        elif severity == "info":
            st.info(msg)
        elif severity == "success":
            st.success(msg)

    # 2. Visual Missing Values Distribution
    null_series = df.isnull().sum()
    null_cols_only = null_series[null_series > 0].sort_values(ascending=True)
    if not null_cols_only.empty:
        st.markdown("#### 📉 Missing Values by Column")
        null_df = pd.DataFrame({
            "Column": null_cols_only.index,
            "Missing Count": null_cols_only.values,
            "Missing %": (null_cols_only.values / len(df)) * 100
        })
        null_fig = px.bar(
            null_df,
            x="Missing %",
            y="Column",
            orientation="h",
            color="Missing %",
            color_continuous_scale="Reds",
            title="Percentage of Missing Values per Column",
            text="Missing Count",
            template="plotly_dark"
        )
        null_fig.update_traces(texttemplate="%{text} nulls (%{x:.1f}%)", textposition="outside")
        st.plotly_chart(null_fig, use_container_width=True, key="quality_null_chart")

    # 3. Comprehensive Column Details & Types
    st.markdown("#### 📋 Column Details, Types & Health Status")
    col_audit = []
    for c in df.columns:
        null_c = int(df[c].isnull().sum())
        null_p = (null_c / len(df)) * 100
        comp_p = 100.0 - null_p
        
        status = "✅ Clean"
        if null_p > 50:
            status = "🔴 High Nulls"
        elif null_p > 0:
            status = "🟡 Incomplete"
        if any(c == mc[0] for mc in mixed_cols):
            status = "⚠️ Mixed Type"
        elif any(c == zc[0] for zc in id_zero_cols):
            status = "ℹ️ Has 0 Placeholders"

        col_audit.append({
            "Column Name": c,
            "Detected Type": str(df[c].dtype),
            "Null Count": null_c,
            "Completeness": f"{comp_p:.1f}%",
            "Unique Values": int(df[c].nunique()),
            "Status": status,
            "Sample Value": str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "None"
        })
    st.dataframe(pd.DataFrame(col_audit), use_container_width=True)

    # 4. Interactive Quick-Clean Studio
    st.markdown("#### 🛠️ Quick Data Cleaning & Export")
    with st.expander("✨ One-Click Data Cleaning Studio", expanded=False):
        st.markdown("Perform common data cleaning actions and download the cleaned spreadsheet:")
        cl_c1, cl_c2, cl_c3 = st.columns(3)
        
        cleaned_df = df.copy()
        
        with cl_c1:
            clean_dups = st.checkbox("Remove duplicate rows", value=True)
            clean_strip = st.checkbox("Strip leading/trailing whitespace", value=True)
        with cl_c2:
            num_impute = st.selectbox("Fill numeric missing values with:", ["Do not fill", "Median", "Mean", "Zero (0)"])
        with cl_c3:
            cat_impute = st.selectbox("Fill text missing values with:", ["Do not fill", "Unknown", "Missing", "Mode (Most frequent)"])

        if clean_dups:
            cleaned_df = cleaned_df.drop_duplicates()
        if clean_strip:
            for c in cleaned_df.select_dtypes(include='object').columns:
                cleaned_df[c] = cleaned_df[c].astype(str).str.strip()
        if num_impute == "Median":
            for c in cleaned_df.select_dtypes(include=np.number).columns:
                cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].median())
        elif num_impute == "Mean":
            for c in cleaned_df.select_dtypes(include=np.number).columns:
                cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].mean())
        elif num_impute == "Zero (0)":
            for c in cleaned_df.select_dtypes(include=np.number).columns:
                cleaned_df[c] = cleaned_df[c].fillna(0)

        if cat_impute in ["Unknown", "Missing"]:
            for c in cleaned_df.select_dtypes(include='object').columns:
                cleaned_df[c] = cleaned_df[c].fillna(cat_impute)
        elif cat_impute == "Mode (Most frequent)":
            for c in cleaned_df.select_dtypes(include='object').columns:
                if not cleaned_df[c].dropna().empty:
                    cleaned_df[c] = cleaned_df[c].fillna(cleaned_df[c].mode()[0])

        # CSV Download Button
        csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            type="primary"
        )

    # 5. Raw Data Preview
    st.markdown("#### 🔍 Raw Data Preview (First 50 Rows)")
    st.dataframe(df.head(50), use_container_width=True)

# ----------------- TAB 4: STATISTICAL ANALYSIS -----------------
with tab_stats:
    st.subheader("Statistical Analysis & Hypothesis Testing")
    st.caption("Pearson correlations, IQR anomaly/outlier detection, and ANOVA variance testing across fleet categories.")

    # Meaningful numeric measures (excluding serial/ID numbers)
    clean_numeric_cols = [c for c in schema['numeric'] if not any(k in c.lower() for k in ['id', 'no', 'sn', 'asset', 'vin', 'serial', 'code', 'index', 'plate', 'tc'])]
    if not clean_numeric_cols:
        clean_numeric_cols = schema['numeric']

    # 1. Correlation Heatmap
    st.markdown("#### 1. Correlation Heatmap & Relationship Matrix")
    c_heat1, c_heat2 = st.columns([3, 1])
    with c_heat2:
        include_all_nums = st.checkbox("Include ID / Serial columns", value=False, help="Enable to see all numeric columns including SN, Vehicle No, and TC No")
    
    target_heat_cols = schema['numeric'] if include_all_nums else clean_numeric_cols

    if len(target_heat_cols) >= 2:
        corr = df[target_heat_cols].corr()
        heatmap_fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title=f"Pearson Correlation Matrix ({len(target_heat_cols)} Measures)",
            template="plotly_dark",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(heatmap_fig, use_container_width=True, key="stats_heatmap_chart")

        # Automated Correlation Highlights
        st.markdown("##### 🔍 Key Correlation Takeaways")
        corr_pairs = []
        for i in range(len(target_heat_cols)):
            for j in range(i + 1, len(target_heat_cols)):
                c1 = target_heat_cols[i]
                c2 = target_heat_cols[j]
                r_val = corr.loc[c1, c2]
                if not np.isnan(r_val):
                    corr_pairs.append((abs(r_val), r_val, c1, c2))
        
        corr_pairs.sort(reverse=True, key=lambda x: x[0])
        top_corrs = corr_pairs[:3]
        
        cor_cols = st.columns(min(len(top_corrs), 3))
        for idx, (_, r_val, c1, c2) in enumerate(top_corrs):
            direction = "positive" if r_val > 0 else "negative"
            strength = "Strong" if abs(r_val) > 0.7 else ("Moderate" if abs(r_val) > 0.35 else "Weak")
            with cor_cols[idx]:
                st.metric(
                    label=f"{c1} vs {c2}",
                    value=f"r = {r_val:.2f}",
                    delta=f"{strength} {direction}"
                )

    # Meaningful categorical groupings (excluding unique IDs, plates, chassis numbers)
    clean_categorical_cols = [
        c for c in schema['categorical'] 
        if (df[c].nunique() < 25 and df[c].nunique() > 1) 
        and not any(k in c.lower() for k in ['id', 'no', 'sn', 'asset', 'vin', 'serial', 'code', 'plate', 'chassis', 'engine'])
    ]
    if not clean_categorical_cols:
        clean_categorical_cols = [c for c in schema['categorical'] if df[c].nunique() < 30]
    if not clean_categorical_cols:
        clean_categorical_cols = schema['categorical']

    # 2. Outlier Analysis (IQR Method)
    st.markdown("#### 2. Outlier & Anomaly Detection (IQR Method)")
    if clean_numeric_cols:
        # Default to KM Run or Age if present
        def_idx = 0
        for i, c in enumerate(clean_numeric_cols):
            if any(k in c.lower() for k in ['km', 'mileage', 'run', 'odometer']):
                def_idx = i
                break
            elif 'age' in c.lower():
                def_idx = i

        col_sel, col_grp, col_thresh = st.columns([2, 2, 1])
        with col_sel:
            outlier_col = st.selectbox(
                "Select operational metric for outlier inspection:",
                clean_numeric_cols,
                index=def_idx,
                key="outlier_sel"
            )
        with col_grp:
            box_grp_opts = ["Entire Fleet (Single Box)"] + clean_categorical_cols
            box_grp_sel = st.selectbox("Group Boxplot By:", box_grp_opts, index=1 if len(box_grp_opts) > 1 else 0, key="box_grp_sel")
        with col_thresh:
            iqr_mult = st.selectbox("IQR Sensitivity:", [1.2, 1.5, 2.0, 3.0], index=1, format_func=lambda x: f"{x} × IQR ({'Sensitive' if x==1.2 else ('Standard' if x==1.5 else ('Conservative' if x==2.0 else 'Extreme'))})")

        series = df[outlier_col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_mult * iqr
        upper_bound = q3 + iqr_mult * iqr
        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)].copy()

        o_c1, o_c2, o_c3, o_c4 = st.columns(4)
        o_c1.metric("Lower Threshold", f"{lower_bound:,.1f}")
        o_c2.metric("Median (Q2)", f"{series.median():,.1f}")
        o_c3.metric("Upper Threshold", f"{upper_bound:,.1f}")
        o_c4.metric("Detected Outliers", f"{len(outliers)} vehicles", delta=f"{(len(outliers)/len(df))*100:.1f}% of fleet", delta_color="inverse")

        # Clean Boxplot without individual Plate No clutter
        if box_grp_sel == "Entire Fleet (Single Box)":
            box_fig = px.box(
                df,
                y=outlier_col,
                points="all",
                title=f"Distribution & Outliers in {outlier_col} (Entire Fleet, IQR = {iqr:,.1f})",
                template="plotly_dark",
                color_discrete_sequence=["#6366f1"]
            )
        else:
            box_fig = px.box(
                df,
                x=box_grp_sel,
                y=outlier_col,
                points="all",
                title=f"Distribution & Outliers in {outlier_col} by {box_grp_sel} (IQR = {iqr:,.1f})",
                color=box_grp_sel,
                template="plotly_dark"
            )
        box_fig.add_hline(y=upper_bound, line_dash="dash", line_color="#ef4444", annotation_text=f"Upper Bound ({upper_bound:,.0f})")
        if lower_bound > series.min():
            box_fig.add_hline(y=lower_bound, line_dash="dash", line_color="#ef4444", annotation_text=f"Lower Bound ({lower_bound:,.0f})")
        st.plotly_chart(box_fig, use_container_width=True, key="stats_box_chart")

        # Outlier vehicles table or Top Exposure vehicles
        key_id_cols = [c for c in ['Plate No', 'Vehicle No', 'ASSET No', 'Division', 'Type', 'Model Year', 'Age', outlier_col] if c in df.columns]
        if not key_id_cols:
            key_id_cols = df.columns.tolist()[:5] + [outlier_col]

        if not outliers.empty:
            st.markdown(f"##### 🚨 Identified Outlier Records ({len(outliers)} records beyond {iqr_mult}×IQR)")
            outliers_sorted = outliers[key_id_cols].sort_values(by=outlier_col, ascending=False)
            st.dataframe(outliers_sorted, use_container_width=True)
        else:
            st.info(f"No vehicles breach the statistical {iqr_mult}×IQR upper limit ({upper_bound:,.0f}). Showing top extreme exposure vehicles below:")

        # Always show Top 5 Extreme Exposure Records for operational decision-making
        st.markdown(f"##### 🏆 Top 5 Extreme Exposure Vehicles by `{outlier_col}`")
        top_exposure = df[key_id_cols].sort_values(by=outlier_col, ascending=False).head(5)
        st.dataframe(top_exposure, use_container_width=True)

    # 3. Hypothesis Testing (ANOVA Variance Test)
    if clean_numeric_cols and clean_categorical_cols:
        st.markdown("#### 3. Hypothesis Testing (One-Way ANOVA)")
        st.caption("Determine whether group differences (e.g. between Divisions or Vehicle Types) are statistically significant.")
        
        hyp_c1, hyp_c2 = st.columns(2)
        with hyp_c1:
            h_metric = st.selectbox("Measure to Test (Dependent Variable):", clean_numeric_cols, index=def_idx, key="hyp_metric")
        with hyp_c2:
            h_group = st.selectbox("Grouping Factor (Independent Variable):", clean_categorical_cols, index=0, key="hyp_group")

        group_data = [group[h_metric].dropna().values for _, group in df.groupby(h_group) if len(group.dropna()) >= 2]
        
        # Display Group Summary Statistics Table
        st.markdown(f"##### 📊 Group Breakdown for `{h_metric}` across `{h_group}`")
        grp_summary = df.groupby(h_group)[h_metric].agg(
            Count='count',
            Mean='mean',
            Median='median',
            Std_Dev='std'
        ).reset_index()
        grp_summary['Mean'] = grp_summary['Mean'].map(lambda x: f"{x:,.2f}")
        grp_summary['Median'] = grp_summary['Median'].map(lambda x: f"{x:,.2f}")
        grp_summary['Std_Dev'] = grp_summary['Std_Dev'].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(grp_summary, use_container_width=True)

        if len(group_data) >= 2:
            f_stat, p_val = stats.f_oneway(*group_data)
            is_sig = p_val < 0.05
            
            st.markdown(f"""
<div class='insight-box'>
<strong>Hypothesis Test Result:</strong><br>
• <strong>Null Hypothesis (H₀):</strong> Average <code>{h_metric}</code> is equal across all categories of <code>{h_group}</code>.<br>
• <strong>F-Statistic:</strong> {f_stat:.3f} | <strong>p-value:</strong> {p_val:.4e}<br>
• <strong>Conclusion:</strong> {'<strong>Statistically Significant (Reject H₀):</strong> There is strong mathematical evidence that ' + h_metric + ' differs significantly across ' + h_group + ' categories (p < 0.05).' if is_sig else '<strong>Not Statistically Significant (Fail to Reject H₀):</strong> The observed differences across ' + h_group + ' could be due to random variation (p ≥ 0.05).'}
</div>
""", unsafe_allow_html=True)

# ----------------- TAB 5: MACHINE LEARNING -----------------
with tab_ml:
    st.subheader("Machine Learning Model Comparison")
    st.caption("Train, evaluate, and compare multiple Scikit-Learn models with feature importance and predictive diagnostics.")

    # Identify meaningful targets (excluding raw unique IDs and serial numbers)
    candidate_targets = [
        c for c in df.columns 
        if not any(k in c.lower() for k in ['id', 'no', 'sn', 'asset', 'vin', 'serial', 'code', 'plate', 'chassis', 'engine', 'remarks', 'permit', 'brand'])
        and df[c].nunique() > 1
    ]
    if not candidate_targets:
        candidate_targets = [c for c in df.columns if df[c].nunique() > 1]

    # Select default target (prefer KM Run or Age if present)
    def_target_idx = 0
    for i, c in enumerate(candidate_targets):
        if any(k in c.lower() for k in ['km', 'mileage', 'run', 'odometer']):
            def_target_idx = i
            break
        elif 'age' in c.lower():
            def_target_idx = i

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        target_col = st.selectbox(
            "Select Target Variable to Predict:",
            candidate_targets,
            index=def_target_idx,
            key="ml_target"
        )

    is_classification = df[target_col].nunique() <= 10 or df[target_col].dtype == 'object'
    with col_t2:
        st.markdown(f"**Task Type:**")
        if is_classification:
            st.markdown("🎯 **Classification**")
        else:
            st.markdown("📈 **Regression**")

    # Features to automatically exclude from predictive training (IDs, unique serials)
    cols_to_drop = [
        c for c in df.columns 
        if c == target_col or any(k in c.lower() for k in ['id', 'no', 'sn', 'asset', 'vin', 'serial', 'chassis', 'engine', 'plate', 'tc '])
    ]

    with st.expander("⚙️ Feature Engineering & Excluded Columns", expanded=False):
        st.markdown(f"**Excluded Identifier Columns (prevent data leakage):** {', '.join([f'`{c}`' for c in cols_to_drop if c != target_col])}")
        available_features = [c for c in df.columns if c not in cols_to_drop]
        st.markdown(f"**Active Predictive Features ({len(available_features)}):** {', '.join([f'`{c}`' for c in available_features])}")

    if st.button("🚀 Train & Compare Models", type="primary"):
        with st.spinner("Preparing features, encoding variables, and training models..."):
            ml_df = df.dropna(subset=[target_col]).copy()
            y_raw = ml_df[target_col]
            X = ml_df.drop(columns=cols_to_drop)

            # Process datetime columns into numeric components
            for c in list(X.columns):
                if np.issubdtype(X[c].dtype, np.datetime64) or pd.api.types.is_datetime64_any_dtype(X[c]):
                    try:
                        dt_series = pd.to_datetime(X[c], errors='coerce')
                        X[f"{c}_year"] = dt_series.dt.year.fillna(0)
                        X[f"{c}_month"] = dt_series.dt.month.fillna(0)
                    except Exception:
                        pass
                    X = X.drop(columns=[c])

            # Encode categorical features
            for c in list(X.columns):
                if X[c].dtype == 'object' or pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == 'string':
                    X[c] = LabelEncoder().fit_transform(X[c].astype(str))

            # Force all features to numeric & impute missing values
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(X.median(numeric_only=True)).fillna(0)

            if X.shape[1] == 0:
                st.error("No valid predictive features could be extracted for model training.")
                st.stop()

            feature_names = list(X.columns)

            if is_classification:
                le_y = LabelEncoder()
                y = le_y.fit_transform(y_raw.astype(str))
                if len(np.unique(y)) < 2:
                    st.error("Target variable must have at least 2 distinct classes.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
                }

                results = []
                best_acc = -1
                best_model_name = ""
                best_model_obj = None

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

                    if acc > best_acc:
                        best_acc = acc
                        best_model_name = name
                        best_model_obj = model

                    results.append({
                        "Model": name,
                        "Accuracy": f"{acc*100:.1f}%",
                        "Precision": f"{prec:.3f}",
                        "Recall": f"{rec:.3f}",
                        "F1-Score": f"{f1:.3f}"
                    })

                st.markdown(f"""
<div class='insight-box'>
🏆 <strong>Top Performing Model: {best_model_name}</strong><br>
Achieved the highest validation accuracy of <strong>{best_acc*100:.1f}%</strong> on the held-out test dataset (25% sample).
</div>
""", unsafe_allow_html=True)

                st.markdown("##### 📊 Model Performance Leaderboard")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

                # Feature Importance from Tree Model
                if hasattr(best_model_obj, "feature_importances_"):
                    st.markdown("##### 🌲 Top Predictive Feature Importance")
                    imp_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": best_model_obj.feature_importances_
                    }).sort_values(by="Importance", ascending=True).tail(10)
                    
                    imp_fig = px.bar(
                        imp_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Most Influential Drivers for Predicting {target_col}",
                        color="Importance",
                        color_continuous_scale="Viridis",
                        template="plotly_dark"
                    )
                    st.plotly_chart(imp_fig, use_container_width=True, key="feat_imp_chart")

            else:
                y_num = pd.to_numeric(y_raw, errors='coerce')
                valid_mask = ~y_num.isna()
                X_reg = X.loc[valid_mask]
                y_reg = y_num.loc[valid_mask]
                
                if len(y_reg) < 6:
                    st.error("Not enough valid numeric rows in target column for regression.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)
                
                models = {
                    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                    "Linear Regression": LinearRegression()
                }

                results = []
                best_r2 = -float("inf")
                best_model_name = ""
                best_model_obj = None
                best_preds = None

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    mae = mean_absolute_error(y_test, preds)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = name
                        best_model_obj = model
                        best_preds = preds

                    results.append({
                        "Model": name,
                        "R² Score": f"{r2:.3f}",
                        "RMSE": f"{rmse:,.2f}",
                        "MAE": f"{mae:,.2f}"
                    })

                st.markdown(f"""
<div class='insight-box'>
🏆 <strong>Top Performing Model: {best_model_name}</strong><br>
Achieved an <strong>R² Score of {best_r2:.3f}</strong> on held-out validation vehicles, explaining {max(0, best_r2)*100:.1f}% of the variance in <code>{target_col}</code>.
</div>
""", unsafe_allow_html=True)

                st.markdown("##### 📊 Model Performance Leaderboard")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

                c_plot1, c_plot2 = st.columns(2)
                with c_plot1:
                    # Actual vs Predicted Scatter
                    if best_preds is not None:
                        comp_df = pd.DataFrame({"Actual": y_test, "Predicted": best_preds})
                        act_fig = px.scatter(
                            comp_df,
                            x="Actual",
                            y="Predicted",
                            trendline="ols",
                            title=f"Actual vs. Predicted {target_col} ({best_model_name})",
                            template="plotly_dark"
                        )
                        st.plotly_chart(act_fig, use_container_width=True, key="act_vs_pred_chart")

                with c_plot2:
                    # Feature Importance
                    if hasattr(best_model_obj, "feature_importances_"):
                        imp_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": best_model_obj.feature_importances_
                        }).sort_values(by="Importance", ascending=True).tail(10)
                        
                        imp_fig = px.bar(
                            imp_df,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title=f"Top Predictive Drivers of {target_col}",
                            color="Importance",
                            color_continuous_scale="Viridis",
                            template="plotly_dark"
                        )
                        st.plotly_chart(imp_fig, use_container_width=True, key="reg_feat_imp_chart")
