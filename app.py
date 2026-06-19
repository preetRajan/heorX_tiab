import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from thefuzz import fuzz
from openai import OpenAI
from groq import Groq
import altair as alt

# --- Setup & Config ---
st.set_page_config(page_title="heorX - AI Screening", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
        border: 1px solid #dcdcdc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 1.15rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("heorX | AI Abstract Screening")

# --- Initialize Session State ---
if "df_screening" not in st.session_state:
    st.session_state.df_screening = pd.DataFrame()
if "df_criteria" not in st.session_state:
    st.session_state.df_criteria = pd.DataFrame()
if "inclusion_keywords" not in st.session_state:
    st.session_state.inclusion_keywords = []
if "exclusion_keywords" not in st.session_state:
    st.session_state.exclusion_keywords = []
if "gc" not in st.session_state:
    st.session_state.gc = None

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("Configuration")
    
    api_provider = st.selectbox("API Provider", ["OpenAI", "Groq"])
    api_key = st.text_input("API Key", type="password")
    
    if api_provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    else:
        model_name = st.selectbox("Model", [
            "llama-3.1-8b-instant", 
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ])
        
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    
    st.divider()
    
    st.header("Database Connection")
    credentials_file = st.file_uploader("Upload credentials.json", type=["json"])
    sheet_url_default = "https://docs.google.com/spreadsheets/d/1VCEapxI1H30xWIJwheUr7sH7fOk7oLdPQmqQWueynG0/edit"
    sheet_url = st.text_input("Google Sheet URL", value=sheet_url_default)
    
    if st.button("Connect & Load Data", use_container_width=True):
        if credentials_file is not None:
            try:
                creds_dict = json.load(credentials_file)
                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                st.session_state.gc = gspread.authorize(creds)
                
                sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url).group(1)
                sh = st.session_state.gc.open_by_key(sheet_id)
                
                worksheet_screening = sh.worksheet("Title_abstract_screening")
                st.session_state.df_screening = pd.DataFrame(worksheet_screening.get_all_records())
                
                worksheet_criteria = sh.worksheet("Inclusion/Exclusion Criteria")
                df_crit = pd.DataFrame(worksheet_criteria.get_all_records())
                st.session_state.df_criteria = df_crit
                
                st.success("Database linked successfully.")
            except Exception as e:
                st.error(f"Connection Error: {e}")
        else:
            st.warning("Credentials file required.")

# --- Helper Functions ---
def get_llm_response(prompt, system_prompt="You are a precise data extraction and medical analysis tool."):
    if not api_key:
        st.error("API key is missing.")
        return None
    try:
        if api_provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        else:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def calculate_score(text, incl_keywords, excl_keywords):
    if not isinstance(text, str): return 0
    score = 0
    text_lower = text.lower()
    for kw in incl_keywords:
        if fuzz.token_set_ratio(kw.lower(), text_lower) > 80:
            score += (len(kw.split()) * 2)
    for kw in excl_keywords:
        if fuzz.token_set_ratio(kw.lower(), text_lower) > 80:
            score -= (len(kw.split()) * 2)
    return max(-10, min(10, score))

def highlight_keywords(text, incl_keywords, excl_keywords):
    if not isinstance(text, str): return ""
    highlighted_text = text
    for kw in incl_keywords:
        if kw.strip():
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted_text = pattern.sub(rf'<span style="background-color: #2e7d32; color: #ffffff; padding: 0 4px; font-weight: bold;">\g<0></span>', highlighted_text)
    for kw in excl_keywords:
        if kw.strip():
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted_text = pattern.sub(rf'<span style="background-color: #c62828; color: #ffffff; padding: 0 4px; font-weight: bold;">\g<0></span>', highlighted_text)
    return highlighted_text

def sort_dataframe(df):
    if df.empty: return df
    df = df.copy()
    if 'Score' not in df.columns: df['Score'] = 0
    if 'AI Decision' not in df.columns: df['AI Decision'] = ""
        
    def get_sort_tier(row):
        decision = str(row.get('AI Decision', '')).strip().lower()
        try:
            score = float(row.get('Score', 0))
        except:
            score = 0
        
        if decision == 'unclear': return 0  
        elif -5 <= score <= 5: return 1  
        elif score > 5: return 2  
        else: return 3  
            
    df['sort_tier'] = df.apply(get_sort_tier, axis=1)
    df = df.sort_values(by=['sort_tier', 'Score'], ascending=[True, False]).drop(columns=['sort_tier'])
    return df

def style_dataframe(row):
    decision = str(row.get('AI Decision', '')).strip().lower()
    
    if decision == 'inclusion':
        return ['background-color: #2e7d32; color: #ffffff'] * len(row)
    elif decision == 'exclusion':
        return ['background-color: #c62828; color: #ffffff'] * len(row)
    elif decision == 'unclear':
        return ['background-color: #f9a825; color: #ffffff'] * len(row)
    
    try:
        score = float(row.get('Score', 0))
    except:
        score = 0
        
    if score > 5:
        return ['background-color: rgba(46, 125, 50, 0.15)'] * len(row)
    elif score < -5:
        return ['background-color: rgba(198, 40, 40, 0.15)'] * len(row)
    elif -5 <= score <= 5 and score != 0:
        return ['background-color: rgba(158, 158, 158, 0.15)'] * len(row)
        
    return [''] * len(row)

# --- Main UI Content ---
if not st.session_state.df_screening.empty:
    
    tab_picos, tab_engine, tab_review = st.tabs(["**1. Setup & PICOS**", "**2. Screening Engine**", "**3. Review & Sync**"])
    
    # --- TAB 1: PICOS & Keywords ---
    with tab_picos:
        col1, col2 = st.columns(2)
        with col1:
            incl_criteria = st.text_area("Inclusion Criteria", 
                value=st.session_state.df_criteria.get("Inclusion Criteria", [""])[0] if not st.session_state.df_criteria.empty else "", 
                height=150)
        with col2:
            excl_criteria = st.text_area("Exclusion Criteria", 
                value=st.session_state.df_criteria.get("Exclusion Criteria", [""])[0] if not st.session_state.df_criteria.empty else "", 
                height=150)
                
        if st.button("Generate Keywords", help="Automatically extracts semantic single/double word keywords from criteria using LLM."):
            with st.spinner("Generating keywords..."):
                prompt = f"""
                Analyze the criteria and extract distinct, high-value keywords.
                CRITICAL RULES: 
                1. Output ONLY single words or 2-word phrases maximum.
                2. Do NOT output long phrases or full sentences.
                
                Inclusion Criteria: {incl_criteria}
                Exclusion Criteria: {excl_criteria}
                
                Return JSON format: {{"inclusion_keywords": [], "exclusion_keywords": []}}
                """
                response = get_llm_response(prompt)
                if response:
                    try:
                        res_json = json.loads(response)
                        st.session_state.inclusion_keywords = res_json.get("inclusion_keywords", [])
                        st.session_state.exclusion_keywords = res_json.get("exclusion_keywords", [])
                    except json.JSONDecodeError:
                        st.error("Data parse failure from LLM.")
                        
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            incl_kw_text = st.text_area("Inclusion Keywords", 
                value=", ".join(st.session_state.inclusion_keywords), height=100)
            st.session_state.inclusion_keywords = [k.strip() for k in incl_kw_text.split(",") if k.strip()]
        with col4:
            excl_kw_text = st.text_area("Exclusion Keywords", 
                value=", ".join(st.session_state.exclusion_keywords), height=100)
            st.session_state.exclusion_keywords = [k.strip() for k in excl_kw_text.split(",") if k.strip()]

        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if st.button("Clear Keywords", use_container_width=True, help="Wipe generated keywords"):
                st.session_state.inclusion_keywords = []
                st.session_state.exclusion_keywords = []
                st.rerun()
        with col_c2:
            if st.button("Clear PICOS Criteria", use_container_width=True, help="Wipe criteria fields"):
                st.session_state.df_criteria = pd.DataFrame()
                st.rerun()
        with col_c3:
            if st.button("Clear Screening Results", use_container_width=True, help="Wipe AI decisions, rationales, and scores"):
                if 'AI Decision' in st.session_state.df_screening.columns:
                    st.session_state.df_screening['AI Decision'] = ""
                    st.session_state.df_screening['Rationale'] = ""
                    st.session_state.df_screening['Score'] = 0
                st.rerun()

    # --- TAB 2: AI Screening Engine ---
    with tab_engine:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Calculate Scores", help="Executes token-set fuzzy matching against keywords to apply positive/negative integers."):
                df = st.session_state.df_screening
                total = len(df)
                progress_bar = st.progress(0, text="Scoring abstracts...")
                
                scores = []
                for idx, row in df.iterrows():
                    scores.append(calculate_score(f"{row.get('Title','')} {row.get('Abstract','')}", 
                                                  st.session_state.inclusion_keywords, 
                                                  st.session_state.exclusion_keywords))
                    progress_bar.progress((idx + 1) / total, text=f"Scoring abstract {idx+1} of {total}...")
                
                df['Score'] = scores
                progress_bar.empty()
                st.session_state.df_screening = sort_dataframe(df)
                st.rerun()

        with col_btn2:
            if st.button("Execute AI Screening", help="Runs deep LLM analysis using 12 strict PICOST categories."):
                df = st.session_state.df_screening
                total = len(df)
                progress_bar = st.progress(0, text="Running AI Screening...")
                
                for i, (index, row) in enumerate(df.iterrows()):
                    title = row.get("Title", "")
                    abstract = row.get("Abstract", "")
                    prompt = f"""
                    Evaluate the following Study Title and Abstract against the criteria.
                    Title: {title}
                    Abstract: {abstract}
                    Inclusion Criteria: {incl_criteria if 'incl_criteria' in locals() else ''}
                    Exclusion Criteria: {excl_criteria if 'excl_criteria' in locals() else ''}
                    
                    TASK 1: Decide if the study should be 'Inclusion', 'Exclusion', or 'Unclear'.
                    TASK 2: You MUST select one or more exact reasons from the 12 categories below that explain your decision:
                    [IN SCOPE]: "Population in scope", "Intervention in scope", "Comparator in scope", "Outcome in scope", "Study design in scope", "Time in scope"
                    [OUT OF SCOPE]: "Population out of scope", "Intervention out of scope", "Comparator out of scope", "Outcome out of scope", "Study design out of scope", "Time out of scope"
                    
                    Return JSON:
                    {{
                        "decision": "Inclusion/Exclusion/Unclear",
                        "categories": ["list", "of", "exact", "categories", "chosen"],
                        "explanation": "Brief explanation"
                    }}
                    """
                    resp = get_llm_response(prompt)
                    if resp:
                        try:
                            res_json = json.loads(resp)
                            df.at[index, 'AI Decision'] = res_json.get("decision", "")
                            df.at[index, 'Rationale'] = f"[{', '.join(res_json.get('categories', []))}] - {res_json.get('explanation', '')}"
                        except:
                            df.at[index, 'AI Decision'] = "Error"
                            
                    progress_bar.progress((i + 1) / total, text=f"Screening abstract {i+1} of {total}...")
                
                progress_bar.empty()
                st.session_state.df_screening = sort_dataframe(df)
                st.rerun()

    # --- TAB 3: Review & Sync ---
    with tab_review:
        with st.expander("Abstract Highlighting Viewer", expanded=False):
            study_ids = st.session_state.df_screening["Study ID"].dropna().astype(str).tolist() if "Study ID" in st.session_state.df_screening.columns else []
            selected_id = st.selectbox("Select Study ID", [""] + study_ids)
            if selected_id:
                row = st.session_state.df_screening[st.session_state.df_screening["Study ID"].astype(str) == selected_id]
                if not row.empty:
                    title = row.iloc[0].get("Title", "No Title")
                    abstract = row.iloc[0].get("Abstract", "No Abstract")
                    st.markdown(f"**{title}**")
                    highlighted_abs = highlight_keywords(abstract, st.session_state.inclusion_keywords, st.session_state.exclusion_keywords)
                    st.markdown(f"<div style='line-height: 1.8; font-size: 1.05em; background: #fafafa; padding: 15px; border-radius: 8px; border: 1px solid #eaeaea;'>{highlighted_abs}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        edited_df = st.data_editor(
            st.session_state.df_screening.style.apply(style_dataframe, axis=1),
            use_container_width=True,
            num_rows="dynamic",
            height=350,
            key="data_editor"
        )
        st.session_state.df_screening = pd.DataFrame(edited_df)
        
        # Bottom Metrics
        df_display = st.session_state.df_screening
        total_inc = len(df_display[df_display['AI Decision'].str.strip().str.lower() == 'inclusion'])
        total_exc = len(df_display[df_display['AI Decision'].str.strip().str.lower() == 'exclusion'])
        total_unc = len(df_display[df_display['AI Decision'].str.strip().str.lower() == 'unclear'])
        
        m1, m2, m3, m4 = st.columns([1,1,1,2])
        m1.metric("Included", total_inc)
        m2.metric("Excluded", total_exc)
        m3.metric("Unclear", total_unc)
        
        # Horizontal Bar Chart for Exclusion Reasons
        st.markdown("#### Exclusion Analytics")
        exc_categories = [
            "Population out of scope", "Intervention out of scope", "Comparator out of scope", 
            "Outcome out of scope", "Study design out of scope", "Time out of scope"
        ]
        exc_counts = {cat: 0 for cat in exc_categories}
        df_exc = df_display[df_display['AI Decision'].str.strip().str.lower() == 'exclusion']
        for rationale in df_exc['Rationale'].dropna():
            for cat in exc_categories:
                if cat.lower() in str(rationale).lower():
                    exc_counts[cat] += 1
                    
        chart_df = pd.DataFrame(list(exc_counts.items()), columns=['Reason', 'Count'])
        chart = alt.Chart(chart_df).mark_bar(color='#c62828').encode(
            x=alt.X('Count:Q', axis=alt.Axis(tickMinStep=1)),
            y=alt.Y('Reason:N', sort='-x', title=''),
            tooltip=['Reason', 'Count']
        ).properties(height=250)
        
        st.altair_chart(chart, use_container_width=True)

        if st.button("Sync Updates to Google Sheet", type="primary", use_container_width=True):
            if st.session_state.gc:
                with st.spinner("Pushing data to Google Cloud..."):
                    try:
                        sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url).group(1)
                        sh = st.session_state.gc.open_by_key(sheet_id)
                        
                        ws_screening = sh.worksheet("Title_abstract_screening")
                        ws_screening.clear()
                        ws_screening.update([st.session_state.df_screening.columns.values.tolist()] + st.session_state.df_screening.values.tolist())
                        
                        ws_criteria = sh.worksheet("Inclusion/Exclusion Criteria")
                        crit_df = pd.DataFrame([{
                            "Inclusion Criteria": incl_criteria if 'incl_criteria' in locals() else "",
                            "Exclusion Criteria": excl_criteria if 'excl_criteria' in locals() else "",
                            "Inclusion keywords": ", ".join(st.session_state.inclusion_keywords),
                            "Exclusion keywords": ", ".join(st.session_state.exclusion_keywords)
                        }])
                        ws_criteria.clear()
                        ws_criteria.update([crit_df.columns.values.tolist()] + crit_df.values.tolist())
                        st.success("Google Sheet synced.")
                    except Exception as e:
                        st.error(f"Failed to update sheet: {e}")
            else:
                st.warning("Not connected to Google Sheets.")

else:
    st.info("Awaiting connection via sidebar credentials.")
