import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import re
from thefuzz import fuzz
from openai import OpenAI
from groq import Groq
import time

# --- Setup & Config ---
st.set_page_config(page_title="heorX - AI Screening", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Sleek UI
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        border: 1px solid #e0e0e0;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #4CAF50;
        color: #4CAF50;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("heorX | AI Abstract Screening")
st.markdown("Automated Title and Abstract Screening using LLMs and Fuzzy Matching.")

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
    st.header("⚙️ Configuration")
    
    st.subheader("Model Selection")
    api_provider = st.selectbox("API Provider", ["OpenAI", "Groq"])
    api_key = st.text_input("API Key", type="password", placeholder="Enter your secret key...")
    
    if api_provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    else:
        # Updated Groq Models
        model_name = st.selectbox("Model", [
            "llama-3.1-8b-instant", 
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ])
        
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, help="Higher values make output more random.")
    
    st.divider()
    
    st.subheader("🔑 Database Connect")
    credentials_file = st.file_uploader("1. Upload credentials.json", type=["json"])
    
    sheet_url_default = "https://docs.google.com/spreadsheets/d/1VCEapxI1H30xWIJwheUr7sH7fOk7oLdPQmqQWueynG0/edit"
    sheet_url = st.text_input("2. Google Sheet URL", value=sheet_url_default)
    
    if st.button("🔗 Connect & Load Data", use_container_width=True):
        if credentials_file is not None:
            try:
                creds_dict = json.load(credentials_file)
                scopes = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]
                creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
                st.session_state.gc = gspread.authorize(creds)
                
                # Extract Sheet ID
                sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url).group(1)
                sh = st.session_state.gc.open_by_key(sheet_id)
                
                # Load Title_abstract_screening
                worksheet_screening = sh.worksheet("Title_abstract_screening")
                st.session_state.df_screening = pd.DataFrame(worksheet_screening.get_all_records())
                
                # Load Inclusion/Exclusion Criteria
                worksheet_criteria = sh.worksheet("Inclusion/Exclusion Criteria")
                df_crit = pd.DataFrame(worksheet_criteria.get_all_records())
                st.session_state.df_criteria = df_crit
                
                st.success("✅ Database linked successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.warning("Please upload credentials.json first.")

# --- Helper Functions ---
def get_llm_response(prompt, system_prompt="You are a helpful medical research assistant."):
    if not api_key:
        st.error("Please provide an API key in the sidebar.")
        return None
    
    try:
        if api_provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        else:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def calculate_score(text, incl_keywords, excl_keywords):
    if not isinstance(text, str):
        return 0
    score = 0
    text_lower = text.lower()
    for kw in incl_keywords:
        if fuzz.partial_ratio(kw.lower(), text_lower) > 85:
            score += 2
    for kw in excl_keywords:
        if fuzz.partial_ratio(kw.lower(), text_lower) > 85:
            score -= 2
    return max(-10, min(10, score))

def highlight_keywords(text, incl_keywords, excl_keywords):
    if not isinstance(text, str):
        return ""
    highlighted_text = text
    for kw in incl_keywords:
        if kw.strip():
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted_text = pattern.sub(rf'<span style="background-color: #d4edda; color: #155724; border-radius: 3px; padding: 0 2px; font-weight: 600;">\g<0></span>', highlighted_text)
            
    for kw in excl_keywords:
        if kw.strip():
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted_text = pattern.sub(rf'<span style="background-color: #f8d7da; color: #721c24; border-radius: 3px; padding: 0 2px; font-weight: 600;">\g<0></span>', highlighted_text)
            
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
        except (ValueError, TypeError):
            score = 0
        
        if decision == 'unclear': return 0  # Top priority
        elif -5 <= score <= 5: return 1  # Tricky articles
        elif score > 5: return 2  # Likely inclusion
        else: return 3  # Likely exclusion
            
    df['sort_tier'] = df.apply(get_sort_tier, axis=1)
    df = df.sort_values(by=['sort_tier', 'Score'], ascending=[True, False]).drop(columns=['sort_tier'])
    return df

def style_dataframe(row):
    color = ''
    decision = str(row.get('AI Decision', '')).strip().lower()
    try:
        score = float(row.get('Score', 0))
    except (ValueError, TypeError):
        score = 0
    
    if decision == 'unclear':
        color = 'background-color: #fff3cd; color: #856404;' # Yellow
    elif score > 5:
        color = 'background-color: #e2f3e5;' # Light Green
    elif score < -5:
        color = 'background-color: #fce4e4;' # Light Red
    elif -5 <= score <= 5:
        color = 'background-color: #e2e3e5;' # Light Gray
        
    return [color] * len(row)

# --- Main UI Content ---
if not st.session_state.df_screening.empty:
    
    # Sleek Tabbed Interface
    tab_picos, tab_engine, tab_review = st.tabs(["📋 1. Setup & PICOS", "🤖 2. Screening Engine", "🔍 3. Review & Sync"])
    
    # --- TAB 1: PICOS & Keywords ---
    with tab_picos:
        st.subheader("Define Criteria & Extract Keywords")
        st.markdown("Provide your exact **PICOS** framework below. The AI will generate matching semantic keywords to score abstracts.")
        
        col1, col2 = st.columns(2)
        with col1:
            incl_criteria = st.text_area("✅ Inclusion Criteria", 
                value=st.session_state.df_criteria.get("Inclusion Criteria", [""])[0] if not st.session_state.df_criteria.empty else "", 
                height=150)
        with col2:
            excl_criteria = st.text_area("❌ Exclusion Criteria", 
                value=st.session_state.df_criteria.get("Exclusion Criteria", [""])[0] if not st.session_state.df_criteria.empty else "", 
                height=150)
                
        if st.button("✨ Auto-Generate Keywords"):
            with st.spinner("Analyzing criteria & generating keywords..."):
                prompt = f"""
                Based on the following PICOS criteria, extract distinct, concise keywords or short phrases.
                Inclusion Criteria: {incl_criteria}
                Exclusion Criteria: {excl_criteria}
                Provide the output strictly as a JSON object with two lists: 'inclusion_keywords' and 'exclusion_keywords'.
                """
                response = get_llm_response(prompt)
                if response:
                    try:
                        res_json = json.loads(response)
                        st.session_state.inclusion_keywords = res_json.get("inclusion_keywords", [])
                        st.session_state.exclusion_keywords = res_json.get("exclusion_keywords", [])
                        st.success("Keywords generated successfully!")
                    except json.JSONDecodeError:
                        st.error("Failed to parse JSON from LLM.")
                        
        st.markdown("---")
        st.markdown("##### 📝 Active Keywords (Editable)")
        col3, col4 = st.columns(2)
        with col3:
            incl_kw_text = st.text_area("Inclusion Keywords (comma separated)", 
                value=", ".join(st.session_state.inclusion_keywords), height=100)
            st.session_state.inclusion_keywords = [k.strip() for k in incl_kw_text.split(",") if k.strip()]
        with col4:
            excl_kw_text = st.text_area("Exclusion Keywords (comma separated)", 
                value=", ".join(st.session_state.exclusion_keywords), height=100)
            st.session_state.exclusion_keywords = [k.strip() for k in excl_kw_text.split(",") if k.strip()]


    # --- TAB 2: AI Screening Engine ---
    with tab_engine:
        st.subheader("Automated Scoring & Screening")
        st.markdown("Execute the keyword fuzzy-matching algorithms, then run the full generative AI screening.")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            st.info("Step 1: Matches dataset against keywords to apply base +/- scores. Organizes trickiest (-5 to +5) to the top.")
            if st.button("📊 1. Calculate Scores & Sort"):
                with st.spinner("Scoring abstracts..."):
                    df = st.session_state.df_screening
                    df['Score'] = df.apply(lambda row: calculate_score(f"{row.get('Title','')} {row.get('Abstract','')}", 
                                                                       st.session_state.inclusion_keywords, 
                                                                       st.session_state.exclusion_keywords), axis=1)
                    st.session_state.df_screening = sort_dataframe(df)
                    st.success("Scoring and sorting completed!")
                    st.rerun()

        with col_btn2:
            st.info("Step 2: Uses the LLM to read abstracts and provide Inclusion/Exclusion/Unclear logic and PICOS rationale.")
            if st.button("🤖 2. Execute AI Screening"):
                with st.spinner("Running deep AI Screening logic..."):
                    df = st.session_state.df_screening
                    for index, row in df.iterrows():
                        title = row.get("Title", "")
                        abstract = row.get("Abstract", "")
                        prompt = f"""
                        You are an expert medical researcher performing abstract screening.
                        Evaluate the following Study Title and Abstract against the criteria.
                        
                        Title: {title}
                        Abstract: {abstract}
                        Inclusion Criteria: {incl_criteria if 'incl_criteria' in locals() else ''}
                        Exclusion Criteria: {excl_criteria if 'excl_criteria' in locals() else ''}
                        
                        Decide if the study should be 'Inclusion', 'Exclusion', or 'Unclear'.
                        Provide a detailed Rationale addressing PICOS (Population, Intervention, Comparison, Outcomes, Study Design).
                        
                        Return strictly as JSON:
                        {{
                            "decision": "Inclusion/Exclusion/Unclear",
                            "rationale": "Detailed rationale text here"
                        }}
                        """
                        resp = get_llm_response(prompt)
                        if resp:
                            try:
                                res_json = json.loads(resp)
                                df.at[index, 'AI Decision'] = res_json.get("decision", "")
                                df.at[index, 'Rationale'] = res_json.get("rationale", "")
                            except:
                                df.at[index, 'AI Decision'] = "Error"
                    
                    st.session_state.df_screening = sort_dataframe(df)
                    st.success("AI Screening completed!")
                    st.rerun()


    # --- TAB 3: Review & Sync ---
    with tab_review:
        st.subheader("Interactive Review & Manual Edits")
        st.markdown("Use this interface to view individual abstracts, modify the AI's decisions, and push data back to your Google Sheet.")
        
        # Expandable Abstract Viewer
        with st.expander("🔍 Open Abstract Highlighting Viewer", expanded=False):
            study_ids = st.session_state.df_screening["Study ID"].dropna().astype(str).tolist() if "Study ID" in st.session_state.df_screening.columns else []
            selected_id = st.selectbox("Select Study ID to view Abstract", [""] + study_ids)
            
            if selected_id:
                row = st.session_state.df_screening[st.session_state.df_screening["Study ID"].astype(str) == selected_id]
                if not row.empty:
                    title = row.iloc[0].get("Title", "No Title")
                    abstract = row.iloc[0].get("Abstract", "No Abstract")
                    st.markdown(f"**{title}**")
                    highlighted_abs = highlight_keywords(abstract, st.session_state.inclusion_keywords, st.session_state.exclusion_keywords)
                    st.markdown(f"<div style='line-height: 1.8; font-size: 1.05em; background: #fafafa; padding: 15px; border-radius: 8px; border: 1px solid #eaeaea;'>{highlighted_abs}</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Data Editor
        edited_df = st.data_editor(
            st.session_state.df_screening.style.apply(style_dataframe, axis=1),
            use_container_width=True,
            num_rows="dynamic",
            height=400,
            key="data_editor"
        )
        st.session_state.df_screening = pd.DataFrame(edited_df)

        # Sync Button
        if st.button("💾 Sync Updates to Google Sheet", type="primary"):
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
                        
                        st.success("Google Sheet synced and updated successfully!")
                    except Exception as e:
                        st.error(f"Failed to update sheet: {e}")
            else:
                st.warning("Not connected to Google Sheets.")

else:
    # Empty State Display
    st.info("👈 Please enter your credentials and Google Sheet URL in the sidebar, then click **Connect & Load Data** to begin.")
