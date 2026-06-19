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
st.set_page_config(page_title="heorX - AI Title & Abstract Screening", layout="wide")
st.title("heorX - AI Title and Abstract Screening")
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
if "selected_abstract_id" not in st.session_state:
    st.session_state.selected_abstract_id = None
# --- Sidebar: Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_provider = st.selectbox("API Provider", ["OpenAI", "Groq"])
    api_key = st.text_input("API Key", type="password")
    
    if api_provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    else:
        model_name = st.selectbox("Model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
        
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    
    st.header("🔑 Google Sheets Connect")
    st.markdown("Upload your Google Service Account `credentials.json`:")
    credentials_file = st.file_uploader("Upload JSON", type=["json"])
    
    sheet_url_default = "https://docs.google.com/spreadsheets/d/1VCEapxI1H30xWIJwheUr7sH7fOk7oLdPQmqQWueynG0/edit"
    sheet_url = st.text_input("Google Sheet URL", value=sheet_url_default)
    
    if st.button("Connect & Load Data"):
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
                
                st.success("✅ Data loaded successfully!")
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
    # Simple regex replace for highlighting
    for kw in incl_keywords:
        if kw.strip():
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted_text = pattern.sub(f'<span style="background-color: #d4edda; color: #155724; font-weight: bold;">\g<0></span>', highlighted_text)
            
    for kw in excl_keywords:
        if kw.strip():
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            highlighted_text = pattern.sub(f'<span style="background-color: #f8d7da; color: #721c24; font-weight: bold;">\g<0></span>', highlighted_text)
            
    return highlighted_text
def sort_dataframe(df):
    if df.empty: return df
    
    df = df.copy()
    if 'Score' not in df.columns:
        df['Score'] = 0
    if 'AI Decision' not in df.columns:
        df['AI Decision'] = ""
        
    def get_sort_tier(row):
        decision = str(row.get('AI Decision', '')).strip().lower()
        score = row.get('Score', 0)
        
        if decision == 'unclear':
            return 0  # Top priority
        elif -5 <= score <= 5:
            return 1  # Tricky articles
        elif score > 5:
            return 2  # Likely inclusion
        else:
            return 3  # Likely exclusion
            
    df['sort_tier'] = df.apply(get_sort_tier, axis=1)
    # Sort by tier, then descending score
    df = df.sort_values(by=['sort_tier', 'Score'], ascending=[True, False]).drop(columns=['sort_tier'])
    return df
def style_dataframe(row):
    color = ''
    decision = str(row.get('AI Decision', '')).strip().lower()
    score = row.get('Score', 0)
    
    if decision == 'unclear':
        color = 'background-color: #fff3cd; color: #856404;' # Yellow
    elif score > 5:
        color = 'background-color: #e2f3e5;' # Light Green
    elif score < -5:
        color = 'background-color: #fce4e4;' # Light Red
    elif -5 <= score <= 5:
        color = 'background-color: #e2e3e5;' # Light Gray
        
    return [color] * len(row)
# --- Main UI ---
if not st.session_state.df_screening.empty:
    
    st.header("📝 1. PICOS & Keywords")
    show_picos = st.toggle("Show/Hide PICOS & Keywords", value=True)
    
    if show_picos:
        col1, col2 = st.columns(2)
        with col1:
            incl_criteria = st.text_area("Inclusion Criteria (PICOS)", 
                value=st.session_state.df_criteria.get("Inclusion Criteria", [""])[0] if not st.session_state.df_criteria.empty else "", 
                height=150)
        with col2:
            excl_criteria = st.text_area("Exclusion Criteria (PICOS)", 
                value=st.session_state.df_criteria.get("Exclusion Criteria", [""])[0] if not st.session_state.df_criteria.empty else "", 
                height=150)
                
        if st.button("Generate Keywords using AI"):
            with st.spinner("Generating keywords..."):
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
                        
        col3, col4 = st.columns(2)
        with col3:
            incl_kw_text = st.text_area("Inclusion Keywords (comma separated)", 
                value=", ".join(st.session_state.inclusion_keywords), height=100)
            st.session_state.inclusion_keywords = [k.strip() for k in incl_kw_text.split(",") if k.strip()]
        with col4:
            excl_kw_text = st.text_area("Exclusion Keywords (comma separated)", 
                value=", ".join(st.session_state.exclusion_keywords), height=100)
            st.session_state.exclusion_keywords = [k.strip() for k in excl_kw_text.split(",") if k.strip()]
    st.divider()
    st.header("📊 2. Screening Data")
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("1. Match Keywords & Score"):
            with st.spinner("Scoring abstracts..."):
                df = st.session_state.df_screening
                df['Score'] = df.apply(lambda row: calculate_score(f"{row.get('Title','')} {row.get('Abstract','')}", 
                                                                   st.session_state.inclusion_keywords, 
                                                                   st.session_state.exclusion_keywords), axis=1)
                st.session_state.df_screening = sort_dataframe(df)
                st.success("Scoring and sorting completed!")
                st.rerun()
    with col_btn2:
        if st.button("2. Start AI Screening"):
            with st.spinner("Running AI Screening..."):
                df = st.session_state.df_screening
                for index, row in df.iterrows():
                    # Only screen if not already decided, or force all? We'll screen all.
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
                    Provide a detailed Rationale addressing PICOS (Population, Intervention, Comparison, Outcomes, Study Design) - 5 categories for inclusion matching, 5 for exclusion matching.
                    
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
    # Display Editable DataFrame
    st.markdown("### Interactive Dataset")
    st.markdown("Edit the data directly below. 'Unclear' rows and scored rows are color-coded.")
    
    # We display styled dataframe using st.dataframe, but st.data_editor doesn't fully support Pandas styler yet.
    # We can provide a standard data editor for edits, and a styled view if needed.
    # Since manual editing is required, we use data_editor.
    edited_df = st.data_editor(
        st.session_state.df_screening.style.apply(style_dataframe, axis=1),
        use_container_width=True,
        num_rows="dynamic",
        key="data_editor"
    )
    st.session_state.df_screening = pd.DataFrame(edited_df)
    # Save to Google Sheets
    if st.button("💾 Update Google Sheet"):
        if st.session_state.gc:
            with st.spinner("Updating Google Sheet..."):
                try:
                    sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url).group(1)
                    sh = st.session_state.gc.open_by_key(sheet_id)
                    
                    # Update Screening Sheet
                    ws_screening = sh.worksheet("Title_abstract_screening")
                    ws_screening.clear()
                    ws_screening.update([st.session_state.df_screening.columns.values.tolist()] + st.session_state.df_screening.values.tolist())
                    
                    # Update Criteria Sheet
                    ws_criteria = sh.worksheet("Inclusion/Exclusion Criteria")
                    crit_df = pd.DataFrame([{
                        "Inclusion Criteria": incl_criteria if 'incl_criteria' in locals() else "",
                        "Exclusion Criteria": excl_criteria if 'excl_criteria' in locals() else "",
                        "Inclusion keywords": ", ".join(st.session_state.inclusion_keywords),
                        "Exclusion keywords": ", ".join(st.session_state.exclusion_keywords)
                    }])
                    ws_criteria.clear()
                    ws_criteria.update([crit_df.columns.values.tolist()] + crit_df.values.tolist())
                    
                    st.success("Google Sheet updated successfully!")
                except Exception as e:
                    st.error(f"Failed to update sheet: {e}")
        else:
            st.warning("Not connected to Google Sheets.")
    st.divider()
    st.header("🔍 3. Abstract Viewer")
    study_ids = st.session_state.df_screening["Study ID"].dropna().astype(str).tolist() if "Study ID" in st.session_state.df_screening.columns else []
    selected_id = st.selectbox("Select Study ID to view Abstract", study_ids)
    
    if selected_id:
        row = st.session_state.df_screening[st.session_state.df_screening["Study ID"].astype(str) == selected_id]
        if not row.empty:
            title = row.iloc[0].get("Title", "No Title")
            abstract = row.iloc[0].get("Abstract", "No Abstract")
            
            st.subheader(title)
            
            highlighted_abs = highlight_keywords(abstract, st.session_state.inclusion_keywords, st.session_state.exclusion_keywords)
            st.markdown(f"<div style='line-height: 1.6; font-size: 1.1em;'>{highlighted_abs}</div>", unsafe_allow_html=True)
            
else:
    st.info("👈 Please connect and load data from the sidebar to begin.")
