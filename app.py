import streamlit as st
import os
import textwrap
from dotenv import load_dotenv
import subprocess
import urllib.request
from urllib.error import URLError
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration MUST be the first Streamlit command ---
st.set_page_config(
    page_title="Keppler AI Portal", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)
LOGO_URL_LARGE = "Keppler.jpeg"
LOGO_URL_SMALL = "Keppler.jpeg"

st.image(LOGO_URL_LARGE, width=100)
def ensure_ollama_running():
    """Checks if Ollama is running, and starts it in the background if not."""
    try:
        # 1. Ping the local server to see if it responds
        urllib.request.urlopen("http://localhost:11434/", timeout=1)
        # If it succeeds, do nothing.
    except (urllib.error.URLError, ConnectionRefusedError):
        # 2. If it fails, start Ollama in the background
        try:
            # Popen runs it as a background process so it doesn't freeze Streamlit
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, # Hides the messy server logs
                stderr=subprocess.DEVNULL
            )
            
            # 3. Wait up to 10 seconds for it to fully boot up
            with st.spinner("Starting local Ollama server..."):
                for _ in range(10):
                    time.sleep(1)
                    try:
                        urllib.request.urlopen("http://localhost:11434/", timeout=1)
                        return # Success! Exit the function.
                    except:
                        pass
            
            st.error("⚠️ Tried to start Ollama, but it didn't respond after 10 seconds.")
        except FileNotFoundError:
            # This happens if the system doesn't know where the 'ollama' file is
            st.error("🚨 'ollama' command not found! Ensure Ollama is installed and in your system PATH.")
            st.stop()

# Run the check before doing anything else
ensure_ollama_running()

# --- Force Sidebar Reset on Refresh ---
st.markdown("""
    <script>
        // Tell Streamlit to forget that the user closed the sidebar
        try {
            window.parent.localStorage.setItem('stSidebar', 'true');
            window.parent.localStorage.setItem('stSidebarExpanded', 'true');
        } catch (e) {}
    </script>
""", unsafe_allow_html=True)

# --- Constants & Configuration (ABSOLUTE PATHS FIX) ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Database Initialization ---
from database.db_utils import (
    initialize_extended_schema, 
    get_user_vault, 
    get_document_markdown, 
    get_chat_history
)

@st.cache_resource
def init_db():
    print("Initializing Enterprise Database Schema...")
    initialize_extended_schema()
    print("System initialization complete.")
    return True

init_db()

# --- App State Management ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'current_app' not in st.session_state:
    st.session_state.current_app = "Home"

# --- Main Application Router ---
def main():
    if not st.session_state.logged_in:
        # Failsafe import for auth UI
        try:
            from components.auth_ui import render_auth_ui
        except ImportError:
            from auth_ui import render_auth_ui
        render_auth_ui()
    else:
        # --- SaaS SIDEBAR TOGGLE & CUSTOM STYLING ---
        st.markdown(textwrap.dedent("""
            <style>
            /* YOUR CUSTOM CHAT & METADATA CSS */
            .user-msg{background:#1c2333;border:1px solid #30363d;border-radius:12px;padding:12px 16px;margin:8px 0;max-width:85%;margin-left:auto}
            .assistant-msg{background:#1a1f2e;border:1px solid #21262d;border-radius:12px;padding:16px;margin:8px 0;max-width:95%;line-height:1.7}
            .mode-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;margin-right:6px;letter-spacing:.5px}
            .badge-naive{background:#1a3a5c;color:#58a6ff}
            .badge-local{background:#1a3a2c;color:#3fb950}
            .badge-global{background:#3a1a3a;color:#d2a8ff}
            .badge-mix{background:#3a2a1a;color:#ffa657}
            .badge-error{background:#3a1a1a;color:#f85149}
            .badge-clarify{background:#2a2a1a;color:#e3b341}
            .client-banner{border:1px solid #30363d;border-radius:8px;padding:8px 12px;font-size:13px;color:#58a6ff;font-weight:600}
            .meta-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;font-size:12px;margin:6px 0}
            .meta-row{display:flex;justify-content:space-between;padding:2px 0}
            .meta-key{color:#8b949e}.meta-val{color:#e6edf3;font-weight:600}

            footer {visibility: hidden; display: none;}
            #MainMenu {visibility: hidden; display: none;}
            </style>
        """), unsafe_allow_html=True)

        # --- SaaS SIDEBAR NAVIGATION ---
        with st.sidebar:
            st.markdown(f"### 👋 Welcome back, **{st.session_state.username}**!")
            st.divider()

            # --- CLIENT SELECTION DROPDOWN ---
            from modules.precision_ocr import BLUEPRINTS
            st.markdown("### 🏢 Active Client")
            st.session_state.current_client = st.selectbox(
                "Select Client Template:",
                list(BLUEPRINTS.keys()),
                index=0
            )
            st.info(f"Using: {st.session_state.current_client} Template")
            st.divider()
            
            # Dynamic Menu Options based on Role
            menu_options = ["Home", "Universal OCR", "Document Vault"]
            if st.session_state.username == "admin":
                menu_options.append("Admin Console 🛡️")
                
            st.session_state.current_app = st.radio(
                "📌 Select Module:",
                menu_options
            )
            st.divider()
            if st.button("🚪 Logout", width="stretch"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.user_id = None
                st.rerun()

        # --- 1. HOME DASHBOARD ---
        if st.session_state.current_app == "Home":
            st.title("🧠 Keppler AI Portal")
            st.markdown("### System Dashboard")
            st.divider()
            
            # --- MODEL STATUS ROW ---
            st.subheader("📡 Local AI Engine Status")
            cols = st.columns(5)
            models_to_check = {
                # "Gemma 4": "gemma4:26b",
                # "Qwen 2.5": "qwen2.5vl:32b",
                
            }
            
            # Get list of installed models once
            try:
                import ollama
                resp = ollama.list()
                # Handle different versions of ollama library
                if hasattr(resp, 'models'):
                    installed_models = [m.model for m in resp.models]
                elif isinstance(resp, dict) and 'models' in resp:
                    installed_models = [m.get('name', m.get('model')) for m in resp['models']]
                else:
                    installed_models = []
            except Exception as e:
                installed_models = []
                st.sidebar.error(f"Ollama Connection Error: {e}")
                
            for i, (display_name, model_id) in enumerate(models_to_check.items()):
                with cols[i]:
                    is_ok = any(model_id in m for m in installed_models)
                    if is_ok:
                        st.success(f"**{display_name}**\nOnline ✅")
                    else:
                        st.error(f"**{display_name}**\nOffline ❌")
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success("#### 📄 Universal OCR\nUpload scanned documents, invoices, or medical forms. This engine extracts data with flawless spatial accuracy.")
           
        # --- 2. UNIVERSAL OCR ---
        elif st.session_state.current_app == "Universal OCR":
            from modules.precision_ocr import render_ocr_app
            render_ocr_app()
            
        # --- 3. KNOWLEDGE RAG ---
        elif st.session_state.current_app == "Knowledge RAG":
            from modules.rag_chatbot import render_rag_app
            render_rag_app()

        # --- 4. DOCUMENT VAULT & KNOWLEDGE BASE ---
        elif st.session_state.current_app == "Document Vault":
            st.title("📂 Document Vault & Knowledge Base")
            st.markdown("Select previously extracted documents to combine them into your RAG Graph without re-uploading, and review past conversations.")
            st.divider()
            
            tab1, tab2 = st.tabs(["📑 Extracted Documents", "💬 Global Chat History"])
            
            with tab1:
                vault_records = get_user_vault(st.session_state.user_id)
                
                if not vault_records:
                    st.info("Your vault is currently empty. Process documents in the **Universal OCR** tab first.")
                else:
                    selected_for_rag = []
                    
                    st.markdown("### 🗄️ Select Documents for RAG")
                    for record in vault_records:
                        doc_id, filename, category, confidence, extract_date = record
                        
                        # Expandable UI with Selectors
                        with st.expander(f"📄 {filename} | Type: {category} | Date: {extract_date[:10]}"):
                            c1, c2, c3 = st.columns([2, 1, 1])
                            with c1:
                                st.write(f"**Confidence:** {confidence}%")
                            with c2:
                                # MULTI-SELECT CHECKBOX
                                if st.checkbox("➕ Add to RAG Graph", key=f"rag_sel_{doc_id}"):
                                    selected_for_rag.append((doc_id, filename))
                            with c3:
                                # ACTUAL TEXT VIEWER
                                if st.button("👁️ View Extracted Text", key=f"view_{doc_id}"):
                                    st.session_state.view_doc_id = doc_id

                    # Action Button to push straight to RAG
                    if selected_for_rag:
                        st.divider()
                        if st.button(f"🧠 Load {len(selected_for_rag)} Selected Documents into RAG Chatbot", type="primary", width="stretch"):
                            st.session_state.docs_to_rag = selected_for_rag
                            st.session_state.current_app = "Knowledge RAG"
                            st.rerun()

                    # Render Text Viewer if requested
                    if "view_doc_id" in st.session_state:
                        st.divider()
                        st.subheader("Document Content")
                        md_text = get_document_markdown(st.session_state.view_doc_id, st.session_state.user_id)
                        st.text_area("Markdown Extraction", md_text, height=300)

            with tab2:
                st.markdown("### 🗄️ Past RAG Conversations")
                # Fetching without session_id gets the whole history
                all_history = get_chat_history(st.session_state.user_id, "LightRAG")
                
                if not all_history:
                    st.write("No chat history found.")
                else:
                    for msg in all_history[-30:]: # Show last 30 messages
                        if msg["role"] != "system":
                            with st.chat_message(msg["role"]):
                                st.markdown(msg["content"])

        # --- 5. ADMIN CONSOLE ---
        elif st.session_state.current_app == "Admin Console 🛡️":
            from modules.admin_panel import render_admin_panel
            render_admin_panel()

if __name__ == "__main__":
    main()