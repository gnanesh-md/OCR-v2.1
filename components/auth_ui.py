# components/auth_ui.py
import streamlit as st
from database.db_utils import register_user, verify_login

def render_auth_ui():
    """Renders the login and registration tabs."""
    st.title("🔒 Keppler Access Portal")
    st.markdown("Please log in to access the applications.")
    
    # Create tabs for Login and Register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    # --- LOGIN TAB ---
    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            submit_login = st.form_submit_button("Login")
            
            if submit_login:
                if login_username and login_password:
                    success, user_id = verify_login(login_username, login_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.user_id = user_id
                        st.rerun() # Refresh the page to show the main app
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please fill in both fields.")

    # --- REGISTER TAB ---
    with tab2:
        with st.form("register_form"):
            st.subheader("Create an Account")
            reg_username = st.text_input("New Username", key="reg_user")
            reg_password = st.text_input("New Password", type="password", key="reg_pass")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            submit_register = st.form_submit_button("Register")
            
            if submit_register:
                if reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                elif len(reg_password) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif reg_username and reg_password:
                    success, message = register_user(reg_username, reg_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields.")