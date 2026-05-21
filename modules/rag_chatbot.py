# modules/rag_chatbot.py
import streamlit as st
import os
import io
import uuid
import asyncio
import concurrent.futures
import PyPDF2
import docx
from PIL import Image
import ollama
from litellm import acompletion, aembedding
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from database.db_utils import get_chat_history, save_chat_message, get_document_markdown, archive_document
import numpy as np
import time

# --- SAFE ASYNC RUNNER ---
# Runs a coroutine in an isolated thread with its own event loop.
# This prevents Streamlit's script interruption from leaving coroutines
# dangling with the 'coroutine was never awaited' RuntimeWarning.
def run_safely(coroutine):
    """Runs an async coroutine safely in a dedicated background thread."""
    def _runner(coro):
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_runner, coroutine)
        return future.result()
# -----------------------------------------------------------

# --- CUSTOM LITELLM WRAPPERS FOR LOCAL OLLAMA ---
async def local_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    model_to_use = kwargs.get("model", "ollama/qwen2.5vl:32b")
    response = await acompletion(
        model=model_to_use,
        messages=messages,
        api_base="http://localhost:11434"
    )
    return response.choices[0].message.content

async def local_embedding_func(texts, **kwargs):
    response = await aembedding(
        model="ollama/nomic-embed-text",
        input=texts,
        api_base="http://localhost:11434"
    )
    return np.array([data['embedding'] for data in response.data])
# -----------------------------------------------------------

def pil_to_bytes(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()
    text_content = ""

    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_content += extracted + "\n"
                
    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text_content = "\n".join([para.text for para in doc.paragraphs])
        
    elif filename.endswith((".txt", ".md", ".csv")):
        text_content = uploaded_file.getvalue().decode("utf-8")
        
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        # Use Local Ollama Vision for fallback extraction
        img = Image.open(uploaded_file)
        img_bytes = pil_to_bytes(img)
        response = ollama.chat(
            model='gemma4:26b',
            messages=[{
                'role': 'user',
                'content': 'Extract all text and data from this image exactly.',
                'images': [img_bytes]
            }]
        )
        text_content = response['message']['content']

    return text_content

# --- LIGHTRAG EXECUTION BLOCKS ---
async def _async_insert_text(user_id, combined_text):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    workspace_dir = os.path.join(BASE_DIR, "database", f"rag_workspace_user_{user_id}")
    os.makedirs(workspace_dir, exist_ok=True)

    rag = LightRAG(
        working_dir=workspace_dir,
        llm_model_func=local_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768, # IMPORTANT: Nomic-embed-text uses 768 dimensions
            max_token_size=8192,
            func=local_embedding_func
        )
    )
    await rag.initialize_storages()
    await rag.ainsert(combined_text)
    return True

    llm_model = kwargs.get("llm_model", "ollama/qwen2.5vl:32b")
    rag = LightRAG(
        working_dir=workspace_dir,
        llm_model_func=local_llm_func,
        llm_model_kwargs={"model": llm_model},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=local_embedding_func
        )
    )
    await rag.initialize_storages()
    
    query_prompt = f"""
    User Query: {prompt}
    System Instructions: Answer the query using ONLY the provided context. 
    Extract any requested tabular data into Markdown format. 
    You MUST provide your final response translated into: {target_language}.
    """
    
    response = await rag.aquery(query_prompt, param=QueryParam(mode="hybrid"))
    return response
# -----------------------------------------------------------

def render_rag_app():
    st.header("🕸️ Multi-lingual Context Graph Chatbot (Local)")
    st.info("Knowledge graph is securely stored on your local disk. 0 bytes are sent to the cloud.")

    if 'rag_doc_session_id' not in st.session_state:
        st.session_state.rag_doc_session_id = str(uuid.uuid4())
        
    user_id = st.session_state.get("user_id", 1)
    session_id = st.session_state.rag_doc_session_id

    if 'graph_built' not in st.session_state:
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", f"rag_workspace_user_{user_id}"))
        if os.path.exists(workspace_dir) and len(os.listdir(workspace_dir)) > 0:
            st.session_state.graph_built = True
        else:
            st.session_state.graph_built = False

    # --- VAULT INTERCEPTION LOGIC ---
    if "docs_to_rag" in st.session_state and st.session_state.docs_to_rag:
        docs = st.session_state.docs_to_rag
        
        with st.spinner(f"Loading {len(docs)} documents directly from Vault into local Knowledge Graph..."):
            combined_text = ""
            for doc_id, filename in docs:
                md_text = get_document_markdown(doc_id, user_id)
                if md_text:
                    combined_text += f"\n\n--- Document: {filename} ---\n{md_text}"
            
            if combined_text:
                try:
                    success = run_safely(_async_insert_text(user_id, combined_text))
                    if success:
                        st.session_state.graph_built = True
                        st.success(f"Successfully injected {len(docs)} vaulted documents into the Graph without uploading!")
                except Exception as e:
                    st.error(f"Failed to load from vault: {str(e)}")
        
        st.session_state.docs_to_rag = []

    with st.sidebar:
        st.subheader("🗣️ Language Options")
        target_language = st.selectbox("Output Language", ["English", "Hindi", "Spanish", "Telugu", "French", "German"])
        
        st.divider()
        st.subheader("🧠 Engine Info")
        st.caption("Active Model: `qwen2.5vl:32b` (Fixed)")
        
        st.divider()
        st.subheader("📤 Upload Context")
        uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True)
        
        if st.button("Process Documents", width='stretch'):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Vectorizing locally via Nomic-Embed..."):
                    start_time = time.time()
                    combined_text = ""
                    for file in uploaded_files:
                        extracted = extract_text_from_file(file)
                        combined_text += f"\n\n--- Document: {file.name} ---\n{extracted}"
                        
                        if extracted.strip():
                            archive_document(
                                user_id=user_id, 
                                filename=file.name, 
                                category="RAG_CONTEXT", 
                                markdown=extracted, 
                                confidence=100.0 
                            )
                    
                    if combined_text.strip():
                        try:
                            success = run_safely(_async_insert_text(user_id, combined_text))
                            if success:
                                st.session_state.graph_built = True
                                elapsed_time = time.time() - start_time
                                st.success(f"Documents processed in {elapsed_time:.2f}s & Saved to Vault!")
                                time.sleep(2)
                                st.rerun()

                        except Exception as e:
                            st.error(f"Failed to build graph: {str(e)}")
                    else:
                        st.error("Could not extract any text from the uploaded documents.")
        
        st.divider()
        if st.button("🧹 Clear Chat History", width='stretch'):
            st.session_state.rag_doc_session_id = str(uuid.uuid4())
            st.rerun()

    chat_history = get_chat_history(user_id, "LightRAG", session_id)
    
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.graph_built:
            st.warning("Please upload and process documents first before asking questions.")
            return

        with st.chat_message("user"):
            st.markdown(prompt)
        save_chat_message(user_id, "LightRAG", session_id, "user", prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"reasoning via {selected_llm}..."):
                start_time = time.time()
                try:
                    response = run_safely(_async_query_graph(user_id, prompt, target_language, llm_model="ollama/qwen2.5vl:32b"))
                    
                    # Clean up <|think|> tags if they bleed into chat output
                    if "<channel|>" in response:
                        response = response.split("<channel|>")[-1].strip()
                    end_time = time.time() # ⏱️ END TIMER
                    elapsed_time = end_time - start_time
                        
                    # Append the time to the bottom of the assistant's response visually
                    visual_response = response + f"\n\n*⏱️ Generated locally in {elapsed_time:.2f}s*"
                    
                    st.markdown(visual_response)
                    save_chat_message(user_id, "LightRAG", session_id, "assistant", response)
                except Exception as e:
                    st.error(f"Error querying graph: {str(e)}")
        
        st.rerun()