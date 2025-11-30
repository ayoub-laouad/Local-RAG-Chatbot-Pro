import streamlit as st
import os
import tempfile
import sqlite3
import uuid
import json
from datetime import datetime
from typing import List, Dict


# --- IMPORTS LANGCHAIN ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================================
# 1. CONFIGURATION & CONSTANTES
# ============================================================================

MODELE_LLM = "llama3.2:1b"
DB_FILE = "chat_history.db"
AVAILABLE_MODELS = ["llama3.2:1b", "llama3.2:3b", "mistral:7b", "phi3:mini"]

# Configuration Streamlit
st.set_page_config(
    page_title="RAG Chatbot Pro", 
    page_icon="💾", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# 2. CSS PERSONNALISÉ AMÉLIORÉ
# ============================================================================

st.markdown("""
<style>
    /* Style général */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Conteneur principal */
    .main .block-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    div[data-testid="stSidebar"] button {
        text-align: left;
        border: 1px solid #e0e0e0;
        background: white;
        color: #333;
        border-radius: 8px;
        margin: 2px 0;
        padding: 8px 12px;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-size: 0.9em;
    }
    
    div[data-testid="stSidebar"] button:hover {
        background: #f5f7fa;
        border: 1px solid #2196f3;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: 2px solid #667eea;
        box-shadow: 0 3px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Style pour les captions dans la sidebar */
    div[data-testid="stSidebar"] .element-container p {
        margin: 0;
        padding: 2px 0;
        font-size: 0.75em;
        color: #666;
    }
    
    /* Boutons icônes (favori et supprimer) - CORRIGÉ */
    div[data-testid="stSidebar"] button[aria-label="Marquer comme favori"],
    div[data-testid="stSidebar"] button[aria-label="Supprimer cette conversation"] {
        padding: 0 !important;
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        border-radius: 50% !important;
        font-size: 14px !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 auto !important;
    }
    
    /* Bouton refresh */
    div[data-testid="stSidebar"] button[aria-label="Rafraîchir"] {
        padding: 0 !important;
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        border-radius: 50% !important;
        font-size: 14px !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Supprimer le padding des colonnes pour les icônes */
    div[data-testid="stSidebar"] .stColumn:first-child,
    div[data-testid="stSidebar"] .stColumn:last-child {
        padding: 0 !important;
    }

    div[data-testid="stSidebar"] .stColumn:first-child > div,
    div[data-testid="stSidebar"] .stColumn:last-child > div {
        padding: 0 !important;
    }
    
    /* Séparateurs plus discrets */
    div[data-testid="stSidebar"] hr {
        margin: 8px 0;
        border: none;
        border-top: 1px solid #e8e8e8;
    }
    
    /* Messages chat */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Statistiques cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9em;
        margin-top: 5px;
    }
    
    /* Progress bar personnalisée */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Bouton d'export */
    .export-btn {
        background: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .export-btn:hover {
        background: #45a049;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 3. GESTION BASE DE DONNÉES
# ============================================================================

def init_db():
    """Initialise la base de données avec tables étendues et migration automatique"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Table messages
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_used TEXT,
            tokens_used INTEGER,
            sources TEXT
        )
    ''')
    
    # Migration : Ajouter les nouvelles colonnes si elles n'existent pas
    try:
        c.execute("SELECT model_used FROM messages LIMIT 1")
    except sqlite3.OperationalError:
        st.info("🔄 Migration de la base de données en cours...")
        c.execute("ALTER TABLE messages ADD COLUMN model_used TEXT")
        c.execute("ALTER TABLE messages ADD COLUMN tokens_used INTEGER")
        c.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
        st.success("✅ Migration réussie!")
    
    # Table sessions avec métadonnées
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0,
            tags TEXT,
            favorite INTEGER DEFAULT 0
        )
    ''')
    
    # Migration sessions : créer des entrées pour les sessions existantes
    c.execute('''
        INSERT OR IGNORE INTO sessions (session_id, message_count)
        SELECT session_id, COUNT(*) as count
        FROM messages
        GROUP BY session_id
    ''')
    
    # Table documents
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            page_count INTEGER,
            file_size INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()


def save_message(session_id, role, content, model_used=None, sources=None):
    """Sauvegarde un message avec métadonnées"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    sources_json = json.dumps(sources) if sources else None
    
    c.execute('''
        INSERT INTO messages (session_id, role, content, model_used, sources) 
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, role, content, model_used, sources_json))
    
    # Mise à jour de la session
    c.execute('''
        INSERT OR REPLACE INTO sessions (session_id, last_activity, message_count)
        VALUES (?, datetime('now'), 
                COALESCE((SELECT message_count FROM sessions WHERE session_id = ?), 0) + 1)
    ''', (session_id, session_id))
    
    conn.commit()
    conn.close()


def load_messages(session_id):
    """Charge tous les messages d'une session"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT role, content, timestamp, model_used, sources 
        FROM messages 
        WHERE session_id = ? 
        ORDER BY id
    ''', (session_id,))
    
    messages = []
    for row in c.fetchall():
        msg = {
            "role": row[0],
            "content": row[1],
            "timestamp": row[2],
            "model_used": row[3],
            "sources": json.loads(row[4]) if row[4] else None
        }
        messages.append(msg)
    
    conn.close()
    return messages


def get_sessions_with_details():
    """Récupère toutes les sessions avec détails"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    query = '''
        SELECT s.session_id, 
               COALESCE(s.title, m.content, 'Nouvelle conversation') as title,
               s.created_at,
               s.last_activity,
               s.message_count,
               s.favorite
        FROM sessions s
        LEFT JOIN (
            SELECT session_id, content
            FROM messages
            WHERE role = 'user' AND id IN (
                SELECT MIN(id) FROM messages WHERE role = 'user' GROUP BY session_id
            )
        ) m ON s.session_id = m.session_id
        ORDER BY s.favorite DESC, s.last_activity DESC
    '''
    
    c.execute(query)
    sessions = c.fetchall()
    conn.close()
    return sessions


def delete_session(session_id):
    """Supprime une session complète"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    c.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
    c.execute('DELETE FROM documents WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()


def toggle_favorite(session_id):
    """Bascule le statut favori d'une session"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        UPDATE sessions 
        SET favorite = CASE WHEN favorite = 0 THEN 1 ELSE 0 END
        WHERE session_id = ?
    ''', (session_id,))
    conn.commit()
    conn.close()


def get_statistics():
    """Récupère les statistiques globales"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    stats = {}
    
    # Total messages
    c.execute('SELECT COUNT(*) FROM messages')
    stats['total_messages'] = c.fetchone()[0]
    
    # Total sessions
    c.execute('SELECT COUNT(*) FROM sessions')
    stats['total_sessions'] = c.fetchone()[0]
    
    # Total documents
    c.execute('SELECT COUNT(*) FROM documents')
    stats['total_documents'] = c.fetchone()[0]
    
    # Messages aujourd'hui
    c.execute('''
        SELECT COUNT(*) FROM messages 
        WHERE DATE(timestamp) = DATE('now')
    ''')
    stats['messages_today'] = c.fetchone()[0]
    
    conn.close()
    return stats


def export_session(session_id):
    """Exporte une session en format texte"""
    messages = load_messages(session_id)
    export_text = f"=== Export de conversation ===\n"
    export_text += f"ID Session: {session_id}\n"
    export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for msg in messages:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        export_text += f"{role} ({msg['timestamp']}):\n{msg['content']}\n\n"
        export_text += "-" * 80 + "\n\n"
    
    return export_text


# Initialisation
init_db()


# ============================================================================
# 4. GESTION DE SESSION STREAMLIT
# ============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "show_stats" not in st.session_state:
    st.session_state.show_stats = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODELE_LLM

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000

if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0


# ============================================================================
# 5. FONCTION RAG AMÉLIORÉE
# ============================================================================

@st.cache_resource
def process_pdfs(uploaded_files, chunk_size=1000, chunk_overlap=200):
    """Traite les PDFs avec paramètres configurables"""
    if not uploaded_files: 
        return None
    
    all_splits = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Traitement de {uploaded_file.name}...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            splits = text_splitter.split_documents(docs)
            
            # Ajout métadonnées
            for split in splits:
                split.metadata['source_file'] = uploaded_file.name
            
            all_splits.extend(splits)
            
            # Sauvegarde info document
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''
                INSERT INTO documents (session_id, filename, page_count, file_size)
                VALUES (?, ?, ?, ?)
            ''', (st.session_state.session_id, uploaded_file.name, len(docs), uploaded_file.size))
            conn.commit()
            conn.close()
            
        except Exception as e:
            st.error(f"Erreur lors du traitement de {uploaded_file.name}: {str(e)}")
        finally:
            os.remove(tmp_path)
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.text("✅ Traitement terminé!")
    progress_bar.empty()
    status_text.empty()

    if all_splits:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        return vectorstore
    
    return None


# ============================================================================
# 6. FONCTIONS UI - SIDEBAR
# ============================================================================

def render_sessions_list(sessions):
    """Affiche la liste des conversations"""
    for sess_id, title, created, last_activity, msg_count, is_fav in sessions:
        is_active = (sess_id == st.session_state.session_id)
        
        with st.container():
            cols = st.columns([0.6, 10, 0.6])
            
            # Bouton favori
            with cols[0]:
                if st.button(
                    "⭐" if is_fav else "☆", 
                    key=f"fav_{sess_id}", 
                    help="Marquer comme favori"
                ):
                    toggle_favorite(sess_id)
                    st.rerun()
            
            # Contenu de la conversation
            with cols[1]:
                titre = (title[:45] + '...') if len(title) > 45 else title
                
                # Formater la date
                try:
                    date_str = datetime.strptime(last_activity, '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y')
                except:
                    date_str = last_activity[:10] if last_activity else "N/A"
                
                if st.button(
                    f"{titre}",
                    key=f"sess_{sess_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    help=f"Ouvrir la conversation du {date_str}"
                ):
                    st.session_state.session_id = sess_id
                    st.rerun()
                
                # Info sous le titre
                st.caption(f"📅 {date_str} • 💬 {msg_count} message{'s' if msg_count > 1 else ''}")
            
            # Bouton supprimer
            with cols[2]:
                if st.button(
                    "🗑️", 
                    key=f"del_{sess_id}", 
                    help="Supprimer cette conversation"
                ):
                    delete_session(sess_id)
                    if sess_id == st.session_state.session_id:
                        st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
            
            # Séparateur entre conversations
            if not is_active:
                st.markdown("---")


def render_tab_chat():
    """Onglet Chat de la sidebar"""
    # Nouveau chat + Refresh
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ Nouvelle Discussion", type="primary", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    
    with col2:
        if st.button("🔄", help="Rafraîchir"):
            st.rerun()
    
    # Recherche
    search = st.text_input("🔍 Rechercher", placeholder="Rechercher une conversation...")
    
    st.divider()
    
    # Liste des sessions
    sessions = get_sessions_with_details()
    
    if search:
        sessions = [s for s in sessions if search.lower() in s[1].lower()]
    
    with st.container(height=400):
        render_sessions_list(sessions)


def render_tab_config():
    """Onglet Configuration de la sidebar"""
    st.subheader("Paramètres du modèle")
    
    # Sélection modèle
    selected_model = st.selectbox(
        "Modèle LLM",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model
    
    # Paramètres RAG
    st.divider()
    st.subheader("Paramètres RAG")
    
    st.session_state.chunk_size = st.slider("Taille des chunks", 500, 2000, st.session_state.chunk_size, 100)
    st.session_state.chunk_overlap = st.slider("Chevauchement", 0, 500, st.session_state.chunk_overlap, 50)
    st.session_state.temperature = st.slider("Température", 0.0, 1.0, st.session_state.temperature, 0.1)
    
    # Export
    st.divider()
    if st.button("💾 Exporter cette conversation", use_container_width=True):
        export = export_session(st.session_state.session_id)
        st.download_button(
            "📥 Télécharger",
            export,
            file_name=f"conversation_{st.session_state.session_id[:8]}.txt",
            mime="text/plain"
        )


def render_tab_stats():
    """Onglet Statistiques de la sidebar"""
    stats = get_statistics()
    
    st.metric("📨 Messages totaux", stats['total_messages'])
    st.metric("💬 Conversations", stats['total_sessions'])
    st.metric("📄 Documents", stats['total_documents'])
    st.metric("🆕 Messages aujourd'hui", stats['messages_today'])
    
    st.divider()
    
    if st.button("🗑️ Nettoyer anciennes conversations", use_container_width=True):
        if st.button("⚠️ Confirmer suppression", type="secondary"):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''
                DELETE FROM messages WHERE session_id IN (
                    SELECT session_id FROM sessions 
                    WHERE last_activity < datetime('now', '-30 days')
                )
            ''')
            conn.commit()
            conn.close()
            st.success("Nettoyage effectué!")
            st.rerun()


def render_documents_uploader():
    """Upload de documents"""
    st.divider()
    st.header("📚 Documents")
    uploaded_files = st.file_uploader(
        "Chargez vos PDFs", 
        type="pdf", 
        accept_multiple_files=True, 
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.caption(f"✅ {len(uploaded_files)} fichier(s) chargé(s)")
    
    return uploaded_files


def render_sidebar():
    """Affiche la sidebar complète"""
    with st.sidebar:
        st.title("🗂️ Gestionnaire de Discussions")
        
        # Onglets
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "⚙️ Config", "📊 Stats"])
        
        with tab1:
            render_tab_chat()
        
        with tab2:
            render_tab_config()
        
        with tab3:
            render_tab_stats()
        
        # Upload de documents
        uploaded_files = render_documents_uploader()
    
    return uploaded_files


# ============================================================================
# 7. FONCTIONS UI - CONTENU PRINCIPAL
# ============================================================================

def render_header_stats():
    """Affiche les statistiques en header"""
    col1, col2, col3, col4 = st.columns(4)
    stats = get_statistics()
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_messages']}</div>
            <div class="stat-label">Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_sessions']}</div>
            <div class="stat-label">Conversations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_documents']}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">0</div>
            <div class="stat-label">Actifs</div>
        </div>
        """, unsafe_allow_html=True)


def render_chat_history(messages):
    """Affiche l'historique des messages"""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Affichage des sources si disponibles
            if message.get("sources") and message["role"] == "assistant":
                with st.expander("📚 Sources utilisées"):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"**Source {i}:** Page {source.get('page', '?')} - {source.get('file', 'Document')}")
                        st.text(source.get('content', '')[:200] + "...")


def build_rag_chain(vectorstore, selected_model, temperature):
    """Construit la chaîne RAG"""
    llm = ChatOllama(model=selected_model, temperature=temperature)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant expert. Reformule la question pour qu'elle soit autonome et claire."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_retriever = create_history_aware_retriever(llm, retriever, context_prompt)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant IA expert et précis. 
        Utilise le contexte suivant pour répondre à la question de manière détaillée et structurée:
        
        {context}
        
        Règles:
        - Réponds uniquement basé sur le contexte fourni
        - Si l'information n'est pas dans le contexte, dis-le clairement
        - Structure ta réponse avec des paragraphes
        - Cite les sources quand pertinent
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, qa_chain)
    
    return rag_chain


def handle_user_input(user_input, uploaded_files, rag_chain, current_messages):
    """Gère l'input utilisateur et la génération de réponse"""
    # Sauvegarde message utilisateur
    save_message(st.session_state.session_id, "user", user_input)
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    is_first_message = len(current_messages) == 0
    
    # Réponse assistant
    with st.chat_message("assistant"):
        response_text = ""
        sources_list = []
        
        if rag_chain:
            with st.spinner("🔍 Analyse des documents..."):
                history_objs = [
                    HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) 
                    for m in current_messages
                ]
                
                try:
                    resp = rag_chain.invoke({
                        "input": user_input, 
                        "chat_history": history_objs
                    })
                    
                    response_text = resp['answer']
                    st.markdown(response_text)
                    
                    # Préparation sources pour sauvegarde
                    for doc in resp.get('context', []):
                        sources_list.append({
                            'page': doc.metadata.get('page', '?'),
                            'file': doc.metadata.get('source_file', 'Document'),
                            'content': doc.page_content[:200]
                        })
                    
                    # Affichage sources
                    with st.expander("📚 Sources utilisées"):
                        for i, doc in enumerate(resp.get('context', []), 1):
                            st.caption(f"**Source {i}:** Page {doc.metadata.get('page','?')} - {doc.metadata.get('source_file', 'Document')}")
                            st.text(doc.page_content[:200] + "...")
                            
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    response_text = "Désolé, une erreur s'est produite lors du traitement de votre demande."
        
        else:
            llm_simple = ChatOllama(model=st.session_state.selected_model, temperature=st.session_state.temperature)
            
            with st.spinner("💭 Réflexion en cours..."):
                try:
                    resp = llm_simple.invoke([HumanMessage(content=user_input)])
                    response_text = resp.content
                    st.markdown(response_text)
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    response_text = "Désolé, une erreur s'est produite."
    
    # Sauvegarde réponse
    save_message(
        st.session_state.session_id, 
        "assistant", 
        response_text,
        model_used=st.session_state.selected_model,
        sources=sources_list if sources_list else None
    )
    
    if is_first_message:
        st.rerun()


# ============================================================================
# 8. MAIN - LOGIQUE PRINCIPALE
# ============================================================================

# En-tête
st.title("💾 RAG Chatbot Pro - Assistant Intelligent")

# Affichage sidebar et récupération fichiers
uploaded_files = render_sidebar()

# Affichage statistiques
render_header_stats()

st.divider()

# Charger les messages actuels
current_messages = load_messages(st.session_state.session_id)

# Afficher l'historique
render_chat_history(current_messages)

# Préparation RAG chain
rag_chain = None
if uploaded_files:
    with st.spinner("🔄 Indexation des documents..."):
        vectorstore = process_pdfs(
            uploaded_files, 
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap
        )
        
    if vectorstore:
        rag_chain = build_rag_chain(
            vectorstore, 
            st.session_state.selected_model,
            st.session_state.temperature
        )

# Input utilisateur
if user_input := st.chat_input("💬 Posez votre question..."):
    handle_user_input(user_input, uploaded_files, rag_chain, current_messages)

# Footer
st.divider()
st.caption("💾 RAG Chatbot Pro | Développé avec Streamlit & LangChain | Powered by Ollama")
