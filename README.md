# 🤖 Local RAG Chatbot Pro

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-orange)

Une solution de **Chatbot RAG (Retrieval-Augmented Generation)** entièrement locale, conçue pour garantir la souveraineté des données. Ce projet permet de discuter avec vos documents PDF sans qu'aucune donnée ne quitte votre machine.

Développé dans le cadre du Master *Systèmes d'Information et Systèmes Intelligents* à l'**INSEA**.

---

## 🚀 Fonctionnalités Clés

* **🛡️ 100% Local & Privé :** Aucune donnée envoyée vers le cloud (utilise Ollama).
* **🧠 Modèles Flexibles :** Compatible avec Llama 3.2, Mistral, Phi-3, etc.
* **💬 Interface "Pro" :**
    * Historique des conversations persistant (SQLite).
    * Gestion de sessions multiples.
    * Mise en favoris et export des discussions.
* **📚 Citations de Sources :** L'IA indique précisément la page et le fichier source de ses réponses.
* **📊 Tableau de Bord :** Statistiques d'utilisation (nombre de messages, documents, tokens).

## 🛠️ Architecture Technique

Le projet repose sur une architecture modulaire décrite dans le rapport :
1.  **Interface :** [Streamlit](https://streamlit.io/)
2.  **Orchestration :** [LangChain](https://www.langchain.com/)
3.  **LLM & Embeddings :** [Ollama](https://ollama.com/) & HuggingFace (`all-MiniLM-L6-v2`)
4.  **Base Vectorielle :** ChromaDB
5.  **Mémoire :** SQLite

## 📦 Installation

### Prérequis
1.  **Python 3.10+** installé.
2.  **[Ollama](https://ollama.com/download)** installé et fonctionnel.
3.  Téléchargez le modèle Llama 3.2 :
    ```bash
    ollama pull llama3.2:1b
    ```

### Étapes
1.  Clonez ce repository :
    ```bash
    git clone [https://github.com/VOTRE-NOM/Local-RAG-Chatbot-Pro.git](https://github.com/VOTRE-NOM/Local-RAG-Chatbot-Pro.git)
    cd Local-RAG-Chatbot-Pro
    ```

2.  Créez un environnement virtuel :
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Utilisation

Lancez l'application principale (GUI) :

```bash
streamlit run gui.py
```

Une version CLI légère est également disponible via :

```bash
python app.py
```

## 📸 Aperçu
L'application permet de charger plusieurs PDF via la barre latérale, de configurer les paramètres du RAG (taille des chunks, température) et de visualiser les sources utilisées pour chaque réponse.

## 📝 Auteur
LAOUAD Ayoub Master M2SI - INSEA, Rabat (Décembre 2025)
