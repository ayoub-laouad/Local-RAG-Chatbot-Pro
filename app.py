import os
import sys

# Chargement des modules
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# Modules pour la chaîne de discussion (Style 2025)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
NOM_DU_PDF = "mon_document.pdf"  # <--- ASSUREZ-VOUS D'AVOIR UN PDF DE CE NOM ICI
MODELE_LLM = "llama3.2:1b"

def main():
    # 0. Vérification
    if not os.path.exists(NOM_DU_PDF):
        print(f"❌ ERREUR : Le fichier '{NOM_DU_PDF}' est introuvable.")
        print("   -> Veuillez glisser un PDF dans ce dossier et le renommer.")
        return

    print(f"📄 1. Chargement de '{NOM_DU_PDF}'...")
    loader = PyPDFLoader(NOM_DU_PDF)
    docs = loader.load()
    
    print("✂️  2. Découpage du texte...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("🧠 3. Vectorisation (Embeddings)...")
    # Utilisation du modèle local (gratuit et rapide)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("💾 4. Base de données vectorielle...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    print("🤖 5. Préparation de l'IA (Llama 3.2)...")
    llm = ChatOllama(model=MODELE_LLM, temperature=0)

    # Création du Prompt (Les instructions pour l'IA)
    system_prompt = (
        "Tu es un assistant utile. Utilise les morceaux de contexte suivants pour répondre à la question. "
        "Si tu ne sais pas, dis-le simplement."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Assemblage de la chaîne RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n✅ --- C'EST PRÊT ! (Tapez 'q' pour quitter) ---\n")
    
    while True:
        question = input("Votre question : ")
        if question.lower() in ["q", "quit", "exit"]:
            break
            
        if question.strip() == "":
            continue

        print("   🔎 Recherche et réflexion...")
        response = rag_chain.invoke({"input": question})
        
        print(f"\n🤖 RAG : {response['answer']}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()