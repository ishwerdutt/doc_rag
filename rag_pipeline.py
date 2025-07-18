import os
from flask import current_app

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.ensemble import EnsembleRetriever # Still imported, but only if you were to add other retrievers
from langchain.retrievers import MultiQueryRetriever
# Removed: from langchain_community.retrievers import BM25Retriever # Removed BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Removed: from sentence_transformers import SentenceTransformer # Not directly needed without CrossEncoder


# --- Global Variables ---
vectorstore = None
qa_chain = None
llm = None
retriever = None


# --- Setup RAG Components ---
def setup_rag_components():
    global vectorstore, qa_chain, llm, retriever 

    current_app.logger.info("Setting up RAG components...")

    # Access config
    pdf_data_path = current_app.config['PDF_DATA_PATH']
    faiss_index_path = current_app.config['FAISS_INDEX_PATH']
    api_key = current_app.config['GOOGLE_API_KEY']
    model_name = "gemini-2.5-pro"

    # Validate API key
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        current_app.logger.error("GEMINI_API_KEY is not set or is default. Please update config.py.")
        llm = None
        raise ValueError("GEMINI_API_KEY is not valid or not set. Please update config.py.")


    # 1. Initialize Embeddings (HuggingFace BGE)
    try:
        current_app.logger.info("Initializing HuggingFace Embeddings with BAAI/bge-large-en-v1.5...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={
                "normalize_embeddings": True,
                "prompt": "Represent this sentence for searching relevant passages:"
            },
            model_kwargs={'device': 'cpu'} 
        )
        current_app.logger.info(f"Embeddings initialized using HuggingFace model: {embeddings.model_name}.")
    except Exception as e:
        current_app.logger.error(f"Error initializing HuggingFace Embeddings: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize HuggingFace Embeddings: {e}")


    # 2. Load or Create FAISS Vector Store
    index_faiss_file = os.path.join(faiss_index_path, "index.faiss")
    index_pkl_file = os.path.join(faiss_index_path, "index.pkl")

    if os.path.exists(index_faiss_file) and os.path.exists(index_pkl_file):
        current_app.logger.info(f"Loading existing FAISS index from {faiss_index_path}...")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        current_app.logger.info("FAISS index loaded.")
        
    else:
        # Document loading and chunking ONLY happens here, when creating a new FAISS index
        current_app.logger.info(f"Creating new FAISS index. Loading and processing documents from {pdf_data_path}...")
        loaded_documents = []
        for filename in os.listdir(pdf_data_path):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_data_path, filename)
                current_app.logger.info(f"Loading PDF: {filepath}")
                loader = PyPDFLoader(filepath)
                loaded_documents.extend(loader.load())
        current_app.logger.info(f"Loaded {len(loaded_documents)} raw documents.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            length_function=len,
            is_separator_regex=False,
        )
        chunks_for_faiss = text_splitter.split_documents(loaded_documents) # Local variable for FAISS creation
        current_app.logger.info(f"Split into {len(chunks_for_faiss)} chunks.")

        current_app.logger.info("Creating new FAISS index (embeddings documents for the first time)...")
        vectorstore = FAISS.from_documents(chunks_for_faiss, embeddings) # This is the expensive part
        vectorstore.save_local(faiss_index_path)
        current_app.logger.info("FAISS index created and saved.")


    # 3. Initialize Gemini LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1
        )
        current_app.logger.info(f"LLM '{model_name}' initialized.")
    except Exception as e:
        current_app.logger.error(f"Error initializing Gemini LLM: {e}")
        llm = None
        raise RuntimeError(f"Failed to initialize Gemini LLM: {e}")

    # 4. Initialize Retriever (Now only MultiQueryVectorRetriever)
    if llm and vectorstore:
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 for each sub-query
            llm=llm
        )
        current_app.logger.info("MultiQueryRetriever initialized as the sole retriever.")
    else:
        retriever = None
        current_app.logger.warning("Retriever not initialized due to missing LLM or vectorstore.")
        if not (llm and vectorstore):
            raise RuntimeError("RAG components (LLM or vectorstore) are not ready for retriever initialization.")

    # 5. Create QA Chain
    if llm and retriever:
        prompt_template_str = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and to the point.

Context: {context}
Question: {question}

Answer:"""
        PROMPT = ChatPromptTemplate.from_template(prompt_template_str)

        doc_chain = create_stuff_documents_chain(llm, prompt=PROMPT)
        qa_chain = create_retrieval_chain(retriever, doc_chain)

        current_app.logger.info("QA chain loaded using LCEL approach.")
    else:
        qa_chain = None
        current_app.logger.warning("QA chain not loaded due to missing LLM or retriever.")
        if not (llm and retriever):
            raise RuntimeError("QA chain could not be loaded due to missing LLM or retriever.")

# --- Get Answer from RAG ---
def get_rag_answer(user_query):
    global vectorstore, qa_chain, retriever

    # --- Intercept specific identity questions ---
    normalized_query = user_query.lower().strip()
    identity_phrases = ["who are you", "what are you", "what is your name"]

    for phrase in identity_phrases:
        if phrase in normalized_query:
            current_app.logger.info(f"Intercepted identity query: '{user_query}'. Providing predefined answer.")
            return "I am a bot designed to answer questions based on clinical equipment documents."
   

    if not vectorstore or not qa_chain or not retriever:
        return "RAG components not fully initialized. Please check the server setup and logs."

    current_app.logger.info(f"Processing user query: '{user_query}'...")
    try:
        docs = retriever.invoke(user_query)
        current_app.logger.info(f"{len(docs)} documents retrieved.")

        context = "\n\n".join([doc.page_content for doc in docs])

        result = qa_chain.invoke({
            "context": context,
            "question": user_query,
            "input": user_query
        })

        if isinstance(result, dict):
            answer = result.get("answer") or result.get("output_text") or result.get("result") or str(result)
        else:
            answer = str(result)

        current_app.logger.info(f"Generated answer: {answer[:100]}...")
        return answer
    except Exception as e:
        current_app.logger.error(f"Error during RAG query for '{user_query}': {e}", exc_info=True)
        return f"An error occurred during query processing: {str(e)}. Please check server logs."