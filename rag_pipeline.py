import os
from flask import current_app

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- Global Variables ---
vectorstore = None
qa_chain = None
llm = None
retriever = None

# --- Setup RAG Components ---
def setup_rag_components():
    global vectorstore, qa_chain, llm, retriever

    current_app.logger.info("[RAG Setup] Initiating components initialization.")
    current_app.logger.debug(f"[RAG Setup] Configured PDF_DATA_PATH: {current_app.config.get('PDF_DATA_PATH')}")
    current_app.logger.debug(f"[RAG Setup] Configured FAISS_INDEX_PATH: {current_app.config.get('FAISS_INDEX_PATH')}")
    current_app.logger.debug(f"[RAG Setup] Configured GOOGLE_API_KEY (first 5 chars): {str(current_app.config.get('GOOGLE_API_KEY'))[:5]}...")

    pdf_data_path = current_app.config['PDF_DATA_PATH']
    faiss_index_path = current_app.config['FAISS_INDEX_PATH']
    api_key = current_app.config['GOOGLE_API_KEY']
    model_name = "gemini-2.5-flash"

    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        current_app.logger.error("[RAG Setup Error] GEMINI_API_KEY is missing or default. Initialization aborted.")
        raise ValueError("GEMINI_API_KEY is invalid or not set. Please update config.py to proceed.")

    # 1. Initialize Embeddings Model
    try:
        current_app.logger.info("[Embeddings] Initializing HuggingFace embeddings (BAAI/bge-large-en-v1.5).")
        current_app.logger.debug("Embeddings: Using model_name='BAAI/bge-large-en-v1.5', device='cpu'.")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True, "prompt": "Represent this sentence for searching relevant passages:"},
            model_kwargs={'device': 'cpu'}
        )
        current_app.logger.info("[Embeddings] Initialized successfully.")
    except Exception as e:
        current_app.logger.exception("[Embeddings] Initialization failed during model loading or setup.")
        raise RuntimeError(f"Failed to initialize embeddings: {e}")

    # 2. Load or Create FAISS Vector Store
    index_faiss_file = os.path.join(faiss_index_path, "index.faiss")
    index_pkl_file = os.path.join(faiss_index_path, "index.pkl")

    if os.path.exists(index_faiss_file) and os.path.exists(index_pkl_file):
        current_app.logger.info(f"[FAISS] Loading existing index from '{faiss_index_path}'.")
        current_app.logger.debug(f"[FAISS] Checking for index files: {index_faiss_file} and {index_pkl_file}.")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        current_app.logger.info("[FAISS] Index loaded successfully.")
    else:
        current_app.logger.info(f"[FAISS] Creating new index. No existing index found at '{faiss_index_path}'.")
        current_app.logger.debug(f"[FAISS] Attempting to create directory for index: {faiss_index_path}.")
        os.makedirs(faiss_index_path, exist_ok=True) # Ensure path exists before saving

        loaded_documents = []
        current_app.logger.info(f"[PDF] Searching for PDFs in '{pdf_data_path}'.")
        for filename in os.listdir(pdf_data_path):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_data_path, filename)
                current_app.logger.info(f"[PDF] Loading document: '{filename}'.")
                try:
                    loader = PyPDFLoader(filepath)
                    loaded_documents.extend(loader.load())
                except Exception as e:
                    current_app.logger.warning(f"[PDF] Could not load '{filename}': {e}")
        current_app.logger.info(f"[PDF] Loaded {len(loaded_documents)} raw documents from all PDFs.")

        current_app.logger.info("[Chunks] Splitting documents into chunks for FAISS.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        current_app.logger.debug(f"Chunker settings: chunk_size={text_splitter.chunk_size}, chunk_overlap={text_splitter.chunk_overlap}.")
        chunks_for_faiss = text_splitter.split_documents(loaded_documents)
        current_app.logger.info(f"[Chunks] Created {len(chunks_for_faiss)} chunks.")

        current_app.logger.info("[FAISS] Embedding chunks and creating new FAISS index. This may take a while...")
        vectorstore = FAISS.from_documents(chunks_for_faiss, embeddings)
        vectorstore.save_local(faiss_index_path)
        current_app.logger.info(f"[FAISS] Index created and saved successfully to '{faiss_index_path}'.")

    # 3. Initialize Gemini LLM
    try:
        current_app.logger.info(f"[LLM] Initializing Gemini model '{model_name}'.")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1
        )
        current_app.logger.debug(f"LLM settings: model='{model_name}', temperature={llm.temperature}.")
        current_app.logger.info("[LLM] Initialized successfully.")
    except Exception as e:
        current_app.logger.exception("[LLM] Initialization failed.")
        raise RuntimeError(f"Failed to initialize LLM: {e}")

    # 4. Initialize Retriever
    if llm and vectorstore:
        current_app.logger.info("[Retriever] Initializing MultiQueryRetriever.")
        # Retriever search_kwargs={"k": 5} means it will fetch top 5 documents
        # Consider increasing 'k' here to get more documents for potential reranking later
        retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={"k": 10}), llm=llm) # Increased k for better recall
        current_app.logger.debug(f"MultiQueryRetriever: base retriever search_kwargs: {retriever.retriever.search_kwargs}.")
        current_app.logger.info("[Retriever] MultiQueryRetriever initialized successfully.")
    else:
        retriever = None
        current_app.logger.warning("[Retriever] Not initialized due to missing LLM or vectorstore. RAG will not function.")
        raise RuntimeError("Retriever initialization failed.")

    # 5. Create QA Chain
    if llm and retriever:
        current_app.logger.info("[QA Chain] Initializing conversational QA chain.")
        prompt_template_str = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know.

Previous conversation:
{chat_history}

Context: {context}
Question: {input}

Answer:"""
        PROMPT = ChatPromptTemplate.from_template(prompt_template_str)
        current_app.logger.debug(f"QA Chain: Prompt template: '{prompt_template_str[:150]}...'")

        doc_chain = create_stuff_documents_chain(llm, prompt=PROMPT)
        qa_chain = create_retrieval_chain(retriever, doc_chain) # This wraps retriever and doc_chain
        
        current_app.logger.info("[QA Chain] Conversational QA chain initialized successfully.")
    else:
        qa_chain = None
        current_app.logger.warning("[QA Chain] Initialization failed due to missing LLM or retriever. RAG will not function.")
        raise RuntimeError("QA chain could not be loaded.")

    current_app.logger.info("[RAG Setup] All RAG components initialized.")

# --- Get Answer from RAG ---
def get_rag_answer(user_query: str, chat_history: list = None):
    global vectorstore, qa_chain, retriever

    if chat_history is None:
        current_app.logger.debug("RAG Query: chat_history was None, initialized to empty list.")
        chat_history = []

    normalized_query = user_query.lower().strip()
    identity_phrases = ["who are you", "what are you", "what is your name"]

    for phrase in identity_phrases:
        if phrase in normalized_query:
            current_app.logger.info(f"[Identity] Predefined answer returned for query: '{user_query[:70]}...'")
            return "I am a bot designed to answer questions based on clinical equipment documents.", chat_history
        
    if not all([vectorstore, qa_chain, retriever]):
        current_app.logger.error("[RAG Error] RAG components are not fully initialized for query processing. Please check startup logs.")
        return "RAG components not fully initialized. Please check the server setup and logs.", chat_history

    current_app.logger.info(f"[Query] Processing user query: '{user_query[:70]}...' (History length: {len(chat_history)}).")
    current_app.logger.debug(f"RAG Query: Full current chat history received: {chat_history}")

    try:
        current_app.logger.debug(f"RAG Query: Invoking qa_chain with input: '{user_query}' and chat_history.")
        result = qa_chain.invoke({
            "input": user_query,
            "chat_history": chat_history
        })
        current_app.logger.debug(f"RAG Query: Raw chain output: {result}") # Log the full result for debugging

        answer = result.get("answer")
        if answer is None:
            current_app.logger.warning(f"RAG Query Warning: 'answer' key not found in chain result. Converting full result to string.")
            answer = str(result)
        
        # Extract and log retrieved documents from the chain's result
        # The 'context_documents' key is where create_retrieval_chain puts the docs.
        retrieved_docs_from_chain = result.get("context_documents", [])
        current_app.logger.info(f"[Retriever] Retrieved {len(retrieved_docs_from_chain)} docs for query: '{user_query[:70]}...'.")
        
        # Log content of retrieved documents (first 100 chars of each)
        if retrieved_docs_from_chain:
            for i, doc in enumerate(retrieved_docs_from_chain):
                current_app.logger.debug(f"Retrieved Doc {i+1} (Source: {doc.metadata.get('source', 'N/A')}): '{doc.page_content[:100]}...'")
        else:
            current_app.logger.debug("Retriever: No documents were returned by the chain's internal retriever.")
        
        current_app.logger.debug(f"RAG Query: Answer before history update: '{answer[:70]}...'")

        # Update chat history
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=answer))

        current_app.logger.info(f"[Answer] Generated for '{user_query[:50]}...': '{answer[:50]}...'")
        current_app.logger.debug(f"RAG Query: Full updated chat history: {chat_history}")

        return answer, chat_history
    except Exception as e:
        current_app.logger.exception(f"[RAG Error] Exception during RAG query for '{user_query[:50]}...'.") # Logs full traceback
        return f"An error occurred during query processing: {str(e)}. Please check server logs.", chat_history