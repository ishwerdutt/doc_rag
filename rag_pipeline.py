import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# For Google Gemini API
from langchain_google_genai import ChatGoogleGenerativeAI
# For Multi-Query Retriever
from langchain.retrievers import MultiQueryRetriever
from flask import current_app # Import current_app to access Flask app context

# --- Global Variables for RAG Components ---
vectorstore = None
qa_chain = None
llm = None
retriever = None # New global for MultiQueryRetriever

# initializing rag components

def setup_rag_components(): 
    global vectorstore, qa_chain, llm, retriever
    
    print("Setting up RAG components...")
    
    # Access config vars from current_app
    pdf_data_path = current_app.config['PDF_DATA_PATH']
    faiss_index_path = current_app.config['FAISS_INDEX_PATH']
    gemini_api_key = current_app.config['GOOGLE_API_KEY'] # Corrected variable name
    gemini_model_name = "gemma-3n-e2b-it"

    # Validate API Key
    if gemini_api_key == "YOUR_GEMINI_API_KEY" or not gemini_api_key:
        print("ERROR: GEMINI_API_KEY is not set. Please update config.py or set the environment variable.")
        llm = None # Prevent LLM initialization
        return

    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embeddings model initialized.")

    # 2. Load or Create FAISS Vector Store
    # FAISS stores its index in two files: .faiss and .pkl
    index_faiss_file = os.path.join(faiss_index_path, "index.faiss")
    index_pkl_file = os.path.join(faiss_index_path, "index.pkl")

    if os.path.exists(index_faiss_file) and os.path.exists(index_pkl_file):
        print(f"Loading existing FAISS index from {faiss_index_path}...")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
    else:
        print(f"Creating new FAISS index from PDFs in {pdf_data_path}...")
        documents = []
        for filename in os.listdir(pdf_data_path):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_data_path, filename)
                print(f"Loading PDF: {filepath}")
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
        print(f"Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=128,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("FAISS vector store created.")

        vectorstore.save_local(faiss_index_path)
        print(f"FAISS index saved to {faiss_index_path}.")

    # 3. Initialize LLM (Google Gemini API)
    try:
        print(f"Initializing Google Gemini LLM: {gemini_model_name}...")
        llm = ChatGoogleGenerativeAI(model=gemini_model_name, google_api_key=gemini_api_key, temperature=0.7)
        print(f"Google Gemini LLM '{gemini_model_name}' initialized.")

    except Exception as e:
        print(f"Error initializing Google Gemini LLM: {e}")
        print(f"Please ensure your API key is correct and the model '{gemini_model_name}' is accessible.")
        llm = None # Set to None to indicate failure

    # 4. Initialize MultiQueryRetriever
    if llm and vectorstore:
        # The MultiQueryRetriever uses the LLM to generate multiple queries
        # from the original user query, then uses the base retriever (vectorstore)
        # to fetch documents for each generated query, and finally combines the results.
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 for each sub-query
            llm=llm
        )
        print("MultiQueryRetriever initialized.")
    else:
        retriever = None
        print("MultiQueryRetriever not initialized due to LLM or vectorstore failure.")

    # 5. Load QA Chain
    if llm and retriever: # Ensure retriever is also initialized for QA chain
        prompt_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and to the point.

Context: {context}
Question: {question}

Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Use load_qa_chain for the final answer generation, not direct llm.invoke
        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
        print("QA chain loaded.")
    else:
        qa_chain = None
        print("QA chain not loaded due to LLM, vectorstore, or retriever initialization failure.")


def get_rag_answer(user_query):
    global vectorstore, qa_chain, retriever # Include retriever in global scope
    if not vectorstore or not qa_chain or not retriever: # Check all components
        return "RAG components not fully initialized. Please check the server setup and logs."

    print(f"Performing advanced retrieval for query: '{user_query}'...")
    try:
        # Use MultiQueryRetriever to get relevant documents
     
        docs = retriever.invoke(user_query)
        print(f"MultiQueryRetriever retrieved {len(docs)} unique documents.")

        # Combine document content for the LLM's context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Get answer from the QA chain
        
       
        answer_result = qa_chain.invoke({"input_documents": docs, "question": user_query})

        # Extract the answer from the chain's output
        
        if isinstance(answer_result, dict) and 'output_text' in answer_result:
            answer = answer_result['output_text']
        else:
            # Fallback for unexpected output format, convert to string
            answer = str(answer_result)

        return answer
    except Exception as e:
        print(f"Error during RAG query: {e}")
        return f"An error occurred during query processing: {str(e)}. Please check server logs."
