import os
import re
from flask import current_app
import psutil
import time
import threading
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Global variables to store initialized RAG components
vectorstore = None
qa_chain = None
llm = None
retriever = None

def extract_equipment_metadata(document_text, filename):
    """
    Extracts equipment metadata from document text using pattern matching.
    Returns a dictionary with structured equipment information.
    """
    metadata = {
        'source_file': filename,
        'equipment_name': None,
        'model_number': None,
        'manufacturer': None,
        'specifications': {},
        'features': [],
        'dimensions': {},
        'power_requirements': None,
        'operating_conditions': {},
        'safety_features': [],
        'maintenance_info': [],
        'document_type': None
    }
    
    # Equipment name patterns
    name_patterns = [
        r'(?:Equipment|Device|System|Machine|Unit|Instrument):\s*([^\n\r]+)',
        r'Model:\s*([A-Z0-9\-\s]+)',
        r'Product Name:\s*([^\n\r]+)',
        r'^([A-Z][A-Za-z0-9\s\-]{5,50})\s*(?:Series|Model|System)'
    ]
    
    # Model number patterns
    model_patterns = [
        r'Model\s*(?:No\.?|Number)?:\s*([A-Z0-9\-]+)',
        r'Part\s*(?:No\.?|Number)?:\s*([A-Z0-9\-]+)',
        r'Serial:\s*([A-Z0-9\-]+)'
    ]
    
    # Manufacturer patterns
    manufacturer_patterns = [
        r'Manufacturer:\s*([^\n\r]+)',
        r'Made by:\s*([^\n\r]+)',
        r'Company:\s*([^\n\r]+)'
    ]
    
    # Dimensions patterns
    dimension_patterns = [
        r'Dimensions?:\s*([0-9.]+)\s*(?:x|×)\s*([0-9.]+)\s*(?:x|×)?\s*([0-9.]*)\s*(mm|cm|m|in|ft)',
        r'Size:\s*([0-9.]+)\s*(?:x|×)\s*([0-9.]+)\s*(?:x|×)?\s*([0-9.]*)\s*(mm|cm|m|in|ft)',
        r'Length:\s*([0-9.]+)\s*(mm|cm|m|in|ft)',
        r'Width:\s*([0-9.]+)\s*(mm|cm|m|in|ft)',
        r'Height:\s*([0-9.]+)\s*(mm|cm|m|in|ft)'
    ]
    
    # Power requirements patterns
    power_patterns = [
        r'Power:\s*([0-9.]+)\s*(W|KW|V|A|VA)',
        r'Voltage:\s*([0-9.]+)\s*(V|VAC|VDC)',
        r'Current:\s*([0-9.]+)\s*(A|mA)',
        r'Frequency:\s*([0-9.]+)\s*(Hz)'
    ]
    
    # Operating conditions patterns
    operating_patterns = [
        r'Temperature:\s*([0-9.\-]+)\s*(?:to|-)?\s*([0-9.\-]*)\s*°?([CF])',
        r'Humidity:\s*([0-9.]+)%?\s*(?:to|-)?\s*([0-9.]*)%?',
        r'Pressure:\s*([0-9.]+)\s*(bar|psi|Pa|kPa)'
    ]
    
    # Features patterns
    feature_keywords = [
        'LCD display', 'LED indicator', 'touchscreen', 'wireless', 'bluetooth',
        'USB port', 'ethernet', 'digital interface', 'analog output',
        'battery powered', 'rechargeable', 'waterproof', 'dustproof',
        'calibration', 'self-test', 'auto-zero', 'data logging'
    ]
    
    # Safety features patterns
    safety_keywords = [
        'emergency stop', 'safety interlock', 'overload protection',
        'thermal protection', 'surge protection', 'isolation',
        'grounding', 'fuse protection', 'circuit breaker'
    ]
    
    text_lower = document_text.lower()
    
    # Extract equipment name
    for pattern in name_patterns:
        match = re.search(pattern, document_text, re.IGNORECASE | re.MULTILINE)
        if match and not metadata['equipment_name']:
            metadata['equipment_name'] = match.group(1).strip()
            break
    
    # Extract model number
    for pattern in model_patterns:
        match = re.search(pattern, document_text, re.IGNORECASE)
        if match:
            metadata['model_number'] = match.group(1).strip()
            break
    
    # Extract manufacturer
    for pattern in manufacturer_patterns:
        match = re.search(pattern, document_text, re.IGNORECASE)
        if match:
            metadata['manufacturer'] = match.group(1).strip()
            break
    
    # Extract dimensions
    for pattern in dimension_patterns:
        match = re.search(pattern, document_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 4:
                metadata['dimensions'] = {
                    'length': f"{groups[0]} {groups[3]}",
                    'width': f"{groups[1]} {groups[3]}",
                    'height': f"{groups[2]} {groups[3]}" if groups[2] else None,
                    'unit': groups[3]
                }
            break
    
    # Extract power requirements
    power_info = {}
    for pattern in power_patterns:
        matches = re.findall(pattern, document_text, re.IGNORECASE)
        for match in matches:
            power_info[f"{match[1].lower()}_value"] = match[0]
    if power_info:
        metadata['power_requirements'] = power_info
    
    # Extract operating conditions
    operating_info = {}
    for pattern in operating_patterns:
        match = re.search(pattern, document_text, re.IGNORECASE)
        if match:
            if 'temperature' in pattern.lower():
                operating_info['temperature'] = {
                    'min': match.group(1),
                    'max': match.group(2) if match.group(2) else match.group(1),
                    'unit': match.group(3)
                }
            elif 'humidity' in pattern.lower():
                operating_info['humidity'] = {
                    'min': match.group(1),
                    'max': match.group(2) if match.group(2) else match.group(1),
                    'unit': '%'
                }
    metadata['operating_conditions'] = operating_info
    
    # Extract features
    found_features = []
    for feature in feature_keywords:
        if feature.lower() in text_lower:
            found_features.append(feature)
    metadata['features'] = found_features
    
    # Extract safety features
    found_safety = []
    for safety in safety_keywords:
        if safety.lower() in text_lower:
            found_safety.append(safety)
    metadata['safety_features'] = found_safety
    
    # Determine document type
    doc_type_keywords = {
        'manual': ['manual', 'user guide', 'instruction'],
        'specification': ['specification', 'datasheet', 'technical data'],
        'installation': ['installation', 'setup', 'mounting'],
        'maintenance': ['maintenance', 'service', 'calibration'],
        'troubleshooting': ['troubleshooting', 'fault', 'error', 'problem']
    }
    
    for doc_type, keywords in doc_type_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            metadata['document_type'] = doc_type
            break
    
    return metadata

def enrich_document_chunks(chunks):
    """
    Enriches document chunks with extracted metadata.
    """
    enriched_chunks = []
    
    for chunk in chunks:
        # Get original metadata
        original_metadata = chunk.metadata.copy()
        
        # Extract equipment metadata from chunk content
        equipment_metadata = extract_equipment_metadata(chunk.page_content, 
                                                        original_metadata.get('source', 'unknown'))
        
        # Merge metadata
        enriched_metadata = {**original_metadata, **equipment_metadata}
        
        # Create metadata summary for better retrieval
        metadata_summary = create_metadata_summary(equipment_metadata)
        
        # Enhance chunk content with metadata context
        enhanced_content = f"""
EQUIPMENT INFO: {metadata_summary}

DOCUMENT CONTENT:
{chunk.page_content}
"""
        
        # Create new document with enhanced content and metadata
        from langchain_core.documents import Document
        enriched_chunk = Document(
            page_content=enhanced_content,
            metadata=enriched_metadata
        )
        
        enriched_chunks.append(enriched_chunk)
    
    return enriched_chunks

def create_metadata_summary(metadata):
    """
    Creates a concise summary of equipment metadata for embedding.
    """
    summary_parts = []
    
    if metadata.get('equipment_name'):
        summary_parts.append(f"Equipment: {metadata['equipment_name']}")
    
    if metadata.get('model_number'):
        summary_parts.append(f"Model: {metadata['model_number']}")
    
    if metadata.get('manufacturer'):
        summary_parts.append(f"Manufacturer: {metadata['manufacturer']}")
    
    if metadata.get('features'):
        summary_parts.append(f"Features: {', '.join(metadata['features'][:5])}")
    
    if metadata.get('dimensions'):
        dim = metadata['dimensions']
        if dim.get('length') and dim.get('width'):
            summary_parts.append(f"Size: {dim['length']} x {dim['width']}")
    
    if metadata.get('power_requirements'):
        power_info = []
        for key, value in metadata['power_requirements'].items():
            power_info.append(f"{key}: {value}")
        summary_parts.append(f"Power: {', '.join(power_info)}")
    
    if metadata.get('document_type'):
        summary_parts.append(f"Doc Type: {metadata['document_type']}")
    
    return " | ".join(summary_parts)

def setup_rag_components():
    """
    Initializes all components for the RAG pipeline with enhanced metadata.
    """
    global vectorstore, qa_chain, llm, retriever

    current_app.logger.info("[RAG Setup] Initiating components with CONTEXTUAL CHUNKING and MULTIQUERY RETRIEVAL.")
    current_app.logger.debug(f"[RAG Setup] Configured PDF_DATA_PATH: {current_app.config.get('PDF_DATA_PATH')}")
    current_app.logger.debug(f"[RAG Setup] Configured FAISS_INDEX_PATH: {current_app.config.get('FAISS_INDEX_PATH')}")

    pdf_data_path = current_app.config['PDF_DATA_PATH']
    faiss_index_path = current_app.config['FAISS_INDEX_PATH']
    api_key = current_app.config['GOOGLE_API_KEY']
    model_name = current_app.config.get('MODEL_NAME')

    # Validate API key presence
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        current_app.logger.error("[RAG Setup Error] GEMINI_API_KEY is missing or default. Initialization aborted.")
        raise ValueError("GEMINI_API_KEY is invalid or not set. Please update config.py to proceed.")

    # 1. Initialize Embeddings Model
    try:
        current_app.logger.info("[Embeddings] Initializing HuggingFace embeddings.")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
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
        current_app.logger.info(f"[FAISS] Loading existing enhanced index from '{faiss_index_path}'.")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        current_app.logger.info("[FAISS] Enhanced index loaded successfully.")
    else:
        current_app.logger.info(f"[FAISS] No index found. Creating new index with SEMANTIC CHUNKING.")
        os.makedirs(faiss_index_path, exist_ok=True)

        loaded_documents = []
        current_app.logger.info(f"[PDF] Processing PDFs with metadata extraction from '{pdf_data_path}'.")
        
        for filename in os.listdir(pdf_data_path):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_data_path, filename)
                current_app.logger.info(f"[PDF] Loading and analyzing document: '{filename}'.")
                try:
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    
                    full_text = "\n".join([doc.page_content for doc in docs])
                    equipment_metadata = extract_equipment_metadata(full_text, filename)
                    
                    current_app.logger.info(f"[Metadata] Extracted for '{filename}': Equipment={equipment_metadata.get('equipment_name', 'Unknown')}, Model={equipment_metadata.get('model_number', 'Unknown')}")
                    
                    for doc in docs:
                        doc.metadata.update(equipment_metadata)
                    
                    loaded_documents.extend(docs)
                except Exception as e:
                    current_app.logger.warning(f"[PDF] Could not load '{filename}': {e}")

        current_app.logger.info(f"[PDF] Loaded {len(loaded_documents)} documents with enhanced metadata.")

        current_app.logger.info("[Chunks] Splitting documents with SemanticChunker.")
        text_splitter = SemanticChunker(
            embeddings, breakpoint_threshold_type="percentile"
        )
        chunks = text_splitter.split_documents(loaded_documents)
        
        current_app.logger.info(f"[Metadata Enhancement] Enriching {len(chunks)} semantic chunks with metadata.")
        enriched_chunks = enrich_document_chunks(chunks)
        
        current_app.logger.info(f"[Chunks] Created {len(enriched_chunks)} enriched semantic chunks.")

        current_app.logger.info("[FAISS] Creating FAISS index from semantic chunks. This may take a while...")
        vectorstore = FAISS.from_documents(enriched_chunks, embeddings)
        vectorstore.save_local(faiss_index_path)
        current_app.logger.info(f"[FAISS] Index created and saved to '{faiss_index_path}'.")

    # 3. Initialize Gemini LLM
    try:
        current_app.logger.info(f"[LLM] Initializing Gemini model '{model_name}' for equipment queries.")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=current_app.config.get("TEMPERATURE")
        )
        current_app.logger.info("[LLM] Initialized successfully.")
    except Exception as e:
        current_app.logger.exception("[LLM] Initialization failed.")
        raise RuntimeError(f"Failed to initialize LLM: {e}")

    # 4. Initialize MultiQueryRetriever (without reranking)
    if llm and vectorstore:
        current_app.logger.info("[Retriever] Initializing base retriever and MultiQueryRetriever.")
        
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        compressor = FlashrankRerank(top_n=7)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever,
        )
        current_app.logger.info("[Retriever] Compression retriever initialized successfully.")
        # retriever = MultiQueryRetriever.from_llm(
        #     retriever=compression_retriever,
        #     llm=llm
        # )
        
        retriever = compression_retriever
        # current_app.logger.info("[Retriever] MultiQueryRetriever initialized successfully.")
    else:
        retriever = None
        current_app.logger.warning("[Retriever] Not initialized due to missing LLM or vectorstore.")
        raise RuntimeError("Retriever initialization failed.")

    # 5. Create Enhanced Conversational QA Chain
    if llm and retriever:
        current_app.logger.info("[QA Chain] Initializing conversational QA chain with MultiQueryRetriever.")
        prompt_template_str = """You are a clinical equipment specialist. Use the following context to answer questions about medical equipment.

The context includes both document content and extracted equipment metadata (specifications, features, dimensions, etc.).

Previous conversation:
{chat_history}

Context with Equipment Information:
{context}

Question: {input}

Instructions:
- Base your answer strictly on the provided context.
- Be precise with technical information, including model numbers, dimensions, and specifications.
- If the context does not contain the answer, state that the information is not available in the provided documents.
- Prioritize accuracy and conciseness.

Answer:"""
        PROMPT = ChatPromptTemplate.from_template(prompt_template_str)

        doc_chain = create_stuff_documents_chain(llm, prompt=PROMPT)
        qa_chain = create_retrieval_chain(retriever, doc_chain)
        
        current_app.logger.info("[QA Chain] Advanced conversational QA chain initialized successfully.")
    else:
        qa_chain = None
        current_app.logger.warning("[QA Chain] Initialization failed due to missing LLM or retriever.")
        raise RuntimeError("QA chain could not be loaded.")

    current_app.logger.info("[RAG Setup] All advanced RAG components initialized successfully.")
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    current_app.logger.info(f"[Memory] Advanced RAG setup memory usage: RSS={mem_info.rss / (1024*1024):.2f} MB")

def get_rag_answer(user_query: str, chat_history: list = None):
    """
    Processes a user query using the advanced RAG pipeline with reranking.
    """
    global vectorstore, qa_chain, retriever

    if chat_history is None:
        chat_history = []

    
    normalized_query = user_query.lower().strip().rstrip('?!.,')
    identity_phrases = {"who are you", "what are you", "what is your name", "what's up", "hello", "hi", "hey", "hi there", "hello there", "hey there", "hi hello", "hello hello", "hey hello", "hi hey", "hello hey", "hey hey", "hi hi", "hello hi", "hey hi", "How are you doing?", "whats up"}

    if normalized_query in identity_phrases:
        current_app.logger.info(f"[Identity] Predefined answer returned for query: '{user_query[:70]}...'")
        return "I am a specialized clinical equipment assistant that can help you find information about medical devices, their specifications, features, installation requirements, and operational details.", chat_history

    # Ensure RAG components are initialized
    if not all([vectorstore, qa_chain, retriever]):
        current_app.logger.error("[RAG Error] Advanced RAG components are not fully initialized.")
        return "Advanced RAG components not fully initialized. Please check the server setup and logs.", chat_history

    try:
        current_app.logger.info(f"[Advanced Query] Processing query with MultiQueryRetriever: '{user_query[:70]}...'")

        # Progress bar setup
        progress_bar = None
        progress_complete = threading.Event()
        
        def show_progress():
            nonlocal progress_bar
            progress_bar = tqdm(total=100, desc=" LLM Processing", unit="%", ncols=80)
            start_time = time.time()
            
            while not progress_complete.is_set():
                elapsed = time.time() - start_time
                # Simulate progress based on elapsed time (most queries take 2-10 seconds)
                progress = min(95, int((elapsed / 8.0) * 100))
                progress_bar.n = progress
                progress_bar.refresh()
                time.sleep(0.1)
            
            # Complete the progress bar
            progress_bar.n = 100
            progress_bar.refresh()
            progress_bar.close()

        # Start progress bar in separate thread
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.start()

        start_time = time.time()
        result = qa_chain.invoke({
            "input": user_query,
            "chat_history": chat_history
        })
        end_time = time.time()
        
        # Stop progress bar
        progress_complete.set()
        progress_thread.join()

        answer = result.get("answer")
        if answer is None:
            current_app.logger.warning(f"Advanced RAG Query Warning: 'answer' key not found in chain result.")
            answer = str(result)

        retrieved_docs = result.get("context", [])
        current_app.logger.info(f"[Retriever] Retrieved {len(retrieved_docs)} docs for query: '{user_query[:70]}...'")

        for i, doc in enumerate(retrieved_docs):
            equipment_name = doc.metadata.get('equipment_name', 'Unknown')
            model_number = doc.metadata.get('model_number', 'N/A')
            current_app.logger.debug(f"Retrieved Doc {i+1}: Equipment={equipment_name}, Model={model_number}")

        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=answer))

        elapsed_time = end_time - start_time
        current_app.logger.info(f"[Advanced Answer] Generated for '{user_query[:50]}...': '{answer[:50]}...'")
        current_app.logger.info(f"[Timing]  Total processing time: {elapsed_time:.2f} seconds")

        return answer, chat_history

    except Exception as e:
        # Ensure progress bar is cleaned up on error
        if 'progress_complete' in locals():
            progress_complete.set()
        current_app.logger.exception(f"[Advanced RAG Error] Exception during query processing: '{user_query[:50]}...'")
        return f"An error occurred during query processing with MultiQueryRetriever: {str(e)}. Please check server logs.", chat_history