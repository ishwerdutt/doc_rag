from flask import Blueprint, render_template, request, jsonify
from rag_pipeline import get_rag_answer
from langchain_core.messages import HumanMessage, AIMessage # Still need these for converting incoming history
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    logger.info(f"Incoming GET request to / from {request.remote_addr}")
    logger.info("New chat session started.")
    return render_template('index.html')

@main_bp.route('/chat', methods=['POST'])
def chat_rag():
    logger.info(f"Incoming POST request to /chat from {request.remote_addr}")
    try:
        data = request.get_json(force=True)
        user_message = data.get('message')
        history_data = data.get('chat_history', [])

        if not user_message:
            logger.warning(f"POST /chat: No message provided by {request.remote_addr}.")
            return jsonify({
                "answer": "Please provide a message.",
                "updated_chat_history": history_data # Return original history if no message
            }), 400

        logger.info(f"POST /chat: User message from {request.remote_addr}: '{user_message}'")
        logger.info(f"POST /chat: History length received from frontend: {len(history_data)}")

        # Convert incoming serializable history (dicts) to LangChain message objects
        chat_history = []
        for msg in history_data:
            if msg.get('type') == 'human':
                chat_history.append(HumanMessage(content=msg.get('content', '')))
            elif msg.get('type') == 'ai':
                chat_history.append(AIMessage(content=msg.get('content', '')))
            # Optional: handle unexpected types or malformed messages by skipping or logging
            else:
                logger.warning(f"POST /chat: Unexpected message type or format in history: {msg}")


        answer, updated_history_lc = get_rag_answer(user_message, chat_history)

        # Convert updated LangChain message objects back to serializable dicts for frontend
        serializable_history = []
        for msg in updated_history_lc:
            # Ensure we're only extracting content and type, which are directly accessible
            if isinstance(msg, HumanMessage):
                serializable_history.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                serializable_history.append({"type": "ai", "content": msg.content})
            else:
                logger.error(f"POST /chat: Unexpected object type in updated_history_lc: {type(msg)}. Attempting to serialize anyway.")
                serializable_history.append({"type": "unknown", "content": str(msg)})


        logger.info(f"POST /chat: Answer generated for '{user_message[:50]}...': {answer[:50]}...")
        logger.debug(f"POST /chat: Full serializable history sent to frontend: {serializable_history}")

        return jsonify({
            "answer": answer,
            "updated_chat_history": serializable_history
        })

    except Exception as e:
        logger.exception(f"Error during RAG query for request from {request.remote_addr}:")
        
        return jsonify({
            "answer": f"An error occurred: {str(e)}. Please try again later.",
            "updated_chat_history": request.json.get('chat_history', []) if request.json else []
        }), 500