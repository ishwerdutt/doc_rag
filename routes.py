from flask import Blueprint, render_template, request, jsonify
from rag_pipeline import get_rag_answer # Import the function to get RAG answer

# Create a Blueprint for the main routes
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Renders the main index page."""
    return render_template('index.html')

@main_bp.route('/query', methods=['POST'])
def query_rag():
    """Handles the RAG query from the frontend."""
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"answer": "Please provide a query."}), 400

    print(f"Received query: '{user_query}'")

    try:
        answer = get_rag_answer(user_query)
        print(f"Generated answer: '{answer}'")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error during RAG query: {e}")
        return jsonify({"answer": f"An error occurred: {str(e)}. Please check server logs."}), 500
