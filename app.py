# app.py
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from config import Config
from routes import main_bp
from rag_pipeline import setup_rag_components
import os
import logging 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') 
app_logger = logging.getLogger(__name__) # Get a logger for app.py itself

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    app.register_blueprint(main_bp)

    # Ensure necessary directories exist
    os.makedirs(app.config['PDF_DATA_PATH'], exist_ok=True)
    os.makedirs(app.config['FAISS_INDEX_PATH'], exist_ok=True)

    # Setup RAG components when the app starts
    app_logger.info("App Creation: Initializing RAG components...")
    with app.app_context():
        try:
            setup_rag_components()
            app_logger.info("App Creation: RAG components initialized successfully.")
        except Exception as e:
            app_logger.critical(f"App Creation FATAL: Failed to initialize RAG components: {e}", exc_info=True)
           

    return app

# Create the app instance at the module level so Gunicorn can find it
app = create_app()


if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG) 
    app_logger.info("App Main: Starting Flask development server.")
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])