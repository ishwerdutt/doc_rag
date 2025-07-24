import google.generativeai as genai

# Set your API key
genai.configure(api_key="AIzaSyD_CSx_yBaWq4iNfJmYUUI0LxGVl5MOq_4")  # Or use os.getenv("GOOGLE_API_KEY")

# List all models
models = genai.list_models()
for model in models:
    print(model.name)