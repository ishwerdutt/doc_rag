import google.generativeai as genai

# Set your API key
<<<<<<< HEAD
genai.configure(api_key="AIzaSyD_CSx_yBaWq4iNfJmYUUI0LxGVl5MOq_4")  # Or use os.getenv("GOOGLE_API_KEY")
=======
genai.configure(api_key="api")  # Or use os.getenv("GOOGLE_API_KEY")
>>>>>>> 89d9e9afc74571784ddbc16ce4e2aa624c1f6fe5

# List all models
models = genai.list_models()
for model in models:
    print(model.name)
