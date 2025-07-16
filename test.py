import google.generativeai as genai

# Set your API key
genai.configure(api_key="AIzaSyCMDMzTUoLLrkTH7ALeNYdrX4sJ6U786_0")  # Or use os.getenv("GOOGLE_API_KEY")

# List all models
models = genai.list_models()
for model in models:
    print(model.name)