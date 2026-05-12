from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# Render will inject this securely from your settings
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/YOUR_USERNAME/vivace-lora" # <-- UPDATE THIS

@app.route("/")
def home():
    # This serves your HTML page
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # This safely handles the conversation behind the scenes
    user_input = request.json.get("message")
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": user_input})
        result = response.json()
        
        # Check if Hugging Face threw an error (like "Model too large")
        if isinstance(result, dict) and "error" in result:
            return jsonify({"response": f"Error: {result['error']}"})
            
        # Clean up the output
        if isinstance(result, list):
            ai_text = result[0].get("generated_text", "").replace(user_input, "").strip()
            return jsonify({"response": ai_text})
            
        return jsonify({"response": "Hmm, I didn't understand that response format."})
        
    except Exception as e:
        return jsonify({"response": f"Server Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
