from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# Render will inject your secret token securely from your Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Connected directly to your Dino101 model!
API_URL = "https://api-inference.huggingface.co/models/Dino101/vivace-lora"

@app.route("/")
def home():
    # This serves your HTML page from the 'templates' folder
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
        
        # 1. If Hugging Face returns a bad server code (like 503 or 404), catch it!
        if response.status_code != 200:
            # We slice the error text to 150 characters so it doesn't flood your chatbox
            return jsonify({"response": f"Hugging Face Error ({response.status_code}): {response.text[:150]}..."})
        
        # 2. If the status is 200 OK, it is safe to read the data
        result = response.json()
        
        # 3. Check if Hugging Face threw a formatted error (like "Model is currently loading")
        if isinstance(result, dict) and "error" in result:
            return jsonify({"response": f"Status: {result['error']}"})
            
        # 4. Clean up the output to only show Vivace's new text
        if isinstance(result, list):
            ai_text = result[0].get("generated_text", "").replace(user_input, "").strip()
            return jsonify({"response": ai_text})
            
        return jsonify({"response": "Hmm, I didn't understand the format Hugging Face sent back."})
        
    except Exception as e:
        # If the code completely breaks, show why
        return jsonify({"response": f"Backend Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
