# Imports
from flask import Flask, render_template, redirect, request, jsonify
from flask_scss import Scss

# My App
# Flask is the hub that manages the routes to all the pages
app = Flask(__name__)

# Home page
# Route to home page
# Home page can add tasks and retrieve tasks
@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    # Placeholder agent response - replace with actual agent logic
    agent_response = f"Agent: I received your message: '{user_message}'. (This is a placeholder response.)"
    return jsonify({"response": agent_response})

if __name__ in "__main__":
    app.run(debug=True)