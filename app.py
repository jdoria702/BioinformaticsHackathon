# Imports
from flask import Flask, render_template, redirect, request
from flask_scss import Scss

# My App
# Flask is the hub that manages the routes to all the pages
app = Flask(__name__)
Scss(app)

# Home page
# Route to home page
# Home page can add tasks and retrieve tasks
@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

if __name__ in "__main__":
    app.run(debug=True)