# app.py
import os
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question,
        max_tokens=50
    )
    return jsonify({"answer": response.choices[0].text.strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

