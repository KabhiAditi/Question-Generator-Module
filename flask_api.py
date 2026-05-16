from flask import Flask, request, jsonify
from question_generator_module import generate_question

app = Flask(__name__)

@app.route("/generate-question", methods=["POST"])
def generate_question_api():
    try:
        payload = request.get_json()
        result = generate_question(payload)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
