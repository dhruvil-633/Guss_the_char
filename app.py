from flask import Flask, render_template, request, jsonify, url_for
from huggingface_hub import hf_hub_download
import joblib
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Your Hugging Face repo for models only
HF_REPO_ID = "dhruvil-633/Guess_the_char"

# Local path for questions and characters
LOCAL_DATA_PATH = "guess_game_models_enhanced_v2"

# Define question sets
QUESTION_SETS = [7, 15, 20, 25, 30, 35, 40]

# ------------------- Model + Data Loader -------------------

def load_model_from_hf(filename):
    """
    Download a model file from Hugging Face if not cached.
    """
    file_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    return joblib.load(file_path)


# ------------------- Load all models and local data -------------------

models = {}
questions = {}

for q_count in QUESTION_SETS:
    print(f"🔄 Loading model_{q_count}.pkl from Hugging Face and local questions...")
    
    # Load model from Hugging Face
    models[q_count] = load_model_from_hf(f"model_{q_count}.pkl")
    
    # Load questions from local folder
    question_file = os.path.join(LOCAL_DATA_PATH, f"questions_{q_count}.pkl")
    questions[q_count] = joblib.load(question_file)

# Load characters from local folder
chars = joblib.load(os.path.join(LOCAL_DATA_PATH, "characters.pkl"))

print("✅ All models (HF) and local data loaded successfully!")

# ------------------- Routes -------------------

@app.route("/")
def index():
    return render_template("index.html", victory_url=url_for('victory'), failed_url=url_for('failed'))

@app.route("/questions/<int:q_count>")
def get_questions(q_count):
    if q_count not in QUESTION_SETS:
        return jsonify({"error": f"Invalid question count. Must be one of {QUESTION_SETS}"}), 400
    return jsonify(questions[q_count])

@app.route("/guess/<int:q_count>", methods=["POST"])
def guess(q_count):
    if q_count not in QUESTION_SETS:
        return jsonify({"error": f"Invalid question count. Must be one of {QUESTION_SETS}"}), 400
    answers = request.json["answers"]
    model = models[q_count]
    pred = model.predict([answers])[0]
    return jsonify({"guess": pred})

@app.route("/victory")
def victory():
    return render_template("victory.html")

@app.route("/failed")
def failed():
    return render_template("failed.html")

@app.route("/characters")
def get_characters():
    return jsonify(chars)

# ------------------- Local dev only -------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000, ))
    app.run(host="0.0.0.0", port=port, debug=True)
