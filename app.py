from flask import Flask, render_template, request, jsonify, url_for
from huggingface_hub import hf_hub_download
import joblib
import os
import sys
import traceback

print("=" * 60, flush=True)
print("🚀 STARTING CHARACTER GUESSER APPLICATION", flush=True)
print("=" * 60, flush=True)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Your Hugging Face repo for models only
HF_REPO_ID = "dhruvil-633/Guess_the_char"

# Local path for questions and characters
LOCAL_DATA_PATH = "guess_game_models_enhanced_v2"

# Define question sets
QUESTION_SETS = [7, 15, 20, 25, 30, 35, 40]

# Cache directory for HuggingFace models
HF_HOME = os.environ.get('HF_HOME', '/app/.cache/huggingface')

# ------------------- Model + Data Loader -------------------

def load_model_from_hf(filename):
    """
    Load a model file from Hugging Face cache (pre-downloaded during build).
    """
    try:
        print(f"📥 Loading {filename} from HuggingFace cache...", flush=True)
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=filename,
            cache_dir=HF_HOME
        )
        print(f"✅ Found {filename} at {file_path}", flush=True)
        model = joblib.load(file_path)
        print(f"✅ Model {filename} loaded successfully", flush=True)
        return model
    except Exception as e:
        print(f"❌ ERROR loading {filename}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise

# ------------------- Load all models and local data -------------------

models = {}
questions = {}
chars = []

try:
    print(f"📂 Working directory: {os.getcwd()}", flush=True)
    print(f"📂 HF_HOME: {HF_HOME}", flush=True)
    print(f"📂 Contents: {os.listdir('.')}", flush=True)
    
    for q_count in QUESTION_SETS:
        print(f"\n🔄 Loading model_{q_count}.pkl from Hugging Face cache...", flush=True)
        
        # Load model from Hugging Face cache (already downloaded during build)
        models[q_count] = load_model_from_hf(f"model_{q_count}.pkl")
        
        # Load questions from local folder
        question_file = os.path.join(LOCAL_DATA_PATH, f"questions_{q_count}.pkl")
        print(f"📂 Loading {question_file}...", flush=True)
        
        if not os.path.exists(question_file):
            raise FileNotFoundError(f"Question file not found: {question_file}")
            
        questions[q_count] = joblib.load(question_file)
        print(f"✅ Loaded questions_{q_count}.pkl", flush=True)

    # Load characters from local folder
    chars_file = os.path.join(LOCAL_DATA_PATH, "characters.pkl")
    print(f"\n📂 Loading {chars_file}...", flush=True)
    
    if not os.path.exists(chars_file):
        raise FileNotFoundError(f"Characters file not found: {chars_file}")
        
    chars = joblib.load(chars_file)
    print(f"✅ Loaded {len(chars)} characters", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("✅ ALL MODELS AND DATA LOADED SUCCESSFULLY!", flush=True)
    print(f"   - {len(models)} models loaded", flush=True)
    print(f"   - {len(questions)} question sets loaded", flush=True)
    print(f"   - {len(chars)} characters loaded", flush=True)
    print("=" * 60, flush=True)

except Exception as e:
    print("\n" + "=" * 60, flush=True)
    print("❌ FATAL ERROR DURING INITIALIZATION:", flush=True)
    print(str(e), flush=True)
    traceback.print_exc()
    print("=" * 60, flush=True)
    sys.stdout.flush()
    # Set empty defaults so app can at least start
    models = {}
    questions = {}
    chars = []

# ------------------- Routes -------------------

@app.route("/")
def index():
    """Main page"""
    if not models:
        return "Error: Models not loaded. Check server logs.", 500
    return render_template("index.html", victory_url=url_for('victory'), failed_url=url_for('failed'))

@app.route("/health")
def health():
    """Health check endpoint for monitoring"""
    is_healthy = len(models) == len(QUESTION_SETS) and len(chars) > 0
    return jsonify({
        "status": "healthy" if is_healthy else "unhealthy",
        "models_loaded": len(models),
        "expected_models": len(QUESTION_SETS),
        "questions_loaded": len(questions),
        "characters_loaded": len(chars)
    }), 200 if is_healthy else 500

@app.route("/questions/<int:q_count>")
def get_questions(q_count):
    """Get questions for a specific question count"""
    if q_count not in QUESTION_SETS:
        return jsonify({"error": f"Invalid question count. Must be one of {QUESTION_SETS}"}), 400
    if q_count not in questions:
        return jsonify({"error": "Questions not loaded"}), 500
    return jsonify(questions[q_count])

@app.route("/guess/<int:q_count>", methods=["POST"])
def guess(q_count):
    """Make a prediction based on answers"""
    if q_count not in QUESTION_SETS:
        return jsonify({"error": f"Invalid question count. Must be one of {QUESTION_SETS}"}), 400
    if q_count not in models:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        answers = request.json["answers"]
        model = models[q_count]
        pred = model.predict([answers])[0]
        return jsonify({"guess": pred})
    except Exception as e:
        print(f"❌ Prediction error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/victory")
def victory():
    """Victory page"""
    return render_template("victory.html")

@app.route("/failed")
def failed():
    """Failed page"""
    return render_template("failed.html")

@app.route("/characters")
def get_characters():
    """Get all characters"""
    if not chars:
        return jsonify({"error": "Characters not loaded"}), 500
    return jsonify(chars)

# ------------------- Local dev only -------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🌐 Starting Flask on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=True)