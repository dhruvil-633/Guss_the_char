"""
Pre-download all models from HuggingFace during Docker build.
This ensures models are cached and don't need to be downloaded at runtime.
"""
from huggingface_hub import hf_hub_download
import os
import sys

HF_REPO_ID = "dhruvil-633/Guess_the_char"
QUESTION_SETS = [7, 15, 20, 25, 30, 35, 40]

# Use the same cache directory as the app
HF_HOME = os.environ.get('HF_HOME', '/app/.cache/huggingface')

print("=" * 60)
print("📥 PRE-DOWNLOADING MODELS FROM HUGGINGFACE")
print("=" * 60)
print(f"Repository: {HF_REPO_ID}")
print(f"Cache directory: {HF_HOME}")
print()

success_count = 0
failed_models = []

for q_count in QUESTION_SETS:
    filename = f"model_{q_count}.pkl"
    try:
        print(f"⏳ Downloading {filename}...", flush=True)
        file_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=filename,
            cache_dir=HF_HOME
        )
        print(f"✅ {filename} cached at {file_path}", flush=True)
        
        # Verify file size
        file_size = os.path.getsize(file_path)
        print(f"   File size: {file_size / (1024*1024):.2f} MB", flush=True)
        success_count += 1
        
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}", flush=True)
        failed_models.append(filename)

print()
print("=" * 60)
if failed_models:
    print(f"⚠️  WARNING: {len(failed_models)} models failed to download:")
    for model in failed_models:
        print(f"   - {model}")
    print()
    print("The application may not work correctly!")
    sys.exit(1)  # Fail the build if models don't download
else:
    print(f"✅ SUCCESS: All {success_count} models downloaded and cached!")
print("=" * 60)