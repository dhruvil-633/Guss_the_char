# train_and_play.py
# Guess the Character: 64 Marvel + DC Heroes
# Two Models: 7-Q and 14-Q (genre = last 2 questions)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ------------------- 1. Load Data -------------------
print("Loading char.csv...")
df_raw = pd.read_csv("marvel_dc_characters.csv")  # 64 rows, 22 columns
print(f"Loaded: {df_raw.shape[0]} characters")

# ------------------- 2. Rename Columns -------------------
df = df_raw.rename(columns={
    "name": "character",
    "male": "is_male",
    "superhero": "is_superhero",
    "detective": "is_detective",
    "comedian": "is_comedian",
    "billionaire": "is_billionaire",
    "from_earth": "is_from_earth",
    "team_member": "is_team_member",
    "sidekick": "has_sidekick"
})

# ------------------- 3. One-Hot Encode Genre -------------------
print("Encoding genre...")
genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
df = pd.concat([df.drop('genre', axis=1), genre_dummies], axis=1)

# ------------------- 4. Fix Rare Characters (appear once) -------------------
char_counts = Counter(df['character'])
rare_chars = [c for c, cnt in char_counts.items() if cnt == 1]

if rare_chars:
    print(f"Found {len(rare_chars)} characters appearing only once. Duplicating...")
    extra_rows = df[df['character'].isin(rare_chars)].copy()
    df = pd.concat([df, extra_rows], ignore_index=True)
    print(f"New total rows: {len(df)}")
else:
    print("All characters appear at least twice.")

# ------------------- 5. Select Features via Information Gain -------------------
binary_cols = [c for c in df.columns if c not in ['character'] + list(genre_dummies.columns)]
genre_cols = list(genre_dummies.columns)

def entropy(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def info_gain(col):
    p1 = col.mean()
    if p1 == 0 or p1 == 1: return 0
    H_parent = entropy(p1)
    p0 = 1 - p1
    H0 = entropy(p0) * p0
    H1 = entropy(p1) * p1
    return H_parent - (H0 + H1)

print("Calculating information gain for binary features...")
gains = df[binary_cols].apply(info_gain)
top_binary = gains.sort_values(ascending=False).index.tolist()

# Round 1: Top 7
TOP_7 = top_binary[:7]

# Round 2: Top 12 binary + 2 genre
TOP_12_BINARY = top_binary[:12]
TOP_14 = TOP_12_BINARY + genre_cols[:2]

print("\nTOP 7 QUESTIONS:")
for i, q in enumerate(TOP_7, 1):
    print(f"  {i}. {q}")

print("\nTOP 14 QUESTIONS (last 2 are genre):")
for i, q in enumerate(TOP_14, 1):
    print(f"  {i}. {q}")

# ------------------- 6. Prepare X and y -------------------
X7 = df[TOP_7].values
X14 = df[TOP_14].values
y = df['character'].values

# ------------------- 7. Train-Test Split (NO stratify) -------------------
X7_train, X7_val, y7_train, y7_val = train_test_split(X7, y, test_size=0.2, random_state=42)
X14_train, X14_val, y14_train, y14_val = train_test_split(X14, y, test_size=0.2, random_state=42)

# ------------------- 8. Train Models -------------------
print("\nTraining 7-Question Model...")
model_7 = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model_7.fit(X7_train, y7_train)
acc7 = accuracy_score(y7_val, model_7.predict(X7_val))
print(f"7-Q Accuracy: {acc7:.3%}")

print("Training 14-Question Model...")
model_14 = RandomForestClassifier(
    n_estimators=800,
    max_depth=16,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model_14.fit(X14_train, y14_train)
acc14 = accuracy_score(y14_val, model_14.predict(X14_val))
print(f"14-Q Accuracy: {acc14:.3%}")

# ------------------- 9. Save Everything -------------------
SAVE_DIR = "guess_game_models"
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(model_7, f"{SAVE_DIR}/model_7.pkl")
joblib.dump(model_14, f"{SAVE_DIR}/model_14.pkl")
joblib.dump(TOP_7, f"{SAVE_DIR}/questions_7.pkl")
joblib.dump(TOP_14, f"{SAVE_DIR}/questions_14.pkl")
joblib.dump(df['character'].tolist(), f"{SAVE_DIR}/characters.pkl")

print(f"\nAll files saved to '{SAVE_DIR}/'")

# ------------------- 10. Play the Game -------------------
def ask(q):
    while True:
        a = input(f"{q}? (y/n): ").strip().lower()
        if a in ("y", "yes"): return 1
        if a in ("n", "no"): return 0
        print("Please answer 'y' or 'n'")

def play_game():
    print("\n" + "="*70)
    print("   GUESS THE CHARACTER – 64 MARVEL & DC HEROES")
    print("="*70)

    # Round 1: 7 questions
    print("\nROUND 1: 7 Questions")
    ans7 = []
    for q in TOP_7:
        ans7.append(ask(q))
    
    guess1 = model_7.predict([ans7])[0]
    print(f"\nAfter 7 questions → I think it's **{guess1}**")
    if input("Is this correct? (y/n): ").lower().startswith('y'):
        print("I WIN IN 7 QUESTIONS!")
        return

    # Round 2: 7 more (5 binary + 2 genre)
    print("\nROUND 2: 7 More Questions")
    extra = [ask(q) for q in TOP_14[7:12]]
    genre_ans = [ask(g.replace("genre_", "Genre: ")) for g in genre_cols[:2]]
    ans14 = ans7 + extra + genre_ans

    guess2 = model_14.predict([ans14])[0]
    print(f"\nAfter 14 questions → I think it's **{guess2}**")
    if input("Correct now? (y/n): ").lower().startswith('y'):
        print("GOT IT IN 14!")
    else:
        print("Better luck next time!")

# ------------------- RUN GAME -------------------
if __name__ == "__main__":
    print("\nTraining complete. Starting game...")
    play_game()