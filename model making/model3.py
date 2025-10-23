import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. LOAD AND PREPARE DATA ====================
print("Loading character data...")
df = pd.read_csv("m_d_char.csv")
print(f"Loaded: {df.shape[0]} characters with {df.shape[1]-1} features")

# ==================== 2. FEATURE ENGINEERING ====================
# Separate features and target
character_col = df['character'] = df['name']
feature_cols = [c for c in df.columns if c not in ['name', 'character', 'genre']]
genre_col = 'genre'

# Create feature matrix
X_features = df[feature_cols].values
y_target = df['character'].values

# Add genre as binary features
genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
X_with_genre = np.hstack([X_features, genre_dummies.values])
all_questions = feature_cols + list(genre_dummies.columns)

print(f"Total features available: {len(all_questions)}")

# ==================== 3. HANDLE CLASS IMBALANCE ====================
# Duplicate ALL characters multiple times to ensure sufficient samples per class
print(f"Expanding dataset for better training (each character x5)...")
df_copies = [df.copy() for _ in range(5)]
df = pd.concat(df_copies, ignore_index=True)

# Rebuild X and y with expanded dataset
X_features = df[feature_cols].values
genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
X_with_genre = np.hstack([X_features, genre_dummies.values])
y_target = df['character'].values

print(f"Dataset expanded to {len(df)} samples ({len(np.unique(y_target))} unique characters)")
print(f"Samples per character: {len(df) // len(np.unique(y_target))}")

# ==================== 4. INTELLIGENT FEATURE SELECTION ====================
def calculate_information_gain(X, y, feature_idx):
    """Calculate information gain for a feature"""
    feature = X[:, feature_idx]
    
    # Calculate parent entropy
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total = len(y)
    parent_entropy = -sum((count/total) * np.log2(count/total) for count in class_counts)
    
    # Calculate weighted child entropy
    unique_values = np.unique(feature)
    child_entropy = 0
    for value in unique_values:
        mask = feature == value
        subset_y = y[mask]
        weight = len(subset_y) / total
        
        unique_sub, sub_counts = np.unique(subset_y, return_counts=True)
        sub_total = len(subset_y)
        sub_entropy = -sum((count/sub_total) * np.log2(count/sub_total) for count in sub_counts if count > 0)
        child_entropy += weight * sub_entropy
    
    return parent_entropy - child_entropy

print("\nCalculating information gain for all features...")
info_gains = []
for idx in range(X_with_genre.shape[1]):
    ig = calculate_information_gain(X_with_genre, y_target, idx)
    info_gains.append((all_questions[idx], ig, idx))

info_gains.sort(key=lambda x: x[1], reverse=True)

print("\nTop 40 Most Informative Features:")
for i, (question, gain, idx) in enumerate(info_gains[:40], 1):
    print(f"{i:2d}. {question:30s} (IG: {gain:.4f})")

# Define question sets
QUESTION_SETS = [7, 15, 20, 25, 30, 35, 40]
X_data = {}
selected_questions = {}

for q_count in QUESTION_SETS:
    selected_questions[q_count] = [q[0] for q in info_gains[:q_count]]
    indices = [q[2] for q in info_gains[:q_count]]
    X_data[q_count] = X_with_genre[:, indices]
    print(f"Selected {q_count} questions for model training")

# ==================== 5. MODEL COMPARISON ====================
print("\n" + "="*70)
print("COMPARING DIFFERENT MODELS")
print("="*70)

# Split data for each question set
train_test_splits = {}
for q_count in QUESTION_SETS:
    X_train, X_test, y_train, y_test = train_test_split(
        X_data[q_count], y_target, test_size=0.2, random_state=42, stratify=y_target
    )
    train_test_splits[q_count] = (X_train, X_test, y_train, y_test)

# Define models to test
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=500,
        learning_rate=0.5,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=25,
        min_samples_split=2,
        random_state=42,
        class_weight='balanced'
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'
    )
}

# Store results for each question set
results = {q_count: {} for q_count in QUESTION_SETS}

for q_count in QUESTION_SETS:
    print(f"\nTesting {q_count}-Question Models:")
    X_train, X_test, y_train, y_test = train_test_splits[q_count]
    for name, model in models.items():
        try:
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model_copy.predict(X_train))
            test_acc = accuracy_score(y_test, model_copy.predict(X_test))
            cv_scores = cross_val_score(model_copy, X_train, y_train, cv=3)
            results[q_count][name] = {
                'model': model_copy,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"  {name:20s} - Train: {train_acc:.3%}, Test: {test_acc:.3%}, CV: {cv_scores.mean():.3%} (±{cv_scores.std():.3%})")
        except Exception as e:
            print(f"  {name:20s} - SKIPPED (Error: {str(e)[:50]})")

# Select best models for each question set
best_models = {}
for q_count in QUESTION_SETS:
    best_model_name = max(results[q_count].items(), key=lambda x: x[1]['cv_mean'])[0]
    best_models[q_count] = {
        'name': best_model_name,
        'model': results[q_count][best_model_name]['model'],
        'test_acc': results[q_count][best_model_name]['test_acc']
    }
    print(f"\n{'='*70}")
    print(f"BEST MODEL FOR {q_count} QUESTIONS: {best_model_name}")
    print(f"  Test Accuracy: {results[q_count][best_model_name]['test_acc']:.3%}")
print(f"{'='*70}")

# ==================== 6. SAVE MODELS ====================
SAVE_DIR = "guess_game_models_enhanced_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

for q_count in QUESTION_SETS:
    joblib.dump(best_models[q_count]['model'], f"{SAVE_DIR}/model_{q_count}.pkl")
    joblib.dump(selected_questions[q_count], f"{SAVE_DIR}/questions_{q_count}.pkl")
    joblib.dump(best_models[q_count]['name'], f"{SAVE_DIR}/model_{q_count}_name.pkl")

joblib.dump(list(df['character'].unique()), f"{SAVE_DIR}/characters.pkl")
print(f"\nModels and data saved to '{SAVE_DIR}/'")

# ==================== 7. ENHANCED GAME LOGIC ====================
def ask_question(question):
    """Ask a yes/no question with validation"""
    clean_q = question.replace('_', ' ').replace('genre ', '')
    while True:
        answer = input(f"  {clean_q}? (y/n): ").strip().lower()
        if answer in ('y', 'yes', '1', 'true'):
            return 1
        if answer in ('n', 'no', '0', 'false'):
            return 0
        print("    Please answer 'y' or 'n'")

def get_top_predictions(model, features, n=3):
    """Get top N predictions with probabilities"""
    probas = model.predict_proba([features])[0]
    top_indices = np.argsort(probas)[-n:][::-1]
    predictions = []
    for idx in top_indices:
        predictions.append({
            'character': model.classes_[idx],
            'confidence': probas[idx]
        })
    return predictions

def play_game():
    """Main game loop"""
    print("\n" + "="*70)
    print("           🎮 GUESS THE CHARACTER GAME 🎮")
    print("              64 Marvel & DC Heroes")
    print("="*70)
    print("\nUsing AI Models:")
    for q_count in QUESTION_SETS:
        print(f"  {q_count} Questions: {best_models[q_count]['name']}")
    print("\nThink of a character from the list, and I'll try to guess it!")
    print("Answer each question with 'y' (yes) or 'n' (no)")

    answers = []
    for round_num, q_count in enumerate(QUESTION_SETS, 1):
        print(f"\n{'-'*70}")
        print(f"ROUND {round_num}: {q_count} Questions")
        print(f"{'-'*70}")
        
        # Ask questions up to q_count, but only those not asked in previous rounds
        start_idx = QUESTION_SETS[round_num-2] if round_num > 1 else 0
        questions_to_ask = selected_questions[q_count][start_idx:q_count]
        
        for i, question in enumerate(questions_to_ask, start_idx + 1):
            print(f"\nQuestion {i}/{q_count}:")
            answers.append(ask_question(question))
        
        # Get top 3 predictions
        top_3 = get_top_predictions(best_models[q_count]['model'], answers, n=3)
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_num} RESULTS ({q_count} Questions):")
        print(f"{'='*70}")
        for i, pred in enumerate(top_3, 1):
            print(f"{i}. {pred['character']:25s} (Confidence: {pred['confidence']:.1%})")
        
        top_guess = top_3[0]['character']
        print(f"\n🎯 My guess: **{top_guess}**")
        
        is_correct = input("\nIs this correct? (y/n): ").strip().lower()
        if is_correct.startswith('y'):
            print(f"\n🎉 I GOT IT IN {q_count} QUESTIONS! Character: {top_guess}")
            return
        
        if q_count == QUESTION_SETS[-1]:
            # Final round
            actual = input("\nWhat character were you thinking of? ").strip()
            print(f"\n😅 You stumped me! The answer was: {actual}")
            print("Thanks for playing! This helps improve the AI.")
            return

# ==================== 8. RUN THE GAME ====================
if __name__ == "__main__":
    print("\n✅ Training complete!")
    input("\nPress ENTER to start the game...")
    
    while True:
        play_game()
        
        play_again = input("\n\nPlay again? (y/n): ").strip().lower()
        if not play_again.startswith('y'):
            print("\n👋 Thanks for playing! Goodbye!")
            break