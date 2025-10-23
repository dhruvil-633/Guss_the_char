# viz_optimized.py – Visualize ALL 7/15/20/25/30/35/40-question models
import joblib
import os
import numpy as np
from sklearn.tree import export_graphviz, plot_tree, export_text
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')          # non-interactive backend

# ==================== CONFIG ====================
MODEL_DIR = "guess_game_models_enhanced_V2"   # <-- your folder
OUT_DIR   = "tree_visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

print("="*70)
print("DECISION TREE VISUALIZATION TOOL – ALL MODELS")
print("="*70)

# ==================== LOAD COMMON DATA ====================
print("\nLoading common data …")
class_names = joblib.load(os.path.join(MODEL_DIR, "characters.pkl"))
print(f"   Characters: {len(class_names)}")

# ==================== HELPER FUNCTIONS ====================

def get_model_type(model):
    name = type(model).__name__
    if "RandomForest" in name:      return "RandomForest"
    if "GradientBoosting" in name:  return "GradientBoosting"
    if "AdaBoost" in name:          return "AdaBoost"
    if "XGB" in name:               return "XGBoost"
    if "LGBM" in name:              return "LightGBM"
    if "CatBoost" in name:          return "CatBoost"
    if "DecisionTree" in name:      return "DecisionTree"
    return "Unknown"

def get_best_tree(model):
    """Return a single sklearn-style tree (best in RF, first in GB, etc.)"""
    mtype = get_model_type(model)

    if mtype == "RandomForest":
        best_idx = max(range(len(model.estimators_)),
                       key=lambda i: model.estimators_[i].tree_.node_count)
        tree = model.estimators_[best_idx]
        print(f"   RandomForest → tree {best_idx}/{len(model.estimators_)} "
              f"({tree.tree_.node_count} nodes)")
        return tree

    if mtype == "GradientBoosting":
        tree = model.estimators_[0, 0]
        print(f"   GradientBoosting → first estimator ({tree.tree_.node_count} nodes)")
        return tree

    if mtype == "AdaBoost":
        best_idx = np.argmax(model.estimator_weights_)
        tree = model.estimators_[best_idx]
        print(f"   AdaBoost → estimator {best_idx} "
              f"(weight {model.estimator_weights_[best_idx]:.4f})")
        return tree

    if mtype in ("XGBoost", "LightGBM", "CatBoost"):
        print(f"   {mtype} → returning booster object")
        return model

    if mtype == "DecisionTree":
        print(f"   DecisionTree → single tree ({model.tree_.node_count} nodes)")
        return model

    raise ValueError(f"Unsupported model type: {mtype}")

# -------------------- visualisation helpers --------------------
def export_tree_dot(tree, feat, cls, filename, max_depth=6):
    export_graphviz(
        tree,
        out_file=filename,
        feature_names=[f.replace('_', ' ').title() for f in feat],
        class_names=[c.replace('_', ' ') for c in cls],
        filled=True, rounded=True, special_characters=True,
        max_depth=max_depth, fontname="Arial",
        impurity=True, proportion=True
    )
    print(f"   DOT → {os.path.basename(filename)}")
    print(f"      SVG: dot -Tsvg {filename} -o {filename.replace('.dot','.svg')}")
    print(f"      PNG: dot -Tpng {filename} -o {filename.replace('.dot','.png')} -Gdpi=300")

def plot_tree_matplotlib(tree, feat, cls, filename, max_depth=5):
    fig, ax = plt.subplots(figsize=(30, 20))
    clean_feat = [f.replace('_', ' ').replace('genre ', 'Genre: ').title() for f in feat]
    clean_cls  = [c.replace('_', ' ') for c in cls]
    plot_tree(tree, ax=ax,
              feature_names=clean_feat, class_names=clean_cls,
              filled=True, rounded=True, fontsize=7,
              max_depth=max_depth, impurity=True, proportion=False)
    plt.title(f"Decision Tree (depth ≤ {max_depth})", fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   PNG (full) → {os.path.basename(filename)}")

def create_simplified_visualization(tree, feat, cls, filename, depth=3):
    fig, ax = plt.subplots(figsize=(20, 12))
    clean_feat = [f.replace('_', ' ').replace('genre ', 'Genre: ').title() for f in feat]
    clean_cls  = [c.replace('_', ' ') for c in cls]
    plot_tree(tree, ax=ax,
              feature_names=clean_feat, class_names=clean_cls,
              filled=True, rounded=True, fontsize=10,
              max_depth=depth, impurity=False, proportion=True)
    plt.title(f"Simplified Tree (depth ≤ {depth})", fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   PNG (simple) → {os.path.basename(filename)}")

def export_tree_text(tree, feat, cls, filename, max_depth=10):
    from sklearn.tree import _tree
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\nDECISION TREE RULES\n" + "="*80 + "\n\n")
        f.write(f"Nodes: {tree.tree_.node_count} | Max depth: {tree.tree_.max_depth}\n")
        f.write(f"Features: {len(feat)} | Classes: {len(cls)}\n\n")

        def recurse(node, depth, path=""):
            if depth > max_depth:
                f.write("  "*depth + "... (truncated)\n")
                return
            indent = "  "*depth
            if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feat[tree.tree_.feature[node]]
                clean = name.replace('_', ' ').replace('genre ', 'Genre: ').title()
                samples = tree.tree_.n_node_samples[node]
                f.write(f"{indent}├─ {clean} = YES? (samples: {samples})\n")
                recurse(tree.tree_.children_left[node], depth+1,
                        path + f"{clean}=YES → ")
                f.write(f"{indent}└─ {clean} = NO?\n")
                recurse(tree.tree_.children_right[node], depth+1,
                        path + f"{clean}=NO → ")
            else:
                vals = tree.tree_.value[node][0]
                idx  = np.argmax(vals)
                conf = vals[idx] / np.sum(vals)
                f.write(f"{indent}GUESS: {cls[idx]}\n")
                f.write(f"{indent}   (samples: {tree.tree_.n_node_samples[node]}, "
                        f"confidence: {conf:.1%})\n")
                f.write(f"{indent}   Path: {path}\n\n")
        recurse(0, 0)

        if hasattr(tree, 'feature_importances_'):
            f.write("\n" + "="*80 + "\nFEATURE IMPORTANCE (top 10)\n" + "="*80 + "\n")
            imp = sorted(zip(feat, tree.feature_importances_),
                         key=lambda x: x[1], reverse=True)[:10]
            for i, (f_name, val) in enumerate(imp, 1):
                f.write(f"{i:2d}. {f_name:30s} {val:.4f}\n")
    print(f"   TXT (rules) → {os.path.basename(filename)}")

def export_sklearn_text_format(tree, feat, filename):
    txt = export_text(tree,
                      feature_names=[f.replace('_', ' ').title() for f in feat],
                      max_depth=8, spacing=3)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("SKLEARN TEXT REPRESENTATION\n" + "="*80 + "\n\n")
        f.write(txt)
    print(f"   TXT (sklearn) → {os.path.basename(filename)}")

# ==================== PROCESS EVERY MODEL ====================
# The folder contains: model_7, model_15, … model_40  + matching questions_*.pkl
QUESTIONS = [7, 15, 20, 25, 30, 35, 40]

for q in QUESTIONS:
    model_file   = os.path.join(MODEL_DIR, f"model_{q}.pkl")
    name_file    = os.path.join(MODEL_DIR, f"model_{q}_name.pkl")
    questions_file = os.path.join(MODEL_DIR, f"questions_{q}.pkl")

    if not os.path.exists(model_file):
        print(f"\nSkipping {q}-Q model – file not found: {model_file}")
        continue

    print("\n" + "="*70)
    print(f"PROCESSING {q}-QUESTION MODEL")
    print("="*70)

    model   = joblib.load(model_file)
    name    = joblib.load(name_file)
    feat    = joblib.load(questions_file)

    print(f"   Model name : {name}")
    print(f"   Questions  : {len(feat)}")

    tree = get_best_tree(model)
    if tree is None:
        print("   Could not extract a tree – skipping visualisations")
        continue

    prefix = f"tree_{q}q"

    # 1. DOT
    export_tree_dot(tree, feat, class_names,
                    os.path.join(OUT_DIR, f"{prefix}.dot"), max_depth=7)

    # 2. Full PNG
    plot_tree_matplotlib(tree, feat, class_names,
                         os.path.join(OUT_DIR, f"{prefix}_full.png"), max_depth=5)

    # 3. Simple PNG
    create_simplified_visualization(tree, feat, class_names,
                                    os.path.join(OUT_DIR, f"{prefix}_simple.png"), depth=3)

    # 4. Rules TXT
    export_tree_text(tree, feat, class_names,
                     os.path.join(OUT_DIR, f"{prefix}_rules.txt"), max_depth=8)

    # 5. sklearn TXT
    export_sklearn_text_format(tree, feat,
                               os.path.join(OUT_DIR, f"{prefix}_sklearn.txt"))

    print(f"   All {q}-Q visualisations created!")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*70)
print("ALL DONE!")
print("="*70)
print(f"\nOutput folder: {OUT_DIR}/")
print("\nFiles per model (replace <n> with 7/15/20/25/30/35/40):")
print("   tree_<n>q.dot          → Graphviz source")
print("   tree_<n>q_full.png     → Full tree (depth ≤5)")
print("   tree_<n>q_simple.png   → Quick view (depth ≤3)")
print("   tree_<n>q_rules.txt    → Human-readable rules + importance")
print("   tree_<n>q_sklearn.txt  → sklearn export_text")
print("\nGraphviz conversion examples:")
print("   dot -Tsvg tree_visualizations/tree_35q.dot -o tree_visualizations/tree_35q.svg")
print("   dot -Tpng tree_visualizations/tree_40q.dot -o tree_visualizations/tree_40q_hq.png -Gdpi=300")
print("\nOpen any *_simple.png for a fast overview, or the SVG for zoomable quality.")