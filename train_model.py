"""
Train, compare and save the insurance-claim prediction model.

Run:  python train_model.py

Outputs
  model.pkl                  best model (scikit-learn Pipeline)
  metrics.json               scores, feature ranges, versions (read by the web app)
  static/img/*.png           charts shown on the "Model performance" page
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "insurance.csv"
IMG_DIR = BASE / "static" / "img"
SEED = 42

# 'charges' is deliberately NOT used: it is only known after a claim is made,
# so using it to predict the claim would be data leakage.
# 'region' is dropped: it has almost no correlation with the target.
FEATURES = ["age", "sex", "bmi", "steps", "children", "smoker"]
TARGET = "insuranceclaim"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET]).drop_duplicates()
    print(f"Loaded {before} rows, using {len(df)} after cleaning.")
    return df


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = df[FEATURES], df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=300, random_state=SEED)),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=SEED)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    print("\n5-fold cross-validation on the training set:")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        results[name] = {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
        print(f"  {name:20s} {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Small hyper-parameter search for the best model family
    best_name = max(results, key=lambda n: results[n]["cv_mean"])
    print(f"\nBest family: {best_name} -> tuning with GridSearchCV")
    grids = {
        "Random Forest": {
            "randomforestclassifier__n_estimators": [200, 400],
            "randomforestclassifier__max_depth": [None, 8, 12],
            "randomforestclassifier__min_samples_leaf": [1, 3],
        },
        "Gradient Boosting": {
            "gradientboostingclassifier__n_estimators": [100, 200],
            "gradientboostingclassifier__learning_rate": [0.05, 0.1],
            "gradientboostingclassifier__max_depth": [2, 3],
        },
        "Logistic Regression": {"logisticregression__C": [0.1, 1, 10]},
    }
    search = GridSearchCV(candidates[best_name], grids[best_name], cv=cv, scoring="accuracy", n_jobs=1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"  best params: {search.best_params_}")

    # Final evaluation on data the model has never seen
    for name, model in candidates.items():
        m = best_model if name == best_name else model.fit(X_train, y_train)
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        results[name].update(
            test_accuracy=float(accuracy_score(y_test, pred)),
            precision=float(precision_score(y_test, pred)),
            recall=float(recall_score(y_test, pred)),
            f1=float(f1_score(y_test, pred)),
            roc_auc=float(roc_auc_score(y_test, proba)),
        )
    print("\nHold-out test results:")
    for name, r in results.items():
        print(f"  {name:20s} acc={r['test_accuracy']:.3f} prec={r['precision']:.3f} "
              f"rec={r['recall']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")

    # ---------- charts ----------
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=["No claim", "Claim"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(IMG_DIR / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(results)
    accs = [results[n]["test_accuracy"] * 100 for n in names]
    bars = ax.barh(names, accs, color=["#0F766E" if n == best_name else "#9FB3C8" for n in names])
    ax.set_xlim(max(0, min(accs) - 10), 100)
    ax.set_xlabel("Test accuracy (%)")
    for b, v in zip(bars, accs):
        ax.text(v - 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", ha="right", color="white", fontweight="bold")
    ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(IMG_DIR / "model_comparison.png", dpi=130); plt.close(fig)

    final_est = best_model[-1]
    if hasattr(final_est, "feature_importances_"):
        imp = final_est.feature_importances_
    else:
        imp = np.abs(final_est.coef_[0]); imp = imp / imp.sum()
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh([FEATURES[i] for i in order], imp[order] * 100, color="#14213D")
    ax.set_xlabel("Importance (%)"); ax.set_title("What drives the prediction")
    fig.tight_layout(); fig.savefig(IMG_DIR / "feature_importance.png", dpi=130); plt.close(fig)

    # ---------- save ----------
    joblib.dump(best_model, BASE / "model.pkl")
    metrics = {
        "best_model": best_name,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in search.best_params_.items()},
        "features": FEATURES,
        "results": results,
        "feature_importance": {FEATURES[i]: float(imp[i]) for i in np.argsort(-imp)},
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
    }
    (BASE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved model.pkl, metrics.json and charts.")


if __name__ == "__main__":
    main()
