# Insurance Claim Prediction (Machine Learning + Flask)

A web app that estimates the chance that a person will make an insurance claim,
based on age, sex, BMI, daily steps, number of children and smoking status.

## Results (hold-out test set, 268 people)

| Model | Test accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 85.8% | 0.881 | 0.905 |
| Random Forest | 98.1% | 0.984 | 0.997 |
| **Gradient Boosting (tuned, used in the app)** | **98.9%** | **0.990** | **0.999** |

Exact numbers can differ slightly if you retrain with a different scikit-learn version.
They are always saved in `metrics.json` and shown on the app's *Model performance* page.

## What was fixed compared with the original project

| Problem in the original | Fix |
|---|---|
| `model.pkl` crashed on newer scikit-learn (`ValueError: incompatible dtype`) | Model is retrained with `train_model.py`; the app shows a clear message if the file is incompatible |
| `charges` was used as an input, but it is only known *after* a claim (data leakage) | Removed. Accuracy stays ~98%, so it was never needed |
| Opening `/predict` in the browser crashed (`prediction` undefined) | GET now redirects to the form |
| Values were read from the form by position, so reordering fields broke predictions | Fields are read by name and validated |
| No input validation (letters or empty values caused a 500 error) | Type, range and required checks with clear messages |
| Only a yes/no answer | Shows the claim probability and a gauge |
| One model, no proper evaluation | 3 models compared with 5-fold cross-validation, tuned with GridSearchCV, tested on unseen data |
| Training code was a messy notebook export that plotted charts and blocked | Clean script that saves charts to `static/img` |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python train_model.py          # trains and saves model.pkl, metrics.json and charts
python app.py                  # open http://127.0.0.1:5000
```

Run the tests: `python -m unittest discover tests`

## Project structure

```
app.py              Flask app (pages, validation, JSON API)
train_model.py      Data cleaning, model comparison, tuning, saving
data/insurance.csv  Dataset (1,338 rows)
templates/          index (form), result, performance pages
static/             CSS and generated charts
tests/test_app.py   9 automated tests
```

## JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":45,"sex":"male","bmi":30,"steps":4000,"children":2,"smoker":"no"}'
# {"claim": false, "claim_probability": 0.0123, "notes": []}
```

## Limitations (mention these in your viva)

- The dataset is small and its patterns are very clean, which is why accuracy is so high. Real insurance data is noisier.
- Inputs outside the training range (age 18-64, BMI 16-53, steps 3,000-10,010) trigger a warning.
- This is a demonstration project, not a tool for real insurance decisions.

## Possible extensions

SHAP explanations, a database to store predictions, Docker deployment, deployment on Render or Hugging Face Spaces.
