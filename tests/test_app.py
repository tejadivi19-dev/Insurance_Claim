"""Run with:  python -m pytest -q    (or: python -m unittest discover tests)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import app  # noqa: E402

GOOD = {"age": "45", "sex": "male", "bmi": "30", "steps": "4000", "children": "2", "smoker": "no"}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_home_loads(self):
        self.assertEqual(self.c.get("/").status_code, 200)

    def test_get_predict_redirects(self):
        r = self.c.get("/predict")
        self.assertEqual(r.status_code, 302)

    def test_valid_prediction_page(self):
        r = self.c.post("/predict", data=GOOD)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"make a claim", r.data)

    def test_missing_field_rejected(self):
        bad = {k: v for k, v in GOOD.items() if k != "bmi"}
        r = self.c.post("/predict", data=bad)
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"BMI is required", r.data)

    def test_non_numeric_and_range(self):
        r = self.c.post("/predict", data={**GOOD, "age": "abc", "bmi": "500"})
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"Age must be a whole number", r.data)
        self.assertIn(b"BMI must be between", r.data)

    def test_api_probability(self):
        r = self.c.post("/api/predict", json=GOOD)
        self.assertEqual(r.status_code, 200)
        p = r.get_json()["claim_probability"]
        self.assertTrue(0.0 <= p <= 1.0)

    def test_api_rejects_bad_json(self):
        self.assertEqual(self.c.post("/api/predict", data="nope").status_code, 400)

    def test_known_patterns(self):
        # low steps + smoker + high BMI -> claim; active non-smoker with normal BMI -> no claim
        high = self.c.post("/api/predict", json={**GOOD, "smoker": "yes", "bmi": "38", "steps": "3200"}).get_json()
        low = self.c.post("/api/predict", json={**GOOD, "smoker": "no", "bmi": "24", "steps": "9000", "children": "3"}).get_json()
        self.assertGreater(high["claim_probability"], low["claim_probability"])

    def test_performance_and_health(self):
        self.assertEqual(self.c.get("/performance").status_code, 200)
        self.assertEqual(self.c.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
