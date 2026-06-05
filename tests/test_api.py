import sys
import os
import unittest

# Add src/ to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fastapi.testclient import TestClient
from app import app

class TestScoreAPI(unittest.TestCase):

	def setUp(self):
		self.client = TestClient(app)

	def test_api_score_endpoint_success(self):
		payload = [
			{
				"id": 1,
				"prompt": "Exact match check",
				"reference_answer": "  Apple  ",
				"model_answer": "apple"
			},
			{
				"id": 2,
				"prompt": "Missing answer check",
				"reference_answer": "banana",
				"model_answer": ""
			},
			{
				"id": 3,
				"prompt": "Levenshtein check",
				"reference_answer": "kitten",
				"model_answer": "sitting"
			},
			{
				"id": 4,
				"prompt": "Test completely wrong answer",
				"reference_answer": "a",
				"model_answer": "b"
			}
		]

		response = self.client.post("/score", json=payload)
		self.assertEqual(response.status_code, 200)
		
		data = response.json()
		self.assertIn("results", data)
		self.assertIn("summary", data)

		results = data["results"]
		summary = data["summary"]

		self.assertEqual(len(results), 4)
		# Assert exact match
		self.assertEqual(results[0]["id"], 1)
		self.assertEqual(results[0]["score"], 1.0)
		self.assertEqual(results[0]["reason"], "Exact match")

		# Assert missing model answer
		self.assertEqual(results[1]["id"], 2)
		self.assertEqual(results[1]["score"], 0.0)
		self.assertEqual(results[1]["reason"], "Model answer is missing")

		# Assert Levenshtein score (kitten/sitting = 4/7)
		self.assertEqual(results[2]["id"], 3)
		self.assertAlmostEqual(results[2]["score"], 4.0 / 7.0)
		self.assertEqual(results[2]["reason"], "Partial match")

		# Assert no rules satisfied
		self.assertEqual(results[3]["id"], 4)
		self.assertEqual(results[3]["score"], 0.0)
		self.assertEqual(results[3]["reason"], "No rules satisfied")

		# Assert summary
		self.assertEqual(summary["no_items"], 4)
		self.assertEqual(summary["no_exact"], 1)
		self.assertEqual(summary["no_failed"], 2)  # banana missing and wrong answer
		self.assertAlmostEqual(summary["avg_score"], (1.0 + 0.0 + 4.0 / 7.0 + 0.0) / 4.0)

	def test_api_empty_payload(self):
		response = self.client.post("/score", json=[])
		self.assertEqual(response.status_code, 200)
		data = response.json()
		self.assertEqual(data["results"], [])
		self.assertEqual(data["summary"]["no_items"], 0)
		self.assertEqual(data["summary"]["avg_score"], 0.0)

	def test_api_all_missing_answers(self):
		payload = [
			{"id": 10, "prompt": "q1", "reference_answer": "", "model_answer": "ans1"},
			{"id": 11, "prompt": "q2", "reference_answer": "ans2", "model_answer": ""}
		]
		response = self.client.post("/score", json=payload)
		self.assertEqual(response.status_code, 200)
		data = response.json()
		results = data["results"]
		summary = data["summary"]

		self.assertEqual(len(results), 2)
		self.assertEqual(results[0]["score"], 0.0)
		self.assertEqual(results[0]["reason"], "Reference answer is missing")
		self.assertEqual(results[1]["score"], 0.0)
		self.assertEqual(results[1]["reason"], "Model answer is missing")

		self.assertEqual(summary["no_items"], 2)
		self.assertEqual(summary["no_exact"], 0)
		self.assertEqual(summary["no_failed"], 2)
		self.assertEqual(summary["avg_score"], 0.0)

	def test_api_all_exact_matches(self):
		payload = [
			{"id": 20, "prompt": "q1", "reference_answer": "  Same  ", "model_answer": "same"},
			{"id": 21, "prompt": "q2", "reference_answer": "Apple", "model_answer": "  apple  "}
		]
		response = self.client.post("/score", json=payload)
		self.assertEqual(response.status_code, 200)
		data = response.json()
		results = data["results"]
		summary = data["summary"]

		self.assertEqual(len(results), 2)
		self.assertEqual(results[0]["score"], 1.0)
		self.assertEqual(results[0]["reason"], "Exact match")
		self.assertEqual(results[1]["score"], 1.0)
		self.assertEqual(results[1]["reason"], "Exact match")

		self.assertEqual(summary["no_items"], 2)
		self.assertEqual(summary["no_exact"], 2)
		self.assertEqual(summary["no_failed"], 0)
		self.assertEqual(summary["avg_score"], 1.0)

	def test_api_levenshtein_extreme_cases(self):
		payload = [
			{"id": 30, "prompt": "q1", "reference_answer": "test", "model_answer": "tent"},
			{"id": 31, "prompt": "q2", "reference_answer": "matplotlib", "model_answer": "matplotli"}
		]
		response = self.client.post("/score", json=payload)
		self.assertEqual(response.status_code, 200)
		data = response.json()
		results = data["results"]
		summary = data["summary"]

		self.assertEqual(len(results), 2)
		self.assertAlmostEqual(results[0]["score"], 0.75)
		self.assertEqual(results[0]["reason"], "Partial match")
		self.assertAlmostEqual(results[1]["score"], 0.9)
		self.assertEqual(results[1]["reason"], "Partial match")

		self.assertEqual(summary["no_items"], 2)
		self.assertEqual(summary["no_exact"], 0)
		self.assertEqual(summary["no_failed"], 0)
		self.assertAlmostEqual(summary["avg_score"], (0.75 + 0.9) / 2.0)

	def test_api_invalid_payload(self):
		# Missing required fields like model_answer
		payload = [
			{
				"id": 1,
				"prompt": "Incomplete payload",
				"reference_answer": "some reference"
			}
		]
		response = self.client.post("/score", json=payload)
		self.assertEqual(response.status_code, 422)  # Unprocessable Entity (validation error)

if __name__ == "__main__":
	unittest.main()
