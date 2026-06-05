import sys
import os
import unittest
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models import ModelResponsePair, ScoringResult
from rules import (
	CatchMissingRule,
	ExactMatchRule,
	LevenshteinSimilarityRule,
	NormalizeTextPreprocessor
)
from scorer import Score


class TestModels(unittest.TestCase):

	def test_scoring_result_creation(self):
		res = ScoringResult(id="123", satisfied=True, score=0.8, reason="test reason")
		self.assertEqual(res.id, 123)
		self.assertEqual(res.score, 0.8)
		self.assertEqual(res.reason, "test reason")
		self.assertTrue(res[0])
		self.assertEqual(res[1], 0.8)

		# JSON serialization
		self.assertEqual(res.json(), {
			"id": 123,
			"score": 0.8,
			"reason": "test reason"
		})

	def test_model_response_pair_creation(self):
		pair = ModelResponsePair(id="456", prompt="hello", reference_answer="world", model_answer="world")
		self.assertEqual(pair.id, 456)
		self.assertEqual(pair.prompt, "hello")
		self.assertEqual(pair.reference_answer, "world")
		self.assertEqual(pair.model_answer, "world")


class TestPreprocessors(unittest.TestCase):

	def test_normalize_text_preprocessor(self):
		preprocessor = NormalizeTextPreprocessor()
		pair = ModelResponsePair(id=1, prompt="hi", reference_answer="  Hello World  ", model_answer="hello world")
		preprocessor.preprocess(pair)
		self.assertEqual(pair.reference_answer, "hello world")
		self.assertEqual(pair.model_answer, "hello world")


class TestRules(unittest.TestCase):

	def test_catch_missing_rule(self):
		rule = CatchMissingRule()
		
		# Missing model answer
		pair1 = ModelResponsePair(id=1, prompt="", reference_answer="abc", model_answer="")
		satisfied, score = rule.score(pair1)
		self.assertTrue(satisfied)
		self.assertEqual(score, 0.0)

		# Missing reference answer
		pair2 = ModelResponsePair(id=2, prompt="", reference_answer="", model_answer="abc")
		satisfied, score = rule.score(pair2)
		self.assertTrue(satisfied)
		self.assertEqual(score, 0.0)

		# Normal case
		pair3 = ModelResponsePair(id=3, prompt="", reference_answer="abc", model_answer="xyz")
		satisfied, score = rule.score(pair3)
		self.assertFalse(satisfied)

	def test_exact_match_rule(self):
		rule = ExactMatchRule()

		# Match
		pair1 = ModelResponsePair(id=1, prompt="", reference_answer="abc", model_answer="abc")
		satisfied, score = rule.score(pair1)
		self.assertTrue(satisfied)
		self.assertEqual(score, 1.0)

		# No match
		pair2 = ModelResponsePair(id=2, prompt="", reference_answer="abc", model_answer="xyz")
		satisfied, score = rule.score(pair2)
		self.assertFalse(satisfied)

	def test_levenshtein_similarity_rule(self):
		rule = LevenshteinSimilarityRule()

		# Empty strings match
		pair1 = ModelResponsePair(id=1, prompt="", reference_answer="", model_answer="")
		satisfied, score = rule.score(pair1)
		self.assertTrue(satisfied)
		self.assertEqual(score, 1.0)

		# Partial match
		pair2 = ModelResponsePair(id=2, prompt="", reference_answer="kitten", model_answer="sitting")
		satisfied, score = rule.score(pair2)
		self.assertTrue(satisfied)
		# Levenshtein distance between kitten and sitting is 3. max_len is 7.
		# score = 1.0 - 3/7 = 4/7 approx 0.5714
		self.assertAlmostEqual(score, 4.0 / 7.0)

		# Completely different (distance = max_len => score = 0)
		pair3 = ModelResponsePair(id=3, prompt="", reference_answer="a", model_answer="b")
		satisfied, score = rule.score(pair3)
		self.assertFalse(satisfied)
		self.assertEqual(score, 0.0)


class TestScore(unittest.TestCase):

	def test_score_aggregation(self):
		score_runner = Score(
			rules=[
				CatchMissingRule(),
				ExactMatchRule(),
				LevenshteinSimilarityRule()
			],
			preprocessors=[
				NormalizeTextPreprocessor()
			],
			default_score=0
		)

		pairs = [
			ModelResponsePair(id="1", prompt="", reference_answer="  apple  ", model_answer="apple"),
			ModelResponsePair(id="2", prompt="", reference_answer="banana", model_answer=""),
			ModelResponsePair(id="3", prompt="", reference_answer="kitten", model_answer="sitting"),
			ModelResponsePair(id="4", prompt="", reference_answer="a", model_answer="b")
		]

		for pair in pairs:
			pair.score = score_runner.score(pair)

		self.assertEqual(pairs[0].score.score, 1.0)  # Exact match (after normalization)
		self.assertEqual(pairs[0].score.reason, "Exact match")

		self.assertEqual(pairs[1].score.score, 0.0)  # Catch missing
		self.assertEqual(pairs[1].score.reason, "Model answer is missing")

		self.assertAlmostEqual(pairs[2].score.score, 4.0 / 7.0)  # Partial match Levenshtein
		self.assertEqual(pairs[2].score.reason, "Partial match")

		self.assertEqual(pairs[3].score.score, 0.0)  # Default score / no rule matches
		self.assertEqual(pairs[3].score.reason, "No rules satisfied")

		# Check summary properties
		summary = score_runner.summary
		self.assertEqual(summary["no_items"], 4)
		self.assertEqual(summary["no_exact"], 1)
		self.assertEqual(summary["no_failed"], 2)  # pairs[1] and pairs[3]
		self.assertAlmostEqual(summary["avg_score"], (1.0 + 0.0 + 4.0 / 7.0 + 0.0) / 4.0)


class TestIntegrationMain(unittest.TestCase):

	def test_end_to_end_execution(self):
		# Create predetermined input data
		input_data = [
			{
				"id": "1",
				"prompt": "Test exact match",
				"reference_answer": "  Apple Pie  ",
				"model_answer": "apple pie"
			},
			{
				"id": "2",
				"prompt": "Test missing answer",
				"reference_answer": "banana",
				"model_answer": ""
			},
			{
				"id": "3",
				"prompt": "Test partial match",
				"reference_answer": "kitten",
				"model_answer": "sitting"
			},
			{
				"id": "4",
				"prompt": "Test completely wrong answer",
				"reference_answer": "a",
				"model_answer": "b"
			}
		]

		# Write to a temporary file
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
			json.dump(input_data, tmp_file)
			tmp_filepath = tmp_file.name

		try:
			# Get path of src/main.py relative to this test file
			main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/main.py'))

			# Run main.py as a subprocess
			result = subprocess.run(
				[sys.executable, main_py_path, tmp_filepath],
				capture_output=True,
				text=True,
				check=True
			)

			# Parse stdout as JSON
			output = json.loads(result.stdout)

			# Verify results structure
			self.assertIn("results", output)
			self.assertIn("summary", output)

			results = output["results"]
			summary = output["summary"]

			# Check individual result records
			self.assertEqual(len(results), 4)

			# Pair 1: Exact match after normalization
			self.assertEqual(results[0]["id"], 1)
			self.assertEqual(results[0]["score"], 1.0)
			self.assertEqual(results[0]["reason"], "Exact match")

			# Pair 2: Catch missing rule
			self.assertEqual(results[1]["id"], 2)
			self.assertEqual(results[1]["score"], 0.0)
			self.assertEqual(results[1]["reason"], "Model answer is missing")

			# Pair 3: Levenshtein partial match
			self.assertEqual(results[2]["id"], 3)
			self.assertAlmostEqual(results[2]["score"], 4.0 / 7.0)
			self.assertEqual(results[2]["reason"], "Partial match")

			# Pair 4: No rules matched (falls back to default score 0)
			self.assertEqual(results[3]["id"], 4)
			self.assertEqual(results[3]["score"], 0.0)
			self.assertEqual(results[3]["reason"], "No rules satisfied")

			# Verify summary aggregation
			self.assertEqual(summary["no_items"], 4)
			self.assertEqual(summary["no_exact"], 1)
			self.assertEqual(summary["no_failed"], 2)  # banana and wrong answers
			self.assertAlmostEqual(summary["avg_score"], (1.0 + 0.0 + 4.0 / 7.0 + 0.0) / 4.0)

		finally:
			# Cleanup temporary file
			if os.path.exists(tmp_filepath):
				os.remove(tmp_filepath)

	def test_empty_input_file(self):
		input_data = []
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
			json.dump(input_data, tmp_file)
			tmp_filepath = tmp_file.name

		try:
			main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/main.py'))
			result = subprocess.run(
				[sys.executable, main_py_path, tmp_filepath],
				capture_output=True,
				text=True,
				check=True
			)
			output = json.loads(result.stdout)
			self.assertEqual(output["results"], [])
			self.assertEqual(output["summary"]["no_items"], 0)
			self.assertEqual(output["summary"]["no_exact"], 0)
			self.assertEqual(output["summary"]["no_failed"], 0)
			self.assertEqual(output["summary"]["avg_score"], 0.0)
		finally:
			if os.path.exists(tmp_filepath):
				os.remove(tmp_filepath)

	def test_all_missing_answers(self):
		input_data = [
			{"id": "10", "prompt": "q1", "reference_answer": "", "model_answer": "ans1"},
			{"id": "11", "prompt": "q2", "reference_answer": "ans2", "model_answer": ""}
		]
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
			json.dump(input_data, tmp_file)
			tmp_filepath = tmp_file.name

		try:
			main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/main.py'))
			result = subprocess.run(
				[sys.executable, main_py_path, tmp_filepath],
				capture_output=True,
				text=True,
				check=True
			)
			output = json.loads(result.stdout)
			results = output["results"]
			summary = output["summary"]

			self.assertEqual(len(results), 2)
			self.assertEqual(results[0]["score"], 0.0)
			self.assertEqual(results[0]["reason"], "Reference answer is missing")
			self.assertEqual(results[1]["score"], 0.0)
			self.assertEqual(results[1]["reason"], "Model answer is missing")

			self.assertEqual(summary["no_items"], 2)
			self.assertEqual(summary["no_exact"], 0)
			self.assertEqual(summary["no_failed"], 2)
			self.assertEqual(summary["avg_score"], 0.0)
		finally:
			if os.path.exists(tmp_filepath):
				os.remove(tmp_filepath)

	def test_all_exact_matches(self):
		input_data = [
			{"id": "20", "prompt": "q1", "reference_answer": "  Same  ", "model_answer": "same"},
			{"id": "21", "prompt": "q2", "reference_answer": "Apple", "model_answer": "  apple  "}
		]
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
			json.dump(input_data, tmp_file)
			tmp_filepath = tmp_file.name

		try:
			main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/main.py'))
			result = subprocess.run(
				[sys.executable, main_py_path, tmp_filepath],
				capture_output=True,
				text=True,
				check=True
			)
			output = json.loads(result.stdout)
			results = output["results"]
			summary = output["summary"]

			self.assertEqual(len(results), 2)
			self.assertEqual(results[0]["score"], 1.0)
			self.assertEqual(results[0]["reason"], "Exact match")
			self.assertEqual(results[1]["score"], 1.0)
			self.assertEqual(results[1]["reason"], "Exact match")

			self.assertEqual(summary["no_items"], 2)
			self.assertEqual(summary["no_exact"], 2)
			self.assertEqual(summary["no_failed"], 0)
			self.assertEqual(summary["avg_score"], 1.0)
		finally:
			if os.path.exists(tmp_filepath):
				os.remove(tmp_filepath)

	def test_levenshtein_extreme_cases(self):
		input_data = [
			{"id": "30", "prompt": "q1", "reference_answer": "test", "model_answer": "tent"},
			{"id": "31", "prompt": "q2", "reference_answer": "matplotlib", "model_answer": "matplotli"}
		]
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
			json.dump(input_data, tmp_file)
			tmp_filepath = tmp_file.name

		try:
			main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/main.py'))
			result = subprocess.run(
				[sys.executable, main_py_path, tmp_filepath],
				capture_output=True,
				text=True,
				check=True
			)
			output = json.loads(result.stdout)
			results = output["results"]
			summary = output["summary"]

			self.assertEqual(len(results), 2)
			self.assertAlmostEqual(results[0]["score"], 0.75)
			self.assertEqual(results[0]["reason"], "Partial match")
			self.assertAlmostEqual(results[1]["score"], 0.9)
			self.assertEqual(results[1]["reason"], "Partial match")

			self.assertEqual(summary["no_items"], 2)
			self.assertEqual(summary["no_exact"], 0)
			self.assertEqual(summary["no_failed"], 0)
			self.assertAlmostEqual(summary["avg_score"], (0.75 + 0.9) / 2.0)
		finally:
			if os.path.exists(tmp_filepath):
				os.remove(tmp_filepath)

	def test_invalid_cli_arguments(self):
		main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/main.py'))
		
		# Run with no arguments
		result1 = subprocess.run(
			[sys.executable, main_py_path],
			capture_output=True,
			text=True,
			check=True
		)
		self.assertIn("python main.py <input file>", result1.stdout)

		# Run with too many arguments
		result2 = subprocess.run(
			[sys.executable, main_py_path, "arg1", "arg2"],
			capture_output=True,
			text=True,
			check=True
		)
		self.assertIn("python main.py <input file>", result2.stdout)


if __name__ == "__main__":
	unittest.main()
