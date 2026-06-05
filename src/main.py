import json
import sys
from models import ModelResponsePair
from scorer import Score
from rules import (
	CatchMissingRule,
	ExactMatchRule,
	LevenshteinSimilarityRule,
	NormalizeTextPreprocessor
)

def main():
	argv = sys.argv
	if len(argv) != 2:
		print("python main.py <input file>")
		return

	score = Score(
		rules = [
			CatchMissingRule(),
			ExactMatchRule(),
			LevenshteinSimilarityRule()
		],
		preprocessors = [
			NormalizeTextPreprocessor()
		],
		default_score=0
	)

	with open(argv[1], 'r') as infile:
		data = json.load(infile)
		pairs = [ModelResponsePair(**item) for item in data]
		
		for pair in pairs:
			pair.score = score.score(pair)

		output = {
			"results": [pair.score.json() for pair in pairs],
			"summary": score.summary
		}
		print(json.dumps(output, indent=4))

if __name__ == "__main__":
	main()