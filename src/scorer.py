from models import ModelResponsePair, ScoringResult

class Score:

	def __init__(self, rules, preprocessors=None, default_score=0.0):
		self.default_score = default_score
		self.rules = rules
		self.preprocessors = preprocessors or []
		self.scores = []

	def score(self, pair: ModelResponsePair) -> ScoringResult:
		# Execute all preprocessors first
		for preprocessor in self.preprocessors:
			preprocessor.preprocess(pair)

		# Evaluate scoring rules
		matched_result = ScoringResult(pair.id, False, self.default_score, "No rules satisfied")
		for r in self.rules:
			result = r.score(pair)
			satisfied, score = result
			if satisfied:
				matched_result = result
				break
		pair.reason = matched_result.reason
		self.scores.append(matched_result.score)
		return matched_result

	@property
	def avg_score(self) -> float:
		if not self.scores:
			return 0.0
		return sum(self.scores) / len(self.scores)

	@property
	def no_items(self) -> int:
		return len(self.scores)

	@property
	def no_exact(self) -> int:
		return sum(1 for s in self.scores if s == 1.0)

	@property
	def no_failed(self) -> int:
		return sum(1 for s in self.scores if s == 0.0)

	@property
	def summary(self) -> dict:
		return {
			"avg_score": self.avg_score,
			"no_items": self.no_items,
			"no_exact": self.no_exact,
			"no_failed": self.no_failed
		}
