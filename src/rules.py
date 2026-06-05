from abc import ABC, abstractmethod
import Levenshtein
from models import ModelResponsePair, ScoringResult

class ScoringRule(ABC):

	@property
	@abstractmethod
	def name(self) -> str:
		pass

	@abstractmethod
	def score(self, pair: ModelResponsePair) -> tuple[bool, float]:
		pass

class CatchMissingRule(ScoringRule):

	@property
	def name(self) -> str:
		return "Catch Missing"

	def score(self, pair: ModelResponsePair) -> tuple[bool, float]:
		if len(pair.reference_answer) == 0:
			return ScoringResult(pair.id, True, 0.0, "Reference answer is missing")
		
		if len(pair.model_answer) == 0:
			return ScoringResult(pair.id, True, 0.0, "Model answer is missing")

		return ScoringResult(pair.id, False, 0.0, "")

class ExactMatchRule(ScoringRule):

	@property
	def name(self) -> str:
		return "Exact Match"

	def score(self, pair: ModelResponsePair) -> tuple[bool, float]:
		if pair.reference_answer == pair.model_answer:
			return ScoringResult(pair.id, True, 1.0, "Exact match")
		return ScoringResult(pair.id, False, 0.0, "Not an exact match")

class LevenshteinSimilarityRule(ScoringRule):

	@property
	def name(self) -> str:
		return "Levenshtein Similarity"

	def score(self, pair: ModelResponsePair) -> tuple[bool, float]:
		ref = pair.reference_answer
		model = pair.model_answer
		max_len = max(len(ref), len(model))
		if max_len == 0:
			return ScoringResult(pair.id, True, 1.0, "Exact match (empty strings)")
		distance = Levenshtein.distance(ref, model)
		score = 1.0 - (distance / max_len)
		if score != 0:
			return ScoringResult(pair.id, True, score, "Partial match")
		return ScoringResult(pair.id, False, 0.0, "not a partial match")

class Preprocessor(ABC):

	@abstractmethod
	def preprocess(self, pair: ModelResponsePair) -> None:
		pass

class NormalizeTextPreprocessor(Preprocessor):

	def preprocess(self, pair: ModelResponsePair) -> None:
		pair.reference_answer = pair.reference_answer.strip().lower()
		pair.model_answer = pair.model_answer.strip().lower()
