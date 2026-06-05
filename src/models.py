class ScoringResult(tuple):
	def __new__(cls, id: int, satisfied: bool, score: float, reason: str):
		return super().__new__(cls, (satisfied, score))

	def __init__(self, id: int, satisfied: bool, score: float, reason: str):
		self.id = int(id)
		self.score = score
		self.reason = reason

	def json(self):
		return {
			"id": self.id,
			"score": self.score,
			"reason": self.reason
		}

class ModelResponsePair:
	def __init__(self, id: int, prompt: str, reference_answer: str, model_answer: str, reason: str = None):
		self.id = int(id)
		self.prompt = prompt
		self.reference_answer = reference_answer
		self.model_answer = model_answer
		self.reason = reason
		self.score = None
