import sys
import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Add current directory to path so relative-style imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ModelResponsePair
from scorer import Score
from rules import (
	CatchMissingRule,
	ExactMatchRule,
	LevenshteinSimilarityRule,
	NormalizeTextPreprocessor
)

app = FastAPI(title="Score CLI Tool API")

class EvaluationItem(BaseModel):
	id: int
	prompt: str
	reference_answer: str
	model_answer: str
	reason: Optional[str] = None

class ScoringResultModel(BaseModel):
	id: int
	score: float
	reason: str

class ScoringSummaryModel(BaseModel):
	avg_score: float
	no_items: int
	no_exact: int
	no_failed: int

class ScoreResponse(BaseModel):
	results: List[ScoringResultModel]
	summary: ScoringSummaryModel

@app.post("/score", response_model=ScoreResponse)
def score_endpoint(items: List[EvaluationItem]):
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
		ModelResponsePair(
			id=item.id,
			prompt=item.prompt,
			reference_answer=item.reference_answer,
			model_answer=item.model_answer,
			reason=item.reason
		) for item in items
	]
	
	for pair in pairs:
		pair.score = score_runner.score(pair)
		
	return {
		"results": [pair.score.json() for pair in pairs],
		"summary": score_runner.summary
	}
