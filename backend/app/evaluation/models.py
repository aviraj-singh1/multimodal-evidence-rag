"""Evaluation fixture models."""

from pydantic import BaseModel, Field


class RequiredGroup(BaseModel):
    name: str
    required_modality_family: str
    acceptable_evidence_ids: list[str] = Field(min_length=1)


class EvaluationQuestion(BaseModel):
    id: str
    question: str
    required_groups: list[RequiredGroup] = Field(min_length=1)
    notes: str


class EvaluationSuite(BaseModel):
    version: str
    questions: list[EvaluationQuestion] = Field(min_length=8)
