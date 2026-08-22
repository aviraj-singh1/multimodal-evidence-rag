"""Prepared real evaluation endpoint."""

from fastapi import APIRouter
from backend.app.api.query import ROOT
from backend.app.evaluation.runner import run_evaluation

router=APIRouter()
@router.post('/api/evaluate')
def evaluate()->dict[str,object]: return run_evaluation(ROOT,export=True)
