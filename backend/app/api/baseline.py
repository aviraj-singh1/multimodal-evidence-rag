"""Text-only baseline query endpoint."""

from functools import lru_cache
from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from backend.app.api.query import ROOT
from backend.app.baseline.service import BaselineService

class BaselineRequest(BaseModel):
    question:str
    top_k:int=Field(default=5,ge=1,le=12)
    @field_validator('question')
    @classmethod
    def nonempty(cls,value:str)->str:
        if not value.strip(): raise ValueError('question must not be empty')
        return value.strip()

@lru_cache(maxsize=1)
def get_baseline_service()->BaselineService: return BaselineService(ROOT)

router=APIRouter()
@router.post('/api/baseline/query')
def query_baseline(request:BaselineRequest)->dict[str,object]:
    return get_baseline_service().query(request.question,request.top_k)
