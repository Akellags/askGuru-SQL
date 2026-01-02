from pydantic import BaseModel, Field
from typing import Optional, List, Any

class SQLRequest(BaseModel):
    question: str = Field(..., example="What are the total invoice amounts for each supplier?")
    tables: Optional[List[str]] = Field(None, description="Optional list of relevant tables to narrow down the schema")

class SQLResponse(BaseModel):
    sql: str
    model_used: str
    confidence_score: Optional[float] = None
    validation_passed: bool
    error: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
