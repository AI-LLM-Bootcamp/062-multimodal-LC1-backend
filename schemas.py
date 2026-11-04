from pydantic import BaseModel
from typing import Optional

class PDFRequest(BaseModel):
    name: str
    selected: bool
    file: str

class PDFUpdate(BaseModel):
    # Used to update a PDF: every field is optional, so the frontend can send
    # only the field that changed, for example {"selected": true}
    name: Optional[str] = None
    selected: Optional[bool] = None
    file: Optional[str] = None

class PDFResponse(BaseModel):
    id: int
    name: str
    selected: bool
    file: str

    class Config:
        from_attributes = True


class QuestionRequest(BaseModel):
    question: str
