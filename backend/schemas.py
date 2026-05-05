from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    user_id: Optional[int] = None


class UserLogin(BaseModel):
    email: str
    password: str


class CaseCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: str = "open"


class CaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class CaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: Optional[str] = None
    status: str
    user_id: int
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    case_id: Optional[int] = None
    uploaded_by: int
    created_at: datetime


class AskRequest(BaseModel):
    question: str
    case_id: int

class AskResponse(BaseModel):
    answer: str
