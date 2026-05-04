import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

import re
import json
from rag import chain_with_memory
from fastapi import APIRouter, Depends, HTTPException
from backend.auth import get_current_user
from backend.models import User
from backend.schemas import AskRequest, AskResponse

router = APIRouter()

def parse_answer(raw: str) -> str:
    """Extract clean answer from raw Mistral output."""
    # Remove 'Output:' prefix if present
    raw = raw.strip()
    if raw.startswith("Output:"):
        raw = raw[len("Output:"):].strip()

    # Try to parse as JSON and extract 'answer' field
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]
    except Exception:
        pass

    # Try regex for {"answer": "..."} pattern
    match = re.search(r'"answer"\s*:\s*"(.*?)"(?:,|\})', raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to extract after Question:/Output: blocks (chat history leaking)
    lines = raw.split("\n")
    clean_lines = []
    skip = False
    for line in lines:
        if line.strip().startswith("Question:") or line.strip().startswith("Output:"):
            skip = True
            continue
        if skip and line.strip() == "":
            skip = False
            continue
        if not skip:
            clean_lines.append(line)
    cleaned = "\n".join(clean_lines).strip()
    if cleaned:
        return cleaned

    return raw


@router.post("/", response_model=AskResponse)
def ask(
    request: AskRequest,
    _: User = Depends(get_current_user),
):
    try:
        session_id = f"case-{request.case_id}"
        response = chain_with_memory.invoke(
            {"question": request.question},
            config={"configurable": {"session_id": session_id}}
        )
        answer = parse_answer(str(response))
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))