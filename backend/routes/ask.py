import os
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from rag import chain_with_memory
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import AskSession, Case, Message, User
from backend.schemas import AskRequest, AskResponse, MessageOut

router = APIRouter()

@router.post("/", response_model=AskResponse)
def ask(
    request: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        case = db.query(Case).filter(Case.id == request.case_id).first()
        if case is None or case.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Case not found")

        session = (
            db.query(AskSession)
            .filter(
                AskSession.case_id == request.case_id,
                AskSession.user_id == current_user.id,
            )
            .first()
        )
        if session is None:
            session = AskSession(case_id=request.case_id, user_id=current_user.id)
            db.add(session)
            db.commit()
            db.refresh(session)

        user_msg = Message(
            session_id=session.id,
            role="user",
            content=request.question,
        )
        db.add(user_msg)
        db.commit()

        answer = chain_with_memory.invoke(
            {"question": request.question},
            config={"configurable": {"session_id": str(session.id)}}
        )

        assistant_msg = Message(
            session_id=session.id,
            role="assistant",
            content=answer,
        )
        db.add(assistant_msg)
        db.commit()

        return AskResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{case_id}", response_model=list[MessageOut])
def get_history(
    case_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = (
        db.query(AskSession)
        .filter(AskSession.case_id == case_id, AskSession.user_id == current_user.id)
        .first()
    )
    if session is None:
        return []

    return (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.timestamp.asc())
        .all()
    )