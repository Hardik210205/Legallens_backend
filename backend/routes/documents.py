import os
import time

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.database import get_db
from backend.models import Case, Document, User
from backend.schemas import DocumentOut

router = APIRouter()

UPLOAD_ROOT = "uploads"


@router.get("", response_model=list[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(Document)
        .filter(Document.uploaded_by == current_user.id)
        .order_by(Document.id.asc())
        .all()
    )


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    case_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if case_id is not None:
        case = db.query(Case).filter(Case.id == case_id).first()
        if case is None or case.user_id != current_user.id:
            raise HTTPException(
                status_code=404,
                detail="Case not found",
            )

    timestamp = int(time.time() * 1000)
    safe_name = os.path.basename(file.filename or "upload")
    rel_path = os.path.join(
        UPLOAD_ROOT, f"{timestamp}_{safe_name}"
    )
    os.makedirs(UPLOAD_ROOT, exist_ok=True)

    async with aiofiles.open(rel_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await out.write(chunk)

    doc = Document(
        filename=safe_name,
        filepath=rel_path,
        case_id=case_id,
        uploaded_by=current_user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None or doc.uploaded_by != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None or doc.uploaded_by != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    path = doc.filepath
    db.delete(doc)
    db.commit()
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return {"message": "deleted"}
