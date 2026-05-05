from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.auth import get_current_admin
from backend.database import get_db
from backend.models import User as UserModel
from backend.schemas import UserOut

router = APIRouter()


@router.get("", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    _: UserModel = Depends(get_current_admin),
):
    return db.query(UserModel).order_by(UserModel.id.asc()).all()


@router.get("/{user_id}", response_model=UserOut)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: UserModel = Depends(get_current_admin),
):
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: UserModel = Depends(get_current_admin),
):
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    db.delete(user)
    db.commit()
    return {"message": "deleted"}
