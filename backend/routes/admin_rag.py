import os
import sys

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from fastapi import APIRouter, Depends, HTTPException, status

from backend.auth import get_current_admin

from rag import clear_index, index_documents

router = APIRouter()


@router.post("/clear-index")
def clear_and_rebuild_index(
    _: object = Depends(get_current_admin),
):
    try:
        clear_index()
        chunks_indexed = index_documents()
        return {
            "message": "Index cleared and rebuilt successfully",
            "chunks_indexed": chunks_indexed,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )