import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import Base, engine
from backend.routes.auth import router as auth_router
from backend.routes.ask import router as ask_router
from backend.routes.cases import router as cases_router
from backend.routes.documents import router as docs_router
from backend.routes.users import router as users_router

app = FastAPI(title="LegalLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    os.makedirs("uploads", exist_ok=True)


@app.get("/")
def root():
    return {
        "status": "LegalLens API running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(cases_router, prefix="/cases", tags=["cases"])
app.include_router(docs_router, prefix="/documents", tags=["documents"])
app.include_router(users_router, prefix="/users", tags=["users"])
app.include_router(ask_router, prefix="/ask", tags=["ask"])
