import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from backend.database import Base, engine
from backend.limiter import limiter
from backend.routes.auth import router as auth_router
from backend.routes.ask import router as ask_router
from backend.routes.admin_rag import router as admin_rag_router
from backend.routes.cases import router as cases_router
from backend.routes.documents import router as docs_router
from backend.routes.users import router as users_router

app = FastAPI(title="LegalLens API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://YOUR_VERCEL_APP_URL",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
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
    return {"status": "ok", "version": "1.0"}


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(cases_router, prefix="/cases", tags=["cases"])
app.include_router(docs_router, prefix="/documents", tags=["documents"])
app.include_router(users_router, prefix="/users", tags=["users"])
app.include_router(ask_router, prefix="/ask", tags=["ask"])
app.include_router(admin_rag_router, prefix="/admin", tags=["admin"])
