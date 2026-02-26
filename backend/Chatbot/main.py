"""
FinAI Platform - Production FastAPI Backend
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers import chat, health
from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finai.log")
    ]
)
logger = logging.getLogger("finai")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"FinAI Platform starting | Model: {settings.OLLAMA_MODEL}")
    yield
    logger.info("FinAI Platform shutting down")

app = FastAPI(
    title="FinAI Intelligence Platform",
    version="1.0.0",
    description="AI-powered Financial Intelligence Chatbot",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{time.time()-start:.3f}s]")
    return response

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.get("/")
async def root():
    return {"status": "online", "platform": "FinAI Intelligence", "version": "1.0.0"}