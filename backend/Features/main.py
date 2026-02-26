import asyncio
from contextlib import asynccontextmanager
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from regime.router import router as regime_router
from news.router import router as news_router
from risk.router import router as risk_router
from behavioral.router import router as behavioral_router
from gamification.router import router as gamification_router

logger = structlog.get_logger()


async def _run_leaderboard_updater():
    """Hourly leaderboard recalculation background task."""
    from core.database import AsyncSessionLocal
    from gamification.leaderboard import update_leaderboard
    while True:
        try:
            async with AsyncSessionLocal() as db:
                await update_leaderboard(db)
        except Exception as e:
            logger.error("leaderboard_update_failed", error=str(e))
        await asyncio.sleep(3600)  # every hour


async def _run_news_processor():
    """Background news stream processor."""
    from news.ingestion.processor import NewsProcessor
    processor = NewsProcessor()
    await processor.run_forever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup_begin")

    # Start background tasks
    asyncio.create_task(_run_leaderboard_updater())
    asyncio.create_task(_run_news_processor())

    # Warm NLP pipeline
    try:
        from news.nlp.pipeline import get_nlp_pipeline
        get_nlp_pipeline()
        logger.info("nlp_pipeline_warmed")
    except Exception as e:
        logger.warning("nlp_warmup_failed", error=str(e))

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="AIFIP — AI-Powered Financial Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Register all module routers
app.include_router(regime_router)
app.include_router(news_router)
app.include_router(risk_router)
app.include_router(behavioral_router)
app.include_router(gamification_router)


@app.get("/health")
async def health():
    return {"status": "ok", "modules": ["regime", "news", "risk", "behavioral", "gamification"]}
