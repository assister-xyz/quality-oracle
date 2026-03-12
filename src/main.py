"""AgentTrust - Active competency verification for AI agents and MCP servers."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api.v1.health import router as health_router
from src.api.v1.evaluate import router as evaluate_router
from src.api.v1.scores import router as scores_router
from src.api.v1.badges import router as badges_router
from src.api.v1.attestations import router as attestations_router
from src.api.v1.enrichment import router as enrichment_router
from src.api.v1.feedback import router as feedback_router
from src.api.v1.payments import router as payments_router
from src.api.v1.battles import router as battles_router
from src.api.v1.arena import router as arena_router
from src.api.v1.aiuc1 import router as aiuc1_router
from src.api.agent_card import router as agent_card_router
from src.storage.mongodb import connect_db, close_db
from src.storage.cache import connect_redis, close_redis

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AgentTrust...")
    await connect_db()
    await connect_redis()
    logger.info(f"AgentTrust running on port {settings.port}")
    yield
    logger.info("Shutting down AgentTrust...")
    await close_db()
    await close_redis()


app = FastAPI(
    title="AgentTrust",
    description="Active competency verification for AI agents, MCP servers, and skills.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(health_router, tags=["health"])
app.include_router(evaluate_router, prefix="/v1", tags=["evaluation"])
app.include_router(scores_router, prefix="/v1", tags=["scores"])
app.include_router(badges_router, prefix="/v1", tags=["badges"])
app.include_router(attestations_router, prefix="/v1", tags=["attestations"])
app.include_router(enrichment_router, prefix="/v1", tags=["enrichment"])
app.include_router(feedback_router, prefix="/v1", tags=["feedback"])
app.include_router(payments_router, prefix="/v1", tags=["payments"])
app.include_router(battles_router, prefix="/v1", tags=["battles"])
app.include_router(arena_router, prefix="/v1", tags=["arena"])
app.include_router(aiuc1_router, prefix="/v1", tags=["aiuc1"])
app.include_router(agent_card_router, tags=["a2a"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
