"""
WaveMaker Docs Agent - Main FastAPI Application
"""

import os

# Force PyTorch backend - disable MLX on Apple Silicon (causes NaN scores)
os.environ["SENTENCE_TRANSFORMERS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow online but force torch

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    settings = get_settings()
    logger.info("Starting WaveMaker Docs Agent...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"AI Provider: {settings.ai_provider}")
    
    # Log the model for the selected provider
    if settings.ai_provider == "anthropic":
        logger.info(f"Model: {settings.anthropic_model}")
    elif settings.ai_provider == "openai":
        logger.info(f"Model: {settings.openai_model}")
    elif settings.ai_provider == "ollama":
        logger.info(f"Model: {settings.ollama_model} @ {settings.ollama_base_url}")
    
    logger.info(f"Qdrant Collection: {settings.qdrant_collection_name}")

    # Start analytics background worker
    if settings.analytics_enabled:
        from src.analytics.worker import start_worker
        await start_worker()
        logger.info("Analytics enabled")

    # Startup: Initialize connections (done lazily in modules)
    yield

    # Shutdown: Cleanup
    logger.info("Shutting down WaveMaker Docs Agent...")
    
    # Stop analytics worker
    if settings.analytics_enabled:
        from src.analytics.worker import stop_worker
        await stop_worker()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="WaveMaker Docs Agent",
        description="AI-powered documentation assistant for WaveMaker",
        version="1.0.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api")
    
    # Include analytics routes
    if settings.analytics_enabled:
        from src.api.analytics import router as analytics_router
        app.include_router(analytics_router)
    
    # Mount static files for dashboard
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import RedirectResponse, FileResponse
    import os
    
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        @app.get("/analytics")
        async def analytics_dashboard():
            """Redirect to analytics dashboard."""
            return FileResponse(os.path.join(static_dir, "analytics.html"))

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
