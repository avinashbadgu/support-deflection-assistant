"""FastAPI app entrypoint. Mounts routes, templates, and static assets."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import routes

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_app() -> FastAPI:
    app = FastAPI(title="Support-Deflection Assistant", version="0.1.0")
    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(routes.router)

    @app.exception_handler(Exception)
    async def _on_exc(request: Request, exc: Exception) -> JSONResponse:
        # Print the real traceback so we can see what 500'd in the log.
        print(f"\n!!! 500 on {request.method} {request.url.path}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    return app


app = create_app()
