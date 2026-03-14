from __future__ import annotations
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings
from .persist import load_model_meta, load_training_last, save_training_last
from .scanner import Scanner
from .state import AppState
from .training import run_training

app = FastAPI(title="Coinbase Crypto Prob Scanner", version="10.0.0")
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
STATE = AppState()
SETTINGS = Settings.from_env()
SCANNER = Scanner(SETTINGS, STATE)


def get_settings() -> Settings:
    return SETTINGS


@app.on_event("startup")
def _startup() -> None:
    os.makedirs(os.path.join(SETTINGS.model_dir, "pt2"), exist_ok=True)
    SCANNER.load_constituents()
    last = load_training_last(SETTINGS.model_dir)
    if last:
        with STATE.lock:
            STATE.training.running = False
            STATE.training.started_at_utc = last.get("started_at_utc")
            STATE.training.finished_at_utc = last.get("finished_at_utc")
            STATE.training.last_result = last.get("last_result")
            STATE.training.last_error = last.get("last_error")
    meta, st = load_model_meta(SETTINGS.model_dir, 2)
    if meta and st == "ok":
        with STATE.lock:
            STATE.model.pt2.trained = True
            STATE.model.pt2.path = os.path.join(SETTINGS.model_dir, "pt2")
            STATE.model.pt2.auc_val = meta.get("auc_val")
            STATE.model.pt2.brier_val = meta.get("brier_val")
            STATE.model.pt2.calibrator = "isotonic"
    SCANNER.start()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "coinbase-crypto-prob-scanner", "version": "10.0.0", "target": "pt2"}


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
def api_status(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    snap = STATE.snapshot_status()
    snap["demo_mode"] = settings.demo_mode
    snap["scan_interval_minutes"] = settings.scan_interval_minutes
    snap["model_dir"] = settings.model_dir
    snap["target"] = {"name": "pt2", "horizon_minutes": settings.target_horizon_minutes, "move_pct": settings.target_move_pct}
    return snap


@app.get("/api/scores")
def api_scores() -> Dict[str, Any]:
    return STATE.snapshot_scores()


@app.get("/api/training/status")
def training_status() -> Dict[str, Any]:
    with STATE.lock:
        return STATE.training.__dict__.copy()


@app.get("/api/debug/coverage")
def debug_coverage(password: str = Query(""), settings: Settings = Depends(get_settings)) -> JSONResponse:
    if not settings.admin_password:
        return JSONResponse(status_code=400, content={"ok": False, "error": "No ADMIN_PASSWORD configured."})
    if password != settings.admin_password:
        return JSONResponse(status_code=401, content={"ok": False, "error": "Invalid password."})
    with STATE.lock:
        items = [s.__dict__ for s in STATE.skipped[:200]]
    return JSONResponse(content={"ok": True, "count": len(items), "items": items})


def _training_thread(settings: Settings) -> None:
    try:
        if not SCANNER.constituents:
            SCANNER.load_constituents()
        symbols = [c.symbol for c in SCANNER.constituents]
        res = run_training(settings, symbols, {})
        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            STATE.training.last_result = res
            STATE.training.last_error = None
            pt2 = res.get("pt2", {})
            STATE.model.pt2.trained = True
            STATE.model.pt2.path = os.path.join(settings.model_dir, "pt2")
            STATE.model.pt2.auc_val = pt2.get("auc_val")
            STATE.model.pt2.brier_val = pt2.get("brier_val")
            STATE.model.pt2.calibrator = "isotonic"
        save_training_last(settings.model_dir, {"started_at_utc": STATE.training.started_at_utc, "finished_at_utc": STATE.training.finished_at_utc, "last_result": res, "last_error": None})
    except Exception as e:
        with STATE.lock:
            STATE.training.running = False
            STATE.training.finished_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            STATE.training.last_error = str(e)
            STATE.training.last_result = None
        save_training_last(settings.model_dir, {"started_at_utc": STATE.training.started_at_utc, "finished_at_utc": STATE.training.finished_at_utc, "last_result": None, "last_error": str(e)})


@app.post("/train")
def train(admin_password: str = Form(""), settings: Settings = Depends(get_settings)) -> JSONResponse:
    if not settings.admin_password:
        return JSONResponse(status_code=400, content={"ok": False, "error": "ADMIN_PASSWORD is not set on the server."})
    if admin_password != settings.admin_password:
        return JSONResponse(status_code=401, content={"ok": False, "error": "Invalid admin password."})
    if settings.demo_mode:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Training requires DEMO_MODE=false."})
    with STATE.lock:
        if STATE.training.running:
            return JSONResponse(status_code=409, content={"ok": False, "error": "Training is already running."})
        STATE.training.running = True
        STATE.training.started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        STATE.training.last_error = None
        STATE.training.last_result = None
    t = threading.Thread(target=_training_thread, args=(settings,), daemon=True)
    t.start()
    return JSONResponse(content={"ok": True, "message": "Training started (pt2 only)."})
