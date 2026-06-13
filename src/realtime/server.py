"""
PHANTOM-ECHO REVEAL v22 — Real-Time FastAPI + WebSocket Server
==============================================================

Run:
    uvicorn src.realtime.server:app --host 0.0.0.0 --port 8000
or simply:
    python -m src.realtime.server

Endpoints:
    GET  /                  -> dashboard (src/frontend/index.html)
    WS   /ws                -> event stream (snapshot replay on connect)
    POST /api/scan/start    -> {n_frames?, frame_delay_s?}
    POST /api/reveal        -> {bbox_min:[3], bbox_max:[3], semantic?}
    GET  /api/state         -> engine state + live tag counts
    GET  /api/kpis          -> KPI table (eval_results.json + Atlas baseline)
"""

import os
import json
import asyncio
import logging
from typing import List, Optional

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.realtime.engine import RealtimeEngine, OUTPUT_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phantom.server")

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

app = FastAPI(title="PHANTOM-ECHO REVEAL", version="22.0")


# ── WebSocket hub ──────────────────────────────────────────────────────────
class Hub:
    def __init__(self):
        # BUG-2 FIX: use a set for O(1) membership checks; maintain list for
        # ordered snapshot iteration. Both structures are updated together.
        self._client_set: set = set()
        self.clients: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        if ws not in self._client_set:
            self.clients.append(ws)
            self._client_set.add(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._client_set:
            self._client_set.discard(ws)
            try:
                self.clients.remove(ws)
            except ValueError:
                pass

    def broadcast_threadsafe(self, event: dict):
        """Called from the engine's worker thread."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event: dict):
        # BUG-2 FIX: snapshot the client list BEFORE iteration so a disconnect
        # mid-broadcast does not corrupt the iterator or skip a client.
        msg = json.dumps(event)
        clients_snapshot = list(self.clients)
        dead = []
        for ws in clients_snapshot:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = Hub()
engine = RealtimeEngine(emit=hub.broadcast_threadsafe)


@app.on_event("startup")
async def _startup():
    hub.loop = asyncio.get_running_loop()
    logger.info("PHANTOM-ECHO REVEAL real-time server ready")
    # QR code: generate a base64 PNG encoding the server URL so the frontend
    # can display it in the sidebar for phone/tablet access.
    asyncio.ensure_future(_emit_qr())


async def _emit_qr():
    """Emit a QR code encoding this server's URL to all connected clients."""
    import socket, base64, io
    await asyncio.sleep(0.5)   # let clients connect first
    try:
        import qrcode
        # Resolve LAN IP so the QR works from phones on the same WiFi
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
        except Exception:
            lan_ip = "localhost"
        finally:
            s.close()
        port = int(os.environ.get("PORT", "8000"))
        url = f"http://{lan_ip}:{port}"
        img = qrcode.make(url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        ev = {"type": "qr", "data_url": f"data:image/png;base64,{b64}",
              "url": url}
        hub.broadcast_threadsafe(ev)
        logger.info(f"QR code emitted for {url}")
    except ImportError:
        logger.info("qrcode library not installed — QR panel skipped. "
                    "Install with: pip install qrcode[pil]")
    except Exception as e:
        logger.warning(f"QR code generation failed: {e}")


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(FRONTEND, media_type="text/html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        # replay snapshot so late joiners see the scene built so far
        for ev in engine.snapshot():
            await ws.send_text(json.dumps(ev))
        # NEW-BUG-11 FIX: enforce a server-side read timeout.
        # The frontend sends 'ping' every 15s. If we don't hear from the client
        # for 30s, assume it froze or crashed and actively close the socket to
        # prevent stale clients bogging down the broadcast loop.
        while True:
            await asyncio.wait_for(ws.receive_text(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning(f"Client {ws.client} timed out (no ping for 30s)")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Client disconnected: {e}")
    finally:
        hub.disconnect(ws)


class ScanRequest(BaseModel):
    n_frames: int = 8
    frame_delay_s: float = 0.6
    source: str = "synthetic"            # "synthetic" | "dataset"
    dataset_path: Optional[str] = None


class RevealRequest(BaseModel):
    bbox_min: List[float]
    bbox_max: List[float]
    semantic: Optional[str] = None
    request_id: str = "tap"


@app.post("/api/scan/start")
async def scan_start(req: ScanRequest):
    # EDGE-3 FIX: if engine is busy return a meaningful reason so the UI can
    # display "Scan already running" instead of silently doing nothing.
    if engine.state == "scanning":
        return {"started": False, "state": engine.state,
                "reason": "already_running"}

    # BUG-8 FIX: pre-flight check for real dataset path before the scan thread
    # starts. Without this, a missing dataset raises RuntimeError deep inside
    # the engine thread, emitting a raw error event and freezing the dashboard
    # permanently for the session. Now we return a clean 400 with instructions.
    if req.source == "dataset":
        import os as _os
        dpath = req.dataset_path or _os.path.join("datasets", "redwood_sample")
        if not _os.path.isdir(dpath):
            from fastapi.responses import JSONResponse as _JSONResponse
            return _JSONResponse(
                status_code=400,
                content={
                    "started": False,
                    "error": "dataset_not_found",
                    "path": dpath,
                    "fix": "Run: python scripts/get_real_dataset.py  "
                           "(downloads ~30MB Redwood RGB-D sequence)",
                })

    # EDGE-2 FIX: warn when n_frames < 3 (SAS needs ≥3 baselines for TEAL)
    # EDGE-5 FIX: enforce minimum 50ms frame delay to prevent WebSocket flood.
    n_frames = max(1, min(req.n_frames, 60))
    frame_delay_s = max(0.05, min(req.frame_delay_s, 3.0))  # min 50ms
    started = engine.start_scan(n_frames=n_frames,
                                frame_delay_s=frame_delay_s,
                                source=req.source,
                                dataset_path=req.dataset_path)
    resp = {"started": started, "state": engine.state}
    if n_frames < 3:
        resp["warning"] = ("n_frames<3: SAS acoustic triangulation requires "
                           "≥3 baselines — TEAL Gaussians will not appear")
    return resp


@app.post("/api/photo")
async def photo(file: UploadFile = File(...)):
    """Upload a photo -> monocular depth -> 3D point cloud in the viewer."""
    os.makedirs("uploads", exist_ok=True)
    dest = os.path.join("uploads", "photo_" + os.path.basename(file.filename or "img.jpg"))
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        result = await asyncio.to_thread(engine.photo_scan, dest)
        return result
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/reveal")
async def reveal(req: RevealRequest):
    if len(req.bbox_min) != 3 or len(req.bbox_max) != 3:
        return JSONResponse({"error": "bbox_min/bbox_max must be length 3"}, status_code=400)
    # EDGE-1 FIX: validate that bbox_min < bbox_max on each axis.
    # A degenerate / inverted box produces 0 Gaussians with no error message.
    import numpy as np
    bmin = np.array(req.bbox_min); bmax = np.array(req.bbox_max)
    if np.any(bmin >= bmax):
        # Swap silently so the reveal still works instead of crashing.
        req.bbox_min = np.minimum(bmin, bmax).tolist()
        req.bbox_max = np.maximum(bmin, bmax).tolist()
        logger.warning(f"reveal: bbox_min>=bbox_max on some axis — swapped: "
                       f"{req.bbox_min} → {req.bbox_max}")
    result = await asyncio.to_thread(
        engine.reveal, req.bbox_min, req.bbox_max, req.semantic, req.request_id)
    return result


@app.get("/api/state")
async def state():
    return {"state": engine.state, "counts": engine.counts,
            "floor_y": engine.floor_y, "ceiling_y": engine.ceiling_y,
            "total": len(engine.all_gaussians)}


class ModeBRequest(BaseModel):
    """Mode B: robot has reached a RED zone and requests auto-generation."""
    robot_position: List[float]          # [x, y, z] current robot world pos
    radius_m: float = 0.8               # search radius around robot pos
    request_id: str = "mode_b"


@app.post("/api/mode_b")
async def mode_b(req: ModeBRequest):
    """Mode B — autonomous robot trigger: when the robot's Nav2 path enters a
    RED zone it cannot navigate through, it POSTs here to request on-the-fly
    generation.  The engine finds all RED Gaussians within radius_m of the
    robot, clusters them, calls reveal(), and broadcasts 'mode_b' events so
    every viewer watches the gap fill in real time.

    This implements the documented Mode B flow:
        robot hits RED zone → pauses → PHANTOM reveals → robot resumes
    """
    if not engine.all_gaussians:
        return JSONResponse({"error": "no scene loaded"}, status_code=400)

    robot_pos = np.array(req.robot_position, dtype=np.float64)
    red = [
        g for g in engine.all_gaussians
        if g.get("tag") == "RED"
        and np.linalg.norm(np.array(g["position"]) - robot_pos) <= req.radius_m
    ]
    if not red:
        return {"revealed": 0, "msg": "No RED Gaussians within radius"}

    pos = np.array([g["position"] for g in red])
    bmin = pos.min(axis=0).tolist()
    bmax = pos.max(axis=0).tolist()
    result = await asyncio.to_thread(
        engine.reveal, bmin, bmax, None, req.request_id)

    # Broadcast Mode B event so the dashboard shows the auto-trigger toast
    engine._emit({"type": "mode_b",
                  "request_id": req.request_id,
                  "semantic": result.get("semantic"),
                  "gaussians": result.get("gaussians", []),
                  "latency_ms": result.get("latency_ms")})
    return {"revealed": len(result.get("gaussians", [])),
            "semantic": result.get("semantic"),
            "latency_ms": result.get("latency_ms"),
            "tier": result.get("tier")}


@app.get("/api/kpis")
async def kpis():
    out = {"atlas_baseline": {"f1": 0.85, "semantic_acc": 0.80, "recon_err_cm": 5.0},
           "targets": {"f1": 0.97, "semantic_acc": 0.93, "recon_err_cm": 1.5}}
    for path in (os.path.join(OUTPUT_DIR, "eval_results.json"), "output/eval_results.json"):
        try:
            with open(path) as f:
                out["phantom"] = json.load(f)
            break
        except Exception:
            continue
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.realtime.server:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")), log_level="info")
