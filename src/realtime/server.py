"""
PHANTOM-ECHO REVEAL v22 — Real-Time FastAPI + WebSocket Server
==============================================================

Run:
    uvicorn src.realtime.server:app --host 0.0.0.0 --port 8000
or simply:
    python -m src.realtime.server

Endpoints:
    GET  /                      -> dashboard (src/frontend/index.html)
    WS   /ws                    -> event stream (snapshot replay on connect)
    POST /api/scan/start        -> {n_frames?, frame_delay_s?}
    POST /api/scan/stop         -> cancel a running scan gracefully
    POST /api/reveal            -> {bbox_min:[3], bbox_max:[3], semantic?}
    POST /api/mode_b            -> robot auto-reveal for RED zones
    POST /api/photo             -> upload image for monocular depth scan
    GET  /api/state             -> engine state + live tag counts
    GET  /api/kpis              -> KPI table (eval_results.json + Atlas baseline)
    GET  /api/scene/export      -> download scene_mesh.ply
    GET  /health                -> server health + uptime
    POST /api/reset             -> clear scene, reset engine to idle
    GET  /api/sonar_demo        -> serve sonar_reveal.html demo
    GET  /api/nav2_export       -> download Nav2 map as .zip (PGM + YAML)
    POST /api/semantic_search   -> MobileCLIP text-to-semantic search
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Optional

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.realtime.engine import RealtimeEngine, OUTPUT_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phantom.server")

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
MOBILE   = os.path.join(os.path.dirname(__file__), "..", "frontend", "mobile.html")

# MISSING-7 FIX: optional demo token for shared-WiFi demos. Set via env var:
#   export PHANTOM_DEMO_TOKEN=secrettoken
# When set, /api/reveal and /api/mode_b require X-Demo-Token header.
DEMO_TOKEN = os.environ.get("PHANTOM_DEMO_TOKEN", "")
_server_start_time = time.time()

app = FastAPI(title="PHANTOM-ECHO REVEAL", version="22.0")


def _check_token(request: Request) -> Optional[JSONResponse]:
    """MISSING-7 FIX: validate demo token when PHANTOM_DEMO_TOKEN is set."""
    if DEMO_TOKEN and request.headers.get("X-Demo-Token") != DEMO_TOKEN:
        return JSONResponse(
            {"error": "Invalid or missing X-Demo-Token header"},
            status_code=403)
    return None


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


_qr_event: Optional[dict] = None   # cached QR event, built once, replayed on connect


def _build_qr_event() -> Optional[dict]:
    """Build (once, then cache) the QR-code event pointing at the phone capture
    page /m. Cached so it can be re-sent to EVERY WebSocket client on connect —
    otherwise a dashboard opened after startup never receives the QR.
    """
    global _qr_event
    if _qr_event is not None:
        return _qr_event
    import socket, base64, io
    try:
        import qrcode
        # Resolve LAN IP so the QR works from phones on the same WiFi.
        # Priority: LAN_IP env var → best Wi-Fi interface → UDP route trick → localhost
        lan_ip = os.environ.get("LAN_IP", "").strip()
        if not lan_ip:
            # Walk all interfaces and prefer a real Wi-Fi/Ethernet subnet over
            # VPN tunnels (which typically sit on 172.16.0.x/10.x virtual adapters).
            try:
                import socket as _s
                all_ips = []
                for info in _s.getaddrinfo(_s.gethostname(), None, _s.AF_INET):
                    ip = info[4][0]
                    if ip.startswith("127."):
                        continue
                    all_ips.append(ip)
                # Score: prefer real Wi-Fi subnets over VirtualBox/VMware/VPN
                # Known virtual adapter ranges to deprioritize:
                #   192.168.56.x  — VirtualBox host-only
                #   192.168.99.x  — Docker/VirtualBox
                #   192.168.122.x — libvirt/KVM
                VIRTUAL_SUBNETS = {(192, 168, 56), (192, 168, 99), (192, 168, 122)}
                def _ip_score(ip):
                    parts = list(map(int, ip.split(".")))
                    subnet3 = tuple(parts[:3])
                    if subnet3 in VIRTUAL_SUBNETS:
                        return (8, 0)          # VirtualBox/Docker — skip
                    if parts[0] == 192 and parts[1] == 168:
                        return (0, -parts[3])  # real home/office WiFi (higher DHCP = better)
                    if parts[0] == 172:
                        return (1, -parts[3])  # 172.x — real WiFi DHCP
                    if parts[0] == 10:
                        return (2, -parts[3])  # corporate WiFi
                    return (9, 0)              # unknown — avoid
                if all_ips:
                    lan_ip = sorted(all_ips, key=_ip_score)[0]
            except Exception:
                pass
        if not lan_ip:
            # Final fallback: UDP route trick (may pick VPN on some setups)
            s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s2.connect(("8.8.8.8", 80))
                lan_ip = s2.getsockname()[0]
            except Exception:
                lan_ip = "localhost"
            finally:
                s2.close()
        port = int(os.environ.get("PORT", "8000"))
        # QR points at the phone capture page (/m), not the heavy 3D dashboard.
        url = f"http://{lan_ip}:{port}/m"
        img = qrcode.make(url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        _qr_event = {"type": "qr", "data_url": f"data:image/png;base64,{b64}",
                     "url": url}
        logger.info(f"QR code ready for {url}")
    except ImportError:
        logger.info("qrcode library not installed — QR panel skipped. "
                    "Install with: pip install qrcode[pil]")
    except Exception as e:
        logger.warning(f"QR code generation failed: {e}")
    return _qr_event


async def _emit_qr():
    """Build the QR at startup and broadcast to any already-connected clients."""
    await asyncio.sleep(0.5)
    ev = _build_qr_event()
    if ev:
        hub.broadcast_threadsafe(ev)


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(FRONTEND, media_type="text/html")


@app.get("/m")
async def mobile():
    """Lightweight phone capture page (the QR code points here).

    A judge scans the QR with their phone, opens this page, taps 'Scan this
    room' (native camera, works over plain HTTP on the LAN), and the photo is
    reconstructed live on the big-screen dashboard.
    """
    return FileResponse(MOBILE, media_type="text/html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        # replay snapshot so late joiners see the scene built so far
        for ev in engine.snapshot():
            await ws.send_text(json.dumps(ev))
        # always (re)send the QR so the panel appears no matter when the
        # dashboard was opened (the QR is broadcast once at startup otherwise).
        qr = _build_qr_event()
        if qr:
            await ws.send_text(json.dumps(qr))
        # NEW-BUG-11 FIX: enforce a server-side read timeout.
        # The frontend sends 'ping' every 15s. If we don't hear from the client
        # for 120s, assume it froze or crashed and actively close the socket to
        # prevent stale clients bogging down the broadcast loop.
        #
        # BUG-PROD-4 FIX: raised from 30s → 90s. A real-data scan takes ~51s
        # (measured in real_data_eval.json). The old 30s timeout fired mid-scan,
        # silently dropping the WebSocket and blanking the judge's dashboard.
        # FIX-2 (analysis report): further raised 90s → 120s. When a browser tab
        # goes to background on mobile/tablet, JS setInterval is throttled from
        # 15s to ~60s cadence — 90s timeout could fire after just 2 missed pings
        # (120s = 2× the throttled interval, giving a 60s safety margin).
        while True:
            await asyncio.wait_for(ws.receive_text(), timeout=120.0)
    except asyncio.TimeoutError:
        logger.warning(f"Client {ws.client} timed out (no ping for 120s)")
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
    # BUG-PROD-4 FIX: validate file size before writing to disk.
    # A 50MB HEIC from an iPhone will OOM the demo server without this guard.
    MAX_BYTES = 20 * 1024 * 1024   # 20MB
    data = await file.read()
    if len(data) > MAX_BYTES:
        return JSONResponse(
            {"error": f"File too large ({len(data)//1024}KB). Maximum 20MB."},
            status_code=413)
    os.makedirs("uploads", exist_ok=True)
    dest = os.path.join("uploads", "photo_" + os.path.basename(file.filename or "img.jpg"))
    with open(dest, "wb") as f:
        f.write(data)
    try:
        result = await asyncio.to_thread(engine.photo_scan, dest)
        return result
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/photo_sweep")
async def photo_sweep(files: List[UploadFile] = File(...)):
    """Tier-2 multi-shot sweep: upload several photos -> reconstruct each ->
    TSDF-fuse into one denoised cloud streamed live to the dashboard."""
    if not files:
        return JSONResponse({"error": "no files uploaded"}, status_code=400)
    os.makedirs("uploads", exist_ok=True)
    stamp = int(time.time())
    paths = []
    for i, f in enumerate(files):
        dest = os.path.join(
            "uploads", f"sweep_{stamp}_{i}_" + os.path.basename(f.filename or f"img{i}.jpg"))
        with open(dest, "wb") as fh:
            fh.write(await f.read())
        paths.append(dest)
    try:
        result = await asyncio.to_thread(engine.photo_sweep, paths)
        return result
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/reveal")
async def reveal(req: RevealRequest, request: Request):
    # MISSING-7 FIX: check demo token when configured
    auth_err = _check_token(request)
    if auth_err:
        return auth_err
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


@app.post("/api/agent")
async def run_agent_endpoint():
    """Run the Prove→Measure→Imagine tool-using agent and stream its reasoning.

    Each agent step is broadcast as a `log` event so it appears live in the
    dashboard's log panel; the final per-region tags + tool-use counts are
    returned as JSON. Deterministic offline by default; set
    PHANTOM_AGENT_LLM=claude (+ ANTHROPIC_API_KEY) for the Claude planner.
    """
    from src.agent import run_agent

    def _emit(ev: dict):
        t = ev.get("type")
        if t == "agent_step":
            hub.broadcast_threadsafe({"type": "log", "msg": (
                f"[agent:{ev.get('planner')}] {ev.get('region_id')} "
                f"-> {ev.get('tool')}: {ev.get('reasoning')}")})
        elif t == "agent_summary":
            hub.broadcast_threadsafe({"type": "log", "msg": (
                f"[agent] done - {ev.get('regions')} in {ev.get('elapsed_s')}s")})

    result = await asyncio.to_thread(run_agent, None, _emit)
    return result.to_dict()


@app.get("/api/state")
async def state():
    return {"state": engine.state, "counts": engine.counts,
            "floor_y": engine.floor_y, "ceiling_y": engine.ceiling_y,
            "total": len(engine.all_gaussians)}


@app.post("/api/scan/stop")
async def scan_stop():
    """MISSING-3 FIX: gracefully cancel a running scan.

    Sets a stop flag that the frame loop reads after each sleep().
    The scan thread will finish the current frame and then exit cleanly,
    emitting a final summary event before setting state='complete'.
    """
    stopped = engine.stop_scan()
    return {"stopped": stopped, "state": engine.state}


class ModeBRequest(BaseModel):
    """Mode B: robot has reached a RED zone and requests auto-generation."""
    robot_position: List[float]          # [x, y, z] current robot world pos
    radius_m: float = 0.8               # search radius around robot pos
    request_id: str = "mode_b"


@app.post("/api/mode_b")
async def mode_b(req: ModeBRequest, request: Request):
    """Mode B — autonomous robot trigger: when the robot's Nav2 path enters a
    RED zone it cannot navigate through, it POSTs here to request on-the-fly
    generation.  The engine finds all RED Gaussians within radius_m of the
    robot, clusters them, calls reveal(), and broadcasts 'mode_b' events so
    every viewer watches the gap fill in real time.

    This implements the documented Mode B flow:
        robot hits RED zone → pauses → PHANTOM reveals → robot resumes
    """
    # MISSING-7 FIX: check demo token when configured
    auth_err = _check_token(request)
    if auth_err:
        return auth_err

    if not engine.all_gaussians:
        return JSONResponse({"error": "no scene loaded"}, status_code=400)

    robot_pos = np.array(req.robot_position, dtype=np.float64)
    # BUG-4 FIX: acquire lock while iterating all_gaussians.
    # The scan thread extends all_gaussians inside self._lock; reading it
    # without the lock here can silently miss newly added RED Gaussians
    # or iterate over a partially-updated list mid-extend().
    with engine._lock:
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

    # BUG-PROD-5 FIX: engine.reveal() already emits a 'reveal_result' event
    # internally. Do NOT emit a second 'mode_b' event with the same Gaussians
    # — the frontend would receive them twice and double the n_revealed counter
    # and add duplicates to the Three.js scene.
    return {
        "revealed": len(result.get("gaussians", [])),
        "semantic": result.get("semantic"),
        "latency_ms": result.get("latency_ms"),
        "tier": result.get("tier"),
        "request_id": req.request_id,
    }


@app.get("/api/kpis")
async def kpis():
    """KPI table endpoint.

    ROOT CAUSE FIX (2026-06-18):
    Previously read eval_results.json (synthetic 3-scene average, mean_f1=0.858)
    and surfaced that as the primary PHANTOM metric. This is WRONG for two reasons:

    1. F1 hole-fill = 0.858 is the *synthetic* self-consistency score — measured
       against the same scene the pipeline was built from. It is not the headline KPI.

    2. reconstruction_error_cm = 0.00 in synthetic scenes because the Gaussian
       positions are self-consistent with the spec by construction. Not meaningful.

    The correct source is real_data_eval.json:
        metric.f1_5cm          = 0.957   ← PHANTOM headline KPI
        metric.recon_err_cm    = 0.98cm  ← vs Atlas 5.0cm
        metric.precision_5cm   = 0.922
        metric.recall_5cm      = 0.994

    This endpoint now surfaces real_data_eval.json as primary, with the
    synthetic 3-scene benchmark as supplementary context.
    """
    out = {
        "atlas_baseline": {
            "f1":             0.823,   # Atlas F1@5cm on same held-out RGB-D protocol
            "semantic_acc":   0.80,
            "recon_err_cm":   5.0,
        },
        "targets": {
            "f1":             0.85,    # competition threshold
            "semantic_acc":   0.93,
            "recon_err_cm":   1.5,
        },
    }

    # ── Primary: real held-out RGB-D evaluation (real_data_eval.json) ────────
    _real_paths = [
        os.path.join(OUTPUT_DIR, "real_data_eval.json"),
        "output/real_data_eval.json",
    ]
    real_data = None
    for path in _real_paths:
        try:
            with open(path) as f:
                real_data = json.load(f)
            break
        except Exception:
            continue

    if real_data is not None:
        m = real_data.get("metric", {})
        out["phantom"] = {
            # Headline numbers — quote these to judges
            "f1":                  m.get("f1_5cm",       0.0),
            "f1_10cm":             m.get("f1_10cm",      0.0),
            "precision":           m.get("precision_5cm", 0.0),
            "recall":              m.get("recall_5cm",   0.0),
            "semantic_acc":        real_data.get("mean_semantic",
                                   real_data.get("metric", {}).get("semantic_acc", 0.0)),
            "recon_err_cm":        m.get("recon_err_cm", 0.0),
            # Context
            "n_gaussians":         real_data.get("n_scene_gaussians", 0),
            "wall_time_s":         real_data.get("wall_time_s", 0),
            "protocol":            m.get("protocol", "held-out RGB-D frame"),
            "source":              "real_data_eval.json (held-out RGB-D, primary)",
            "note":                (
                f"Real-world held-out evaluation. "
                f"F1@5cm {m.get('f1_5cm', 0):.3f} vs Atlas 0.823 "
                f"(+{(m.get('f1_5cm', 0) - 0.823)*100:.1f}%). "
                f"Recon err {m.get('recon_err_cm', 0):.2f}cm vs Atlas 5.0cm."
            ),
        }
        # Attach full scene metrics separately
        fs = m.get("full_scene", {})
        if fs:
            out["phantom"]["full_scene"] = {
                "f1_5cm":        fs.get("f1_5cm", 0),
                "f1_10cm":       fs.get("f1_10cm", 0),
                "recon_err_cm":  fs.get("recon_err_cm", 0),
                "note":          "Full scene including BLUE proactive fills",
            }
    else:
        # No real_data_eval.json yet — fall back to synthetic benchmark
        out["phantom"] = {
            "f1":           0.0,
            "recon_err_cm": 0.0,
            "note":         "Run a scan in dataset mode to generate real_data_eval.json",
            "source":       "no real eval file found",
        }

    # ── Supplementary: synthetic 3-scene benchmark (eval_results.json) ───────
    _synth_paths = [
        os.path.join(OUTPUT_DIR, "eval_results.json"),
        "output/eval_results.json",
    ]
    for path in _synth_paths:
        try:
            with open(path) as f:
                synth = json.load(f)
            out["synthetic_benchmark"] = {
                "mean_f1":       synth.get("mean_f1", 0),
                "mean_semantic": synth.get("mean_semantic", 0),
                "mean_error_cm": synth.get("mean_error_cm", 0),
                "scenes":        synth.get("scenes", []),
                "all_kpis_met":  synth.get("all_kpis_met", False),
                "note": (
                    "Synthetic 3-scene self-consistency benchmark. "
                    "recon_err≈0 is expected on synthetic scenes (not a real metric). "
                    "Use real_data_eval.json numbers for judge-facing claims."
                ),
            }
            break
        except Exception:
            continue

    return out



@app.get("/api/scene/export")
async def export_scene(format: str = "ply"):
    """MISSING-4 FIX: download the reconstructed scene mesh as PLY.

    The pipeline writes scene_mesh.ply and scene_gaussians.ply to OUTPUT_DIR.
    Judges can download and open these in MeshLab / Blender to verify the
    reconstruction quality without requiring a live dashboard connection.
    """
    # Try mesh first, fall back to Gaussian PLY
    candidates = [
        (os.path.join(OUTPUT_DIR, "scene_mesh.ply"),       "phantom_scene_mesh.ply"),
        (os.path.join(OUTPUT_DIR, "scene_gaussians.ply"),  "phantom_scene_gaussians.ply"),
    ]
    for path, filename in candidates:
        if os.path.exists(path):
            return FileResponse(
                path,
                media_type="application/octet-stream",
                filename=filename)
    return JSONResponse(
        {"error": "No mesh file found — run a scan first, then check output/"},
        status_code=404)


@app.get("/health")
async def health():
    """MISSING-5 FIX: standard health-check endpoint.

    Returns engine state, Gaussian count, and server uptime. Useful for
    smoke-testing the deployment and for load-balancer health probes.
    """
    return {
        "status": "ok",
        "engine_state":  engine.state,
        "n_gaussians":   len(engine.all_gaussians),
        "uptime_s":      round(time.time() - _server_start_time, 1),
        "demo_token_set": bool(DEMO_TOKEN),
    }


@app.post("/api/reset")
async def reset_scene():
    """Clear the current scene and reset the engine to idle.

    Use this between demo runs to start fresh without restarting the server.
    Safe to call at any time — if a scan is running, it is stopped first.
    """
    if engine.state == "scanning":
        engine.stop_scan()
        await asyncio.sleep(0.5)   # let the scan thread finish its current frame
    with engine._lock:
        engine.all_gaussians.clear()
        engine.counts.clear()
        engine.events.clear()
        engine.floor_y   = 0.0
        engine.ceiling_y = 2.5
        engine.state     = "idle"
        engine._mode     = "idle"
        engine._running  = False
        engine._stop_requested = False
    # Also reset dynamic costmap cells so ghost obstacles don't persist
    from src.navigation.global_costmap import GlobalCostmap, LocalCostmap
    _gc = GlobalCostmap()
    _lc = LocalCostmap(_gc)
    _lc.reset_dynamic()
    hub.broadcast_threadsafe({"type": "reset", "msg": "Scene cleared — ready for new scan"})
    logger.info("/api/reset: scene cleared")
    return {"reset": True, "state": engine.state}


@app.get("/api/sonar_demo")
async def sonar_demo():
    """Open the Sonar Reveal animated demo (output/sonar_reveal.html).

    This is the standalone acoustic demo showing the phone walking an arc,
    emitting sound wavefronts, and painting the hidden surface TEAL as echoes
    triangulate it — with a live matched-filter scope and surface-error readout.
    Double-clickable HTML file, or served here for dashboard integration.
    """
    path = os.path.join(OUTPUT_DIR, "sonar_reveal.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return JSONResponse(
        {"error": "sonar_reveal.html not found. "
                  "Run: python -m src.realtime.export_acoustic_demo"},
        status_code=404)


@app.get("/api/nav2_export")
async def nav2_export():
    """Export the current scene costmap as a Nav2 offline map (PGM + YAML).

    Returns a ZIP containing phantom_map.pgm + phantom_map.yaml in the standard
    ROS2 map_server format. Any Nav2 deployment can load this directly:
        ros2 run nav2_map_server map_server \
            --ros-args -p map_file:=/path/to/phantom_map.yaml

    No ROS2 installation needed to generate the map — only to load it on the robot.
    """
    import zipfile, io as _io
    cm_path = os.path.join(OUTPUT_DIR, "costmap.npy")
    if not os.path.exists(cm_path):
        return JSONResponse(
            {"error": "No costmap found — run a scan first."},
            status_code=404)

    cm = np.load(cm_path)
    from src.navigation.nav2_bridge import export_for_nav2
    meta = export_for_nav2(
        costmap_2d=cm.astype(np.int8),
        resolution=0.05,
        origin_x=0.0,
        origin_z=0.0,
        output_dir=OUTPUT_DIR,
        map_name="phantom_map",
    )

    # Bundle PGM + YAML into a ZIP for single-click download
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(meta["pgm_path"],  arcname="phantom_map.pgm")
        zf.write(meta["yaml_path"], arcname="phantom_map.yaml")
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=phantom_nav2_map.zip"},
    )


class SemanticSearchRequest(BaseModel):
    query: str
    candidates: List[str]
    top_k: int = 10


@app.post("/api/semantic_search")
async def semantic_search(req: SemanticSearchRequest):
    """Section 5.1 — MobileCLIP text-to-semantic search.

    Embeds `query` and all `candidates` with MobileCLIPEmbedder, returns the
    top-K candidates ranked by cosine similarity.  This is the same embedding
    pipeline used internally by FAISSFloorPlanRetriever — exposed here so the
    dashboard can do live text queries like:

        POST /api/semantic_search
        {"query": "wooden chair", "candidates": ["CHAIR", "SOFA", "TABLE", ...]}

    Returns list of {text, score, rank} sorted by descending similarity.
    Backend: mobileclip → clip → hash (all three work, quality varies).
    """
    if not req.query.strip():
        return JSONResponse({"error": "query must be non-empty"}, status_code=422)
    if not req.candidates:
        return JSONResponse({"error": "candidates list is empty"}, status_code=422)
    if len(req.candidates) > 500:
        return JSONResponse(
            {"error": "candidates list too long (max 500)"}, status_code=422)

    try:
        results = await asyncio.to_thread(
            _run_semantic_search, req.query, req.candidates, req.top_k
        )
        return {
            "query":   req.query,
            "top_k":   req.top_k,
            "backend": results["backend"],
            "results": results["results"],
        }
    except Exception as e:
        logger.error(f"/api/semantic_search failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def _run_semantic_search(query: str,
                          candidates: List[str],
                          top_k: int) -> dict:
    """Thread-safe wrapper (called via asyncio.to_thread)."""
    from src.edge.embedding.mobile_clip import MobileCLIPEmbedder
    emb = MobileCLIPEmbedder()
    results = emb.semantic_search(query, candidates, top_k=top_k)
    return {"backend": emb.backend, "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.realtime.server:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")), log_level="info")
