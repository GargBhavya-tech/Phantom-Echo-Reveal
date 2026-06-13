"""
PHANTOM-ECHO REVEAL — Cloud API Server (All Bugs Fixed)

Fixes vs original:
  BUG-4  : SessionStore memory leak → background thread cleanup
  BUG-4b : _np undefined variable → import numpy as np at module level
  BUG-7  : MobileCLIP first, CLIP only as fallback
  BUG-9  : ISM residual actually subtracted before SAS
  SVQ    : /reveal compresses Gaussians before returning
  LLM    : LLaVA builds VideoScene prompt (not bare string)
  ENV    : simulate controlled by PHANTOM_SIMULATE env var
"""

import time, threading, logging, os, json, uuid, numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SIMULATE      = os.environ.get("PHANTOM_SIMULATE", "true").lower() == "true"
# BUG-V21-1 FIX: constant was referenced on lines 225+287 but never defined.
MAX_SESSION_GAUSSIANS = int(os.environ.get("PHANTOM_MAX_GAUSSIANS", "20000"))
SESSION_TTL_S = 1800   # 30 minutes


# ── Session store (BUG-4 fix: background cleanup) ─────────────────────────
@dataclass
class Session:
    session_id:   str
    created_at:   float = field(default_factory=time.time)
    last_active:  float = field(default_factory=time.time)
    gaussians:    List[Dict] = field(default_factory=list)
    floor_y:      float = 0.0
    ceiling_y:    float = 2.5
    scan_count:   int   = 0
    acoustic_map: Dict[str, float] = field(default_factory=dict)


class SessionStore:
    def __init__(self, ttl_s: float = SESSION_TTL_S):
        self._sessions: Dict[str, Session] = {}
        self._lock  = threading.Lock()
        self._ttl   = ttl_s
        t = threading.Thread(target=self._cleanup_loop, daemon=True)
        t.start()

    def _cleanup_loop(self):
        while True:
            time.sleep(60)
            self.clear_old_sessions()

    def clear_old_sessions(self) -> int:
        now = time.time()
        with self._lock:
            stale = [sid for sid, s in self._sessions.items()
                     if now - s.last_active > self._ttl]
            for sid in stale:
                del self._sessions[sid]
        if stale:
            logger.info(f"SessionStore: evicted {len(stale)} stale sessions")
        return len(stale)

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
            s = self._sessions[session_id]
            s.last_active = time.time()
            return s

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.last_active = time.time()
            return s

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# ── MobileCLIP with CLIP fallback (BUG-7 fix) ────────────────────────────
class SemanticClassifier:
    LABELS = ["sofa","chair","table","bookshelf","monitor","plant","lamp",
              "door","window","wall","floor","ceiling","painting","unknown"]

    def __init__(self):
        self._model     = None
        self._processor = None
        self._backend   = "none"
        self._try_mobileclip()
        if self._backend == "none":
            self._try_clip()

    def _try_mobileclip(self):
        try:
            import mobileclip
            self._model, _, self._processor = mobileclip.create_model_and_transforms(
                "mobileclip_s0", pretrained=True)
            self._model.eval()
            self._backend = "mobileclip"
            logger.info("MobileCLIP loaded")
        except Exception as e:
            logger.info(f"MobileCLIP unavailable ({type(e).__name__}), trying CLIP")

    def _try_clip(self):
        try:
            from transformers import CLIPProcessor, CLIPModel
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._model.eval()
            self._backend = "clip"
            logger.warning("Using CLIP (~100ms) — MobileCLIP preferred")
        except Exception as e:
            logger.warning(f"CLIP unavailable ({e}) — semantic=UNKNOWN")

    def classify(self, rgb_crop: Any) -> str:
        if self._backend == "none":
            return "UNKNOWN"
        try:
            import torch
            texts = [f"a photo of a {l}" for l in self.LABELS]
            if self._backend == "mobileclip":
                import mobileclip
                tok  = mobileclip.tokenize(texts)
                with torch.no_grad():
                    img_feat = self._model.encode_image(rgb_crop)
                    txt_feat = self._model.encode_text(tok)
                logits = (img_feat @ txt_feat.T).squeeze()
            else:
                inp = self._processor(text=texts, images=rgb_crop,
                                      return_tensors="pt", padding=True)
                with torch.no_grad():
                    out = self._model(**inp)
                logits = out.logits_per_image.squeeze()
            return self.LABELS[int(logits.argmax())].upper()
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "UNKNOWN"


# ── ISM subtraction helper (BUG-9 fix) ───────────────────────────────────
def _subtract_ism(raw_echo: np.ndarray, phone_pos: np.ndarray,
                   room_bounds: Dict[str, float]) -> np.ndarray:
    try:
        from src.edge.sensing.ism_filter import (
            subtract_visible_echoes, build_first_order_rir, WallPlane)
        walls = [
            WallPlane(1,0,0,0,label="x0"),
            WallPlane(-1,0,0,-room_bounds["x"],label="xmax"),
            WallPlane(0,1,0,0,label="floor"),
            WallPlane(0,-1,0,-room_bounds["y"],label="ceiling"),
            WallPlane(0,0,1,0,label="z0"),
            WallPlane(0,0,-1,-room_bounds["z"],label="zmax"),
        ]
        rir = build_first_order_rir(phone_pos, phone_pos, walls,
                                    44100, len(raw_echo))
        return subtract_visible_echoes(raw_echo, rir)
    except Exception as e:
        logger.warning(f"ISM subtraction failed ({e}), using raw echo")
        return raw_echo


# ── Server ────────────────────────────────────────────────────────────────
class PhantomServer:
    def __init__(self):
        self._store      = SessionStore()
        self._classifier = SemanticClassifier()

        from src.edge.reconstruction.ddgs_gaussrender import DDGSGaussRender  # FIX-10: class now exists
        from src.cloud.generation.videoscene_pipeline_fixed import generate_gaussians_for_region
        from src.cloud.compression.svq_endpoint import compress_reveal_response
        from src.cloud.llm.llava_wrapper import LLaVASceneDescriber

        self._ddgs     = DDGSGaussRender()
        self._generate = generate_gaussians_for_region
        self._compress = compress_reveal_response
        self._llava    = LLaVASceneDescriber()
        logger.info(f"PhantomServer ready (simulate={SIMULATE})")

    # /scan
    def handle_scan(self, body: Dict) -> Dict:
        t0 = time.time()
        session_id = body.get("session_id", str(uuid.uuid4()))
        sess = self._store.get_or_create(session_id)

        H = body.get("depth_h", 192)
        W = body.get("depth_w", 256)
        depth_flat = body.get("depth_flat", [])
        conf_flat  = body.get("confidence_flat", [])
        rgb_flat   = body.get("rgb_flat", [])

        if depth_flat:
            depth_map = np.array(depth_flat, dtype=np.float32).reshape(H, W)
            conf_map  = (np.array(conf_flat, dtype=np.uint8).reshape(H, W)
                         if conf_flat else np.ones((H, W), dtype=np.uint8))
            rgb_image = (np.array(rgb_flat, dtype=np.uint8).reshape(H, W, 3)
                         if rgb_flat else np.zeros((H, W, 3), dtype=np.uint8))
        else:
            from src.edge.sensing.arkit_depth import SyntheticDepthGenerator
            gen   = SyntheticDepthGenerator({"x": 5.0, "y": 2.5, "z": 4.0})
            frame = gen.generate_frame(np.array([0.5, 1.2, 0.5]))
            depth_map = frame.depth_map
            conf_map  = frame.confidence_map
            rgb_image = frame.rgb_image

        cam_to_world_flat = body.get("cam_to_world")
        cam_to_world = (np.array(cam_to_world_flat, dtype=np.float64).reshape(4, 4)
                        if cam_to_world_flat else np.eye(4))
        intrinsics = body.get("camera_intrinsics",
                               {"fx": 500, "fy": 500, "cx": W/2, "cy": H/2})

        # BUG-4b fix: np is imported at module level — no more _np undefined
        audio_flat = body.get("audio_flat", [])
        if audio_flat:
            raw_echo  = np.array(audio_flat, dtype=np.float32)   # was _np — FIXED
            phone_pos = np.array(body.get("phone_position", [0.5, 1.2, 0.5]))
            residual  = _subtract_ism(raw_echo, phone_pos, {"x":5,"y":2.5,"z":4})
            sess.acoustic_map[f"frame_{sess.scan_count}"] = float(
                np.mean(np.abs(residual)))

        gaussians = self._ddgs.process_depth_frame(
            depth_map, conf_map, rgb_image, cam_to_world, intrinsics)
        # BUG-V19-3 FIX: cap storage so RAM never grows unbounded
        remaining = max(0, MAX_SESSION_GAUSSIANS - len(sess.gaussians))
        if remaining > 0:
            sess.gaussians.extend(gaussians[:remaining])
        sess.scan_count += 1

        if gaussians:
            ys = np.array([g["position"][1] for g in gaussians], dtype=np.float64)
            # BUG-V18-9 FIX: only update floor/ceiling from frames where depth
            # covers a wide vertical range (camera not pointing straight up/down).
            # Filter: only use frames where y_range > 30cm to avoid ceiling-scan
            # or floor-scan poisoning the floor_y estimate.
            y_range = float(np.percentile(ys, 95) - np.percentile(ys, 5))
            if y_range > 0.30:
                # Exponential moving average so later frames can correct early errors
                alpha = 0.3   # 30% weight to new observation, 70% to history
                new_floor   = float(np.percentile(ys, 3))    # 3rd percentile = near floor
                new_ceiling = float(np.percentile(ys, 97))   # 97th percentile = near ceiling
                sess.floor_y   = alpha * new_floor   + (1 - alpha) * sess.floor_y
                sess.ceiling_y = alpha * new_ceiling + (1 - alpha) * sess.ceiling_y
            # Hard bounds: floor can never be above 0.5m, ceiling never below 1.5m
            sess.floor_y   = min(sess.floor_y,   0.5)
            sess.ceiling_y = max(sess.ceiling_y, 1.5)

        return {
            "session_id":      session_id,
            "frame_gaussians": len(gaussians),
            "total_gaussians": len(sess.gaussians),
            "scan_count":      sess.scan_count,
            "processing_ms":   round((time.time() - t0) * 1000, 1),
        }

    # /reveal
    def handle_reveal(self, body: Dict) -> bytes:
        t0 = time.time()
        session_id = body.get("session_id", "")
        sess       = self._store.get(session_id)
        floor_y    = sess.floor_y   if sess else 0.0
        ceiling_y  = sess.ceiling_y if sess else 2.5

        semantic = body.get("semantic", "UNKNOWN")
        bbox_min = np.array(body.get("bbox_min", [0,0,0]), dtype=np.float32)
        bbox_max = np.array(body.get("bbox_max", [1,1,1]), dtype=np.float32)

        scene_desc = "An indoor scene with typical furniture."
        if sess and sess.gaussians:
            try:
                scene_desc = self._llava.describe_scene(
                    np.zeros((192,256,3), dtype=np.uint8), [semantic])
            except Exception:
                pass

        prompt = self._llava.build_videoscene_prompt(
            scene_desc, semantic, bbox_min.tolist(), bbox_max.tolist(),
            acoustic_distance_m=body.get("acoustic_distance_m"))

        gaussians, tier = self._generate(
            semantic=semantic, bbox_min=bbox_min, bbox_max=bbox_max,
            floor_y=floor_y, ceiling_y=ceiling_y,
            prompt=prompt, simulate=SIMULATE)

        if sess:
            # BUG-V19-3 FIX: same cap applied on reveal path
            remaining = max(0, MAX_SESSION_GAUSSIANS - len(sess.gaussians))
            if remaining > 0:
                sess.gaussians.extend(gaussians[:remaining])

        compressed = self._compress(gaussians)
        logger.info(f"Reveal: {semantic}, {len(gaussians)} splats, "
                    f"tier={tier}, {len(compressed)/1024:.1f}KB, "
                    f"{(time.time()-t0)*1000:.0f}ms")
        return compressed

    # /scene
    def handle_scene(self, body: Dict) -> Dict:
        session_id = body.get("session_id", "")
        sess = self._store.get(session_id)
        if not sess:
            return {"error": "session not found", "gaussians": []}
        return {
            "session_id":     session_id,
            "gaussian_count": len(sess.gaussians),
            "floor_y":        sess.floor_y,
            "ceiling_y":      sess.ceiling_y,
            "scan_count":     sess.scan_count,
            "gaussians":      sess.gaussians[:5000],
        }

    # /status
    def handle_status(self) -> Dict:
        return {
            "status":     "running",
            "simulate":   SIMULATE,
            "sessions":   self._store.count(),
            "classifier": self._classifier._backend,
            "timestamp":  time.time(),
        }


# ── Flask wiring ──────────────────────────────────────────────────────────
def create_flask_app():
    try:
        from flask import Flask, request, jsonify, Response
    except ImportError:
        logger.error("Flask not installed. Run: pip install flask")
        return None

    app    = Flask(__name__)
    server = PhantomServer()

    @app.route("/scan",   methods=["POST"])
    def scan():   return jsonify(server.handle_scan(request.json or {}))

    @app.route("/reveal", methods=["POST"])
    def reveal():
        return Response(server.handle_reveal(request.json or {}),
                        mimetype="application/octet-stream")

    @app.route("/scene",  methods=["GET","POST"])
    def scene():
        body = request.json or {}
        if request.method == "GET":
            body["session_id"] = request.args.get("session_id","")
        return jsonify(server.handle_scene(body))

    @app.route("/status", methods=["GET"])
    def status(): return jsonify(server.handle_status())

    return app


if __name__ == "__main__":
    app  = create_flask_app()
    port = int(os.environ.get("PORT", 8000))
    if app:
        logger.info(f"Starting on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
