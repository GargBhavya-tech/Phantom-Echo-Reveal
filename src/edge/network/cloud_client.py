"""
PHANTOM-ECHO REVEAL — Async Edge→Cloud HTTP Client
cloud_client.py

Handles all network communication from the edge (phone) to the cloud
(RunPod A100). Uses asyncio + aiohttp for non-blocking requests.

Features:
    - Async /scan streaming (NDJSON)
    - Async /reveal with SVQ-compressed response
    - Retry with exponential backoff (3 attempts)
    - Timeout: 3s for reveal, 30s for scan stream
    - Offline fallback: queues requests and retries when reconnected
    - Connection health check via /health

Flaw 40 fix: previous sync HTTP calls blocked the main render loop.
"""

import asyncio
import logging
import time
import json
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_CLOUD_URL = "http://localhost:8000"
REVEAL_TIMEOUT_S  = 3.0
SCAN_TIMEOUT_S    = 30.0
MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 0.5   # seconds


@dataclass
class CloudClientConfig:
    base_url:    str   = DEFAULT_CLOUD_URL
    api_key:     str   = ""
    reveal_timeout: float = REVEAL_TIMEOUT_S
    scan_timeout:   float = SCAN_TIMEOUT_S
    max_retries:    int   = MAX_RETRIES
    offline_queue_max: int = 50


class CloudClient:
    """
    Async HTTP client for edge→cloud communication.

    Usage:
        client = CloudClient(config)
        await client.connect()

        # Send scan frame
        result = await client.scan(scan_payload)

        # Reveal occluded region
        gaussians = await client.reveal(reveal_payload)
    """

    def __init__(self, config: Optional[CloudClientConfig] = None):
        self._cfg = config or CloudClientConfig()
        self._session = None
        self._connected = False
        self._offline_queue: List[Dict] = []
        self._last_health_check = 0.0

    async def connect(self) -> bool:
        """Initialize aiohttp session and verify connectivity."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._cfg.api_key}"},
                timeout=aiohttp.ClientTimeout(total=self._cfg.scan_timeout)
            )
            healthy = await self.health_check()
            self._connected = healthy
            if healthy:
                logger.info(f"CloudClient connected: {self._cfg.base_url}")
            else:
                logger.warning(f"CloudClient: server unreachable at {self._cfg.base_url}")
            return healthy
        except ImportError:
            logger.warning("aiohttp not installed — using sync fallback")
            self._connected = False
            return False
        except Exception as e:
            logger.warning(f"CloudClient connect failed: {e}")
            self._connected = False
            return False

    async def health_check(self) -> bool:
        """GET /health — returns True if server is up."""
        now = time.time()
        if now - self._last_health_check < 5.0:
            return self._connected
        try:
            if self._session is None:
                return False
            async with self._session.get(
                f"{self._cfg.base_url}/health",
                timeout=__import__("aiohttp").ClientTimeout(total=2.0)
            ) as resp:
                ok = resp.status == 200
                self._connected = ok
                self._last_health_check = now
                return ok
        except Exception:
            self._connected = False
            return False

    async def scan(self, payload: Dict[str, Any]) -> Optional[Dict]:
        """
        POST /scan — send one depth frame.

        Returns server response dict or None on failure.
        """
        return await self._post_json("/scan", payload, self._cfg.scan_timeout)

    async def reveal(self, payload: Dict[str, Any]) -> Optional[bytes]:
        """
        POST /reveal — request Gaussian generation for occluded region.

        Returns SVQ-compressed bytes or None on failure.
        Retries up to max_retries times with exponential backoff.
        """
        for attempt in range(self._cfg.max_retries):
            try:
                if self._session is None or not self._connected:
                    await self.connect()

                if not self._connected:
                    logger.warning("reveal: not connected, queuing offline")
                    self._queue_offline("reveal", payload)
                    return None

                import aiohttp
                async with self._session.post(
                    f"{self._cfg.base_url}/reveal",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._cfg.reveal_timeout)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        logger.info(f"reveal: {len(data)/1024:.1f}KB received")
                        return data
                    else:
                        text = await resp.text()
                        logger.warning(f"reveal HTTP {resp.status}: {text[:200]}")

            except asyncio.TimeoutError:
                logger.warning(f"reveal timeout (attempt {attempt+1}/{self._cfg.max_retries})")
            except Exception as e:
                logger.warning(f"reveal error: {e} (attempt {attempt+1})")

            if attempt < self._cfg.max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

        self._queue_offline("reveal", payload)
        return None

    async def get_scene(self, session_id: str) -> Optional[Dict]:
        """GET /scene/{session_id} — fetch full Gaussian scene."""
        try:
            if self._session is None:
                return None
            async with self._session.get(
                f"{self._cfg.base_url}/scene/{session_id}",
                timeout=__import__("aiohttp").ClientTimeout(total=5.0)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.warning(f"get_scene failed: {e}")
        return None

    async def scan_stream(self, payloads: List[Dict]) -> AsyncIterator[Dict]:
        """
        Stream multiple scan frames via NDJSON.
        Yields server responses as they arrive.
        """
        for payload in payloads:
            result = await self.scan(payload)
            if result:
                yield result
            await asyncio.sleep(0.033)   # ~30fps pacing

    # ── Sync wrappers (for non-async callers) ─────────────────────────────
    def reveal_sync(self, payload: Dict[str, Any]) -> Optional[bytes]:
        """Synchronous wrapper for reveal (runs in new event loop)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run():
            await self.connect()
            return await self.reveal(payload)

        return loop.run_until_complete(_run())

    def scan_sync(self, payload: Dict[str, Any]) -> Optional[Dict]:
        """Synchronous wrapper for scan."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run():
            await self.connect()
            return await self.scan(payload)

        return loop.run_until_complete(_run())

    # ── Offline queue ──────────────────────────────────────────────────────
    def _queue_offline(self, endpoint: str, payload: Dict) -> None:
        """Queue a request for retry when connectivity is restored."""
        if len(self._offline_queue) >= self._cfg.offline_queue_max:
            self._offline_queue.pop(0)
        self._offline_queue.append({
            "endpoint": endpoint,
            "payload": payload,
            "ts": time.time(),
        })
        logger.debug(f"Offline queue: {len(self._offline_queue)} pending")

    async def flush_offline_queue(self) -> int:
        """Retry all queued requests. Returns count successfully sent."""
        if not self._offline_queue:
            return 0
        if not await self.health_check():
            return 0

        sent = 0
        remaining = []
        for item in self._offline_queue:
            ep = item["endpoint"]
            payload = item["payload"]
            if ep == "scan":
                result = await self.scan(payload)
            elif ep == "reveal":
                result = await self.reveal(payload)
            else:
                result = None

            if result is not None:
                sent += 1
            else:
                remaining.append(item)

        self._offline_queue = remaining
        logger.info(f"Offline flush: {sent} sent, {len(remaining)} remaining")
        return sent

    # ── Internal ───────────────────────────────────────────────────────────
    async def _post_json(self,
                          endpoint: str,
                          payload: Dict,
                          timeout_s: float) -> Optional[Dict]:
        try:
            if self._session is None or not self._connected:
                return self._post_json_sync_fallback(endpoint, payload)
            import aiohttp
            async with self._session.post(
                f"{self._cfg.base_url}{endpoint}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning(f"{endpoint} HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"{endpoint} failed: {e}")
        return None

    def _post_json_sync_fallback(self,
                                   endpoint: str,
                                   payload: Dict) -> Optional[Dict]:
        """requests fallback when aiohttp unavailable."""
        try:
            import requests
            resp = requests.post(
                f"{self._cfg.base_url}{endpoint}",
                json=payload,
                timeout=REVEAL_TIMEOUT_S,
                headers={"Authorization": f"Bearer {self._cfg.api_key}"},
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"sync fallback failed: {e}")
        return None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
