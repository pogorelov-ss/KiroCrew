"""The extraction endpoint's model list, for the Knowledge settings tab.

``GET /api/knowledge/extraction-endpoint/models`` asks the configured
OpenAI-compatible server (``knowledge.extraction_base_url``) which models it serves,
so the dashboard can offer them for ``knowledge.extraction_endpoint_model``.

Served through :mod:`kiro_crew.dashboard.handlers.knowledge`, which callers and tests import
and patch; this module never imports it.
"""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
from aiohttp import web

from kiro_crew.dashboard.handlers._shared import require_owner_dashboard_request
from kiro_crew.knowledge.extraction_endpoint import _get_extraction_endpoint, base_url_refusal
from kiro_crew.knowledge.llm_pool import _read_config

logger = logging.getLogger("kiro_crew.dashboard.handlers.knowledge")

# A model list is a few KB; anything far larger is not one.
MODEL_LIST_MAX_BYTES = 1024 * 1024
MODEL_LIST_TIMEOUT_SECS = 5.0
MODEL_LIST_MAX_MODELS = 200
# Same bound as the knowledge.extraction_endpoint_model write schema.
MODEL_ID_MAX_LEN = 256


def _model_ids(payload: object) -> list[str]:
    """The ``data[].id`` strings of an OpenAI ``/models`` reply, first occurrence kept."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("reply has no data list")
    ids: list[str] = []
    for entry in data:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and 0 < len(model_id) <= MODEL_ID_MAX_LEN:
            if model_id not in ids:
                ids.append(model_id)
        if len(ids) >= MODEL_LIST_MAX_MODELS:
            break
    return ids


async def _fetch_model_ids(models_url: str) -> list[str]:
    """GET *models_url* and return its model ids. Raises on any failure.

    No redirects, for the reason the extraction worker gives: the guard approved this
    host, not the one a redirect names.
    """
    timeout = aiohttp.ClientTimeout(total=MODEL_LIST_TIMEOUT_SECS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(models_url, allow_redirects=False) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            raw = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                raw += chunk
                if len(raw) > MODEL_LIST_MAX_BYTES:
                    raise RuntimeError("reply is too large to be a model list")
    return _model_ids(json.loads(bytes(raw)))


async def list_extraction_endpoint_models(request: web.Request) -> web.Response:
    """GET /api/knowledge/extraction-endpoint/models -- the endpoint's served models.

    Always 200 for the owner: ``models`` is empty and ``error`` names why when the URL
    is unset, refused by the egress guard, unreachable or not answering a model list,
    so the settings tab can fall back to a typed name.
    """
    owner_denied = await require_owner_dashboard_request(
        request, "knowledge.extraction_endpoint.models.read"
    )
    if owner_denied is not None:
        return owner_denied
    endpoint = _get_extraction_endpoint(await asyncio.to_thread(_read_config))
    body: dict = {"base_url": endpoint.base_url, "models": [], "error": None}
    refusal = base_url_refusal(endpoint)
    if refusal is not None:
        body["error"] = refusal
        return web.json_response(body)
    models_url = endpoint.base_url.rstrip("/") + "/models"
    try:
        body["models"] = await _fetch_model_ids(models_url)
    except asyncio.TimeoutError:
        body["error"] = f"{models_url} timed out after {MODEL_LIST_TIMEOUT_SECS:g}s"
    except (aiohttp.ClientError, RuntimeError, ValueError) as exc:
        body["error"] = f"{models_url}: {exc}"
    if body["error"]:
        logger.info("extraction endpoint model list unavailable: %s", body["error"])
    return web.json_response(body)
