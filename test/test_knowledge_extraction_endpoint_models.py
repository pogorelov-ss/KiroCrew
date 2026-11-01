"""``GET /api/knowledge/extraction-endpoint/models``: the settings tab's model list.

The route asks the configured OpenAI-compatible server for ``/models`` so the
dashboard can offer what it serves. It connects only to a URL the extraction
egress guard would accept, follows no redirect, bounds the reply, and answers the
owner only. Every server here is a real aiohttp server on loopback.
"""

from __future__ import annotations

import contextlib
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.dashboard.handlers import knowledge as kh

OWNER = "U0OWNER"


class _ModelServer:
    """``/v1/models`` that answers with ``reply`` (or ``status``, or a redirect)."""

    def __init__(self) -> None:
        self.requests = 0
        self.status = 200
        self.reply: object = {"object": "list", "data": []}
        self.raw: bytes | None = None
        self.redirect_to: str | None = None

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.requests += 1
        if self.redirect_to is not None:
            return web.Response(status=307, headers={"Location": self.redirect_to})
        if self.status != 200:
            return web.Response(status=self.status, text="no models here")
        if self.raw is not None:
            return web.Response(body=self.raw, content_type="application/json")
        return web.json_response(self.reply)


@contextlib.asynccontextmanager
async def _model_server() -> AsyncIterator[tuple[_ModelServer, str]]:
    fake = _ModelServer()
    app = web.Application()
    app.router.add_get("/v1/models", fake._handle)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    try:
        yield fake, f"http://127.0.0.1:{server.port}/v1"
    finally:
        await server.close()


def _gateway(user: str = OWNER) -> web.Application:
    @web.middleware
    async def _identity(request, handler):
        request["app"] = ""
        request["user"] = user
        return await handler(request)

    app = web.Application(middlewares=[_identity])
    state = MagicMock()
    state.owner_id = OWNER
    app["state"] = state
    app.router.add_get(
        "/api/knowledge/extraction-endpoint/models", kh.list_extraction_endpoint_models
    )
    return app


def _config(monkeypatch: pytest.MonkeyPatch, **knowledge: object) -> None:
    section = {"extraction_provider": "openai_compat", **knowledge}
    monkeypatch.setattr(kh, "_read_config", lambda: {"knowledge": section})


async def _get(user: str = OWNER) -> tuple[int, dict]:
    async with TestClient(TestServer(_gateway(user))) as client:
        resp = await client.get("/api/knowledge/extraction-endpoint/models")
        return resp.status, await resp.json()


class TestModelList:
    @pytest.mark.asyncio
    async def test_lists_the_served_model_ids_in_server_order(self, monkeypatch):
        async with _model_server() as (fake, base):
            fake.reply = {
                "data": [
                    {"id": "gemma-4-26B"},
                    {"id": "qwen3.6-35B"},
                    {"id": "gemma-4-26B"},
                    {"id": 7},
                    {"name": "no id"},
                    "not an object",
                    {"id": "x" * 257},
                ]
            }
            _config(monkeypatch, extraction_base_url=base)
            status, body = await _get()
        assert status == 200
        assert body == {"base_url": base, "models": ["gemma-4-26B", "qwen3.6-35B"], "error": None}

    @pytest.mark.asyncio
    async def test_lists_models_before_a_model_is_chosen(self, monkeypatch):
        async with _model_server() as (fake, base):
            fake.reply = {"data": [{"id": "gemma-4-26B"}]}
            _config(monkeypatch, extraction_base_url=base, extraction_endpoint_model="")
            _, body = await _get()
        assert body["models"] == ["gemma-4-26B"]

    @pytest.mark.asyncio
    async def test_caps_the_number_of_models(self, monkeypatch):
        async with _model_server() as (fake, base):
            fake.reply = {"data": [{"id": f"m{i}"} for i in range(kh.MODEL_LIST_MAX_MODELS + 5)]}
            _config(monkeypatch, extraction_base_url=base)
            _, body = await _get()
        assert len(body["models"]) == kh.MODEL_LIST_MAX_MODELS


class TestNothingIsAskedOfAHostTheGuardRefuses:
    @pytest.mark.asyncio
    async def test_no_base_url(self, monkeypatch):
        _config(monkeypatch)
        _, body = await _get()
        assert body["models"] == []
        assert "extraction_base_url is not set" in body["error"]

    @pytest.mark.asyncio
    async def test_a_remote_host_without_the_opt_in_is_not_contacted(self, monkeypatch):
        async def _must_not_fetch(url: str) -> list[str]:
            raise AssertionError(f"fetched {url}")

        monkeypatch.setattr(kh, "_fetch_model_ids", _must_not_fetch)
        _config(monkeypatch, extraction_base_url="http://models.example.com/v1")
        _, body = await _get()
        assert body["models"] == []
        assert "not a loopback address" in body["error"]

    @pytest.mark.asyncio
    async def test_a_redirect_is_not_followed(self, monkeypatch):
        async with (
            _model_server() as (elsewhere, elsewhere_base),
            _model_server() as (
                fake,
                base,
            ),
        ):
            elsewhere.reply = {"data": [{"id": "from-elsewhere"}]}
            fake.redirect_to = f"{elsewhere_base}/models"
            _config(monkeypatch, extraction_base_url=base)
            _, body = await _get()
        assert body["models"] == []
        assert "HTTP 307" in body["error"]
        assert fake.requests == 1
        assert elsewhere.requests == 0


class TestAFailureIsReportedNotRaised:
    @pytest.mark.asyncio
    async def test_http_error(self, monkeypatch):
        async with _model_server() as (fake, base):
            fake.status = 404
            _config(monkeypatch, extraction_base_url=base)
            status, body = await _get()
        assert status == 200
        assert body["models"] == []
        assert body["error"] == f"{base}/models: HTTP 404"

    @pytest.mark.asyncio
    async def test_a_reply_that_is_not_a_model_list(self, monkeypatch):
        async with _model_server() as (fake, base):
            fake.raw = b"<html>hello</html>"
            _config(monkeypatch, extraction_base_url=base)
            _, body = await _get()
        assert body["models"] == []
        assert body["error"].startswith(f"{base}/models: ")

    @pytest.mark.asyncio
    async def test_an_oversized_reply_is_cut_off(self, monkeypatch):
        monkeypatch.setattr(kh, "MODEL_LIST_MAX_BYTES", 64)
        async with _model_server() as (fake, base):
            fake.reply = {"data": [{"id": f"model-{i}"} for i in range(20)]}
            _config(monkeypatch, extraction_base_url=base)
            _, body = await _get()
        assert body["models"] == []
        assert "too large" in body["error"]

    @pytest.mark.asyncio
    async def test_a_stopped_server(self, monkeypatch):
        async with _model_server() as (_fake, base):
            pass
        _config(monkeypatch, extraction_base_url=base)
        _, body = await _get()
        assert body["models"] == []
        assert body["error"].startswith(f"{base}/models: ")


class TestOwnerOnly:
    @pytest.mark.asyncio
    async def test_a_non_owner_dashboard_session_is_refused_before_any_request(self, monkeypatch):
        async with _model_server() as (fake, base):
            fake.reply = {"data": [{"id": "gemma-4-26B"}]}
            _config(monkeypatch, extraction_base_url=base)
            status, body = await _get(user="U0NONOWNER")
        assert status == 403
        assert body.get("code") == "owner_only"
        assert fake.requests == 0
