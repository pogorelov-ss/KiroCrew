"""OpenAI-compatible extraction worker, its egress guard, and pool selection.

The worker talks to a real aiohttp server bound to loopback inside the test, so the
request body, statelessness and error handling are observed on the wire rather than
through a mocked client. The RFC Phase 4 exit criteria are pinned here: a
non-loopback URL is refused without ``extraction_allow_remote`` and audited
(``TestPoolSelectsTheWorker``); a stopped or hung endpoint costs at most one request
per worker per batch, records the chunks as failed and never reaches ACP
(``TestBatchStopsOnEndpointFailure``); and ACP extraction is unchanged by default
(``TestAcpDefaultIsUnchanged``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.config.memory_sections import KnowledgeConfig
from kiro_crew.config.sections import (
    EXTRACTION_TIMEOUT_SECS_DEFAULT,
    EXTRACTION_TIMEOUT_SECS_MAX,
    EXTRACTION_TIMEOUT_SECS_MIN,
)
from kiro_crew.knowledge import extraction_endpoint, llm_pool
from kiro_crew.knowledge.extraction_endpoint import (
    ExtractionEndpoint,
    ExtractionEndpointRefused,
    _get_extraction_endpoint,
    endpoint_refusal,
    is_loopback_host,
)
from kiro_crew.knowledge.extractor import (
    EXTRACTION_PROMPT,
    EXTRACTION_RESPONSE_FORMAT,
    EntityExtractor,
    extraction_failed,
)
from kiro_crew.knowledge.llm_pool import WORKER_RECYCLE_CALLS, AcpWorker, LLMPool
from kiro_crew.knowledge.openai_compat_worker import EXTRACTION_MAX_TOKENS, OpenAICompatWorker

pytestmark = pytest.mark.timeout(60)

# Test-only model name: the worker sends whatever the config names.
_MODEL = "local-test-model"
_REPLY = {
    "title": "Payments service design",
    "entities": [{"name": "PayAPI", "type": "service", "description": "Takes payments"}],
    "relations": [],
    "category": "design_doc",
    "summary": "The payments design.",
}


class _FakeEndpoint:
    """A minimal /chat/completions server that records every request body."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.headers: list[dict] = []
        self.status = 200
        self.delay = 0.0
        # Set: every request waits on it (a hung server); close() releases it.
        self.hold: asyncio.Event | None = None
        self.reply_message: dict = {"role": "assistant", "content": json.dumps(_REPLY)}
        self.server: TestServer | None = None

    async def _handle(self, request: web.Request) -> web.Response:
        self.bodies.append(await request.json())
        self.headers.append(dict(request.headers))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.hold is not None:
            await self.hold.wait()
        if self.status != 200:
            return web.Response(status=self.status, text="model exploded")
        return web.json_response(
            {
                "choices": [{"index": 0, "finish_reason": "stop", "message": self.reply_message}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
        )

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        self.server = TestServer(app, host="127.0.0.1")
        await self.server.start_server()
        return f"http://127.0.0.1:{self.server.port}/v1"

    async def close(self) -> None:
        if self.hold is not None:
            self.hold.set()
        if self.server is not None:
            await self.server.close()


@contextlib.asynccontextmanager
async def _serving() -> AsyncIterator[tuple[_FakeEndpoint, str]]:
    ep = _FakeEndpoint()
    base = await ep.start()
    try:
        yield ep, base
    finally:
        await ep.close()


def _endpoint(base: str, **overrides: object) -> ExtractionEndpoint:
    values: dict = {
        "provider": "openai_compat",
        "base_url": base,
        "model": _MODEL,
        "timeout_secs": 10,
    }
    values.update(overrides)
    return ExtractionEndpoint(**values)


class TestOpenAICompatWorker:
    @pytest.mark.asyncio
    async def test_request_body_carries_model_schema_and_disables_thinking(self):
        async with _serving() as (ep, base):
            worker = OpenAICompatWorker(_endpoint(base))
            await worker.start()
            try:
                reply = await worker.send_message(
                    "extract this", response_format=EXTRACTION_RESPONSE_FORMAT
                )
            finally:
                await worker.shutdown()

            assert json.loads(reply) == _REPLY
            body = ep.bodies[0]
            assert body["model"] == _MODEL
            assert body["messages"] == [{"role": "user", "content": "extract this"}]
            assert body["temperature"] == 0
            assert body["max_tokens"] == EXTRACTION_MAX_TOKENS
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            assert body["response_format"] == EXTRACTION_RESPONSE_FORMAT
            assert worker.prompt_tokens == 11 and worker.completion_tokens == 7

    @pytest.mark.asyncio
    async def test_no_response_format_when_caller_passes_none(self):
        async with _serving() as (ep, base):
            worker = OpenAICompatWorker(_endpoint(base))
            try:
                await worker.send_message("summarize these sections")
            finally:
                await worker.shutdown()
            assert "response_format" not in ep.bodies[0]

    @pytest.mark.asyncio
    async def test_stateless_second_call_sends_no_history(self):
        async with _serving() as (ep, base):
            worker = OpenAICompatWorker(_endpoint(base))
            try:
                await worker.send_message("first chunk")
                await worker.send_message("second chunk")
            finally:
                await worker.shutdown()
            assert [b["messages"] for b in ep.bodies] == [
                [{"role": "user", "content": "first chunk"}],
                [{"role": "user", "content": "second chunk"}],
            ]
            assert worker.stateless is True
            assert worker.context_pct() == 0.0
            assert worker.is_alive() is True

    @pytest.mark.asyncio
    async def test_http_error_raises_naming_the_endpoint(self):
        async with _serving() as (ep, base):
            ep.status = 500
            worker = OpenAICompatWorker(_endpoint(base))
            try:
                with pytest.raises(RuntimeError, match=r"HTTP 500.*model exploded"):
                    await worker.send_message("chunk")
            finally:
                await worker.shutdown()

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        async with _serving() as (ep, base):
            ep.delay = 5.0
            worker = OpenAICompatWorker(_endpoint(base, timeout_secs=0.2))
            try:
                with pytest.raises(asyncio.TimeoutError):
                    await worker.send_message("chunk")
            finally:
                await worker.shutdown()

    @pytest.mark.asyncio
    async def test_malformed_reply_raises(self):
        async with _serving() as (ep, base):
            ep.reply_message = None  # type: ignore[assignment]
            worker = OpenAICompatWorker(_endpoint(base))
            try:
                with pytest.raises(RuntimeError, match="malformed"):
                    await worker.send_message("chunk")
            finally:
                await worker.shutdown()

    @pytest.mark.asyncio
    async def test_only_content_is_returned_when_reasoning_leaks(self):
        async with _serving() as (ep, base):
            ep.reply_message = {
                "role": "assistant",
                "content": '{"summary": "s"}',
                "reasoning_content": "let me think about the chunk",
            }
            worker = OpenAICompatWorker(_endpoint(base))
            try:
                assert await worker.send_message("chunk") == '{"summary": "s"}'
            finally:
                await worker.shutdown()

    @pytest.mark.asyncio
    async def test_sends_no_authorization_header(self):
        """The provider has no API-key setting: nothing names a credential to send."""
        async with _serving() as (ep, base):
            worker = OpenAICompatWorker(_endpoint(base))
            try:
                await worker.send_message("chunk")
            finally:
                await worker.shutdown()
            assert "Authorization" not in ep.headers[0]

    @pytest.mark.asyncio
    async def test_start_refuses_a_remote_endpoint(self):
        worker = OpenAICompatWorker(_endpoint("https://llm.example.org/v1"))
        with pytest.raises(ExtractionEndpointRefused, match="extraction_allow_remote"):
            await worker.start()
        await worker.shutdown()

    @pytest.mark.asyncio
    async def test_extractor_parses_the_schema_constrained_reply(self):
        async with _serving() as (ep, base):
            pool = LLMPool(pool_size=1, use_extraction_provider=True)
            config = {
                "knowledge": {
                    "extraction_provider": "openai_compat",
                    "extraction_base_url": base,
                    "extraction_endpoint_model": _MODEL,
                }
            }
            with patch.object(llm_pool, "_read_config", return_value=config):
                try:
                    results = await EntityExtractor(pool=pool).extract_batch(
                        ["chunk one", "", "two"]
                    )
                finally:
                    await pool.shutdown()
            assert results[0]["entities"] == _REPLY["entities"]
            assert results[1]["entities"] == []
            assert results[2]["category"] == "design_doc"
            assert len(ep.bodies) == 2
            assert all(b["response_format"] == EXTRACTION_RESPONSE_FORMAT for b in ep.bodies)


class TestEgressGuard:
    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.8.9.10", "::1", "LOCALHOST"])
    def test_loopback_hosts(self, host):
        assert is_loopback_host(host)

    @pytest.mark.parametrize(
        "host", ["llm.lab.example", "192.168.1.5", "10.0.0.1", "0.0.0.0", "localhost.evil.com"]
    )
    def test_non_loopback_hosts(self, host):
        assert not is_loopback_host(host)

    @pytest.mark.parametrize(
        "url", ["http://localhost:13305/api/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8080/v1"]
    )
    def test_loopback_url_is_allowed_by_default(self, url):
        assert endpoint_refusal(_endpoint(url)) is None

    def test_remote_url_is_refused_by_default(self):
        reason = endpoint_refusal(_endpoint("https://llm.lab.example/v1"))
        assert reason is not None and "extraction_allow_remote" in reason

    def test_remote_url_is_allowed_with_the_opt_in(self):
        assert endpoint_refusal(_endpoint("https://llm.lab.example/v1", allow_remote=True)) is None

    @pytest.mark.parametrize("url", ["ftp://localhost/v1", "localhost:13305/v1", "file:///etc"])
    def test_non_http_url_is_refused(self, url):
        assert endpoint_refusal(_endpoint(url)) is not None

    def test_missing_url_or_model_is_refused(self):
        assert "base_url" in (endpoint_refusal(_endpoint("")) or "")
        assert "endpoint_model" in (
            endpoint_refusal(_endpoint("http://localhost/v1", model="")) or ""
        )


class TestEndpointConfig:
    def test_defaults_keep_extraction_on_acp(self):
        endpoint = _get_extraction_endpoint({})
        assert endpoint.provider == "acp"
        assert not endpoint.is_openai_compat
        assert endpoint.timeout_secs == EXTRACTION_TIMEOUT_SECS_DEFAULT
        assert endpoint.allow_remote is False

    def test_reads_and_clamps_the_knowledge_keys(self):
        endpoint = _get_extraction_endpoint(
            {
                "knowledge": {
                    "extraction_provider": "openai_compat",
                    "extraction_base_url": " http://localhost:13305/api/v1 ",
                    "extraction_endpoint_model": _MODEL,
                    "extraction_timeout_secs": 1,
                    "extraction_allow_remote": "yes",
                }
            }
        )
        assert endpoint.is_openai_compat
        assert endpoint.base_url == "http://localhost:13305/api/v1"
        assert endpoint.chat_url == "http://localhost:13305/api/v1/chat/completions"
        assert endpoint.model == _MODEL
        assert endpoint.timeout_secs == EXTRACTION_TIMEOUT_SECS_MIN
        # Only a real JSON true opts in to remote egress.
        assert endpoint.allow_remote is False

    def test_unknown_provider_falls_back_to_acp(self):
        assert (
            _get_extraction_endpoint(
                {"knowledge": {"extraction_provider": "cloud-anything"}}
            ).provider
            == "acp"
        )

    def test_loader_defaults_and_values(self, tmp_path: Path):
        def _load(knowledge: dict) -> KiroCrewConfig:
            (tmp_path / "config.json").write_text(json.dumps({"knowledge": knowledge}))
            with patch("kiro_crew.config.loader.config_dir", return_value=tmp_path):
                return KiroCrewConfig.load()

        default = _load({}).knowledge
        assert default.extraction_provider == "acp"
        assert default.extraction_base_url == ""
        assert default.extraction_endpoint_model == ""
        assert default.extraction_allow_remote is False
        assert default.extraction_timeout_secs == EXTRACTION_TIMEOUT_SECS_DEFAULT

        loaded = _load(
            {
                "extraction_provider": "openai_compat",
                "extraction_base_url": "http://localhost:13305/api/v1",
                "extraction_endpoint_model": _MODEL,
                "extraction_allow_remote": True,
                "extraction_timeout_secs": 10**9,
            }
        ).knowledge
        assert loaded.extraction_provider == "openai_compat"
        assert loaded.extraction_endpoint_model == _MODEL
        assert loaded.extraction_allow_remote is True
        assert loaded.extraction_timeout_secs == EXTRACTION_TIMEOUT_SECS_MAX

        # The loader agrees with the pool on hand-edited junk.
        junk = _load(
            {
                "extraction_provider": "cloud-anything",
                "extraction_base_url": 123,
                "extraction_allow_remote": "yes",
            }
        ).knowledge
        assert junk.extraction_provider == "acp"
        assert junk.extraction_base_url == ""
        assert junk.extraction_allow_remote is False

    def test_no_credential_or_token_cap_key(self):
        """Five keys only: no key names a credential env var, and max_tokens is a constant."""
        names = {f for f in KnowledgeConfig.__dataclass_fields__ if f.startswith("extraction_")}
        assert {
            "extraction_provider",
            "extraction_base_url",
            "extraction_endpoint_model",
            "extraction_timeout_secs",
            "extraction_allow_remote",
        } <= names
        assert not any("api_key" in n or "max_tokens" in n for n in names)

    def test_dashboard_write_schema_validates_the_new_keys(self):
        from kiro_crew.dashboard.handlers.core import _EDITABLE_CONFIG

        assert _EDITABLE_CONFIG["knowledge.extraction_provider"]["values"] == [
            "acp",
            "openai_compat",
        ]
        timeout = _EDITABLE_CONFIG["knowledge.extraction_timeout_secs"]
        assert (timeout["min"], timeout["max"]) == (
            EXTRACTION_TIMEOUT_SECS_MIN,
            EXTRACTION_TIMEOUT_SECS_MAX,
        )
        assert "knowledge.extraction_api_key_env" not in _EDITABLE_CONFIG
        assert "knowledge.extraction_max_tokens" not in _EDITABLE_CONFIG

    def test_where_chunks_may_go_is_config_file_only(self):
        """Neither the PATCH route nor an agent's change card can redirect chunks."""
        from kiro_crew.dashboard.handlers.core import _EDITABLE_CONFIG

        assert "knowledge.extraction_base_url" not in _EDITABLE_CONFIG
        assert "knowledge.extraction_allow_remote" not in _EDITABLE_CONFIG


class TestPoolSelectsTheWorker:
    @pytest.mark.asyncio
    async def test_extraction_pool_builds_openai_compat_workers(self):
        pool = LLMPool(pool_size=1, use_extraction_provider=True)
        pool._endpoint = _endpoint("http://localhost:13305/api/v1")
        worker = await pool._create_worker()
        try:
            assert isinstance(worker, OpenAICompatWorker)
        finally:
            await worker.shutdown()

    @pytest.mark.asyncio
    async def test_pool_that_does_not_opt_in_ignores_the_provider(self):
        """A pool that does not opt in (auto_research) stays on a harness."""
        config = {
            "knowledge": {
                "extraction_provider": "openai_compat",
                "extraction_base_url": "http://localhost:13305/api/v1",
                "extraction_endpoint_model": _MODEL,
            }
        }
        pool = LLMPool(pool_size=1)
        created: list[object] = []

        class _Acp:
            def __init__(self, **kwargs: object) -> None:
                created.append(self)

            async def start(self) -> None:
                return None

            async def shutdown(self) -> None:
                return None

        with (
            patch.object(llm_pool, "_read_config", return_value=config),
            patch.object(llm_pool, "AcpWorker", _Acp),
        ):
            await pool.start()
            await pool.shutdown()
        assert len(created) == 1 and not pool._endpoint.is_openai_compat

    @pytest.mark.asyncio
    async def test_acp_default_builds_acp_workers(self):
        pool = LLMPool(pool_size=1, use_extraction_provider=True)
        pool._provider_type = "acp"
        with patch.object(llm_pool, "AcpWorker") as worker_type:
            worker_type.return_value.start = _async_none
            worker = await pool._create_worker()
        assert worker is worker_type.return_value
        assert not isinstance(worker, OpenAICompatWorker)
        assert AcpWorker is not worker_type  # the real class is untouched outside the patch

    @pytest.mark.asyncio
    async def test_refused_endpoint_fails_start_and_is_audited_once(self):
        config = {
            "knowledge": {
                "extraction_provider": "openai_compat",
                "extraction_base_url": "https://llm.lab.example/v1",
                "extraction_endpoint_model": _MODEL,
            }
        }
        audits: list[tuple] = []
        pool = LLMPool(pool_size=1, use_extraction_provider=True)
        with (
            patch.object(llm_pool, "_read_config", return_value=config),
            patch.object(
                extraction_endpoint, "_audit_endpoint_decision", lambda *a, **k: audits.append(a)
            ),
        ):
            for _ in range(3):
                with pytest.raises(ExtractionEndpointRefused):
                    await pool.start()
        assert len(audits) == 1 and audits[0][1] == "denied"
        assert pool._workers == []

    @pytest.mark.asyncio
    async def test_remote_opt_in_is_audited_as_approved(self):
        config = {
            "knowledge": {
                "extraction_provider": "openai_compat",
                "extraction_base_url": "https://llm.lab.example/v1",
                "extraction_endpoint_model": _MODEL,
                "extraction_allow_remote": True,
            }
        }
        audits: list[tuple] = []
        pool = LLMPool(pool_size=1, use_extraction_provider=True)
        with (
            patch.object(llm_pool, "_read_config", return_value=config),
            patch.object(
                extraction_endpoint,
                "_audit_endpoint_decision",
                lambda *a, **k: audits.append((*a, k)),
            ),
        ):
            await pool.start()
            await pool.shutdown()
        assert [a[1] for a in audits] == ["approved"]
        assert audits_kwargs_loopback(audits) == [False]

    @pytest.mark.asyncio
    async def test_loopback_allow_is_audited_on_every_start(self):
        config = {
            "knowledge": {
                "extraction_provider": "openai_compat",
                "extraction_base_url": "http://localhost:13305/api/v1",
                "extraction_endpoint_model": _MODEL,
            }
        }
        audits: list[tuple] = []
        pool = LLMPool(pool_size=1, use_extraction_provider=True)
        with (
            patch.object(llm_pool, "_read_config", return_value=config),
            patch.object(
                extraction_endpoint,
                "_audit_endpoint_decision",
                lambda *a, **k: audits.append((*a, k)),
            ),
        ):
            for _ in range(2):
                await pool.start()
                await pool.shutdown()
        assert [a[1] for a in audits] == ["approved", "approved"]
        assert audits_kwargs_loopback(audits) == [True, True]

    def test_audit_row_names_the_operation_and_endpoint(self):
        rows: list[dict] = []

        class _Sel:
            def log_tool_invocation(self, **kwargs: object) -> None:
                rows.append(kwargs)

        endpoint = _endpoint("https://llm.lab.example/v1")
        with patch("kiro_crew.sel.sel", return_value=_Sel()):
            extraction_endpoint._audit_endpoint_decision(
                endpoint, "denied", loopback=False, error="not a loopback address"
            )
        assert rows == [
            {
                "session_key": "_bg",
                "agent": "knowledge-extraction",
                "tool_name": "knowledge.extraction_endpoint",
                "outcome": "denied",
                "resources": f"base_url=https://llm.lab.example/v1 model={_MODEL}",
                "error": "not a loopback address",
                "metadata": {"allow_remote": False, "loopback": False},
            }
        ]

    @pytest.mark.asyncio
    async def test_stateless_worker_is_never_recycled(self):
        async with _serving() as (ep, base):
            pool = LLMPool(pool_size=1, use_extraction_provider=True)
            config = {
                "knowledge": {
                    "extraction_provider": "openai_compat",
                    "extraction_base_url": base,
                    "extraction_endpoint_model": _MODEL,
                }
            }
            with patch.object(llm_pool, "_read_config", return_value=config):
                try:
                    await pool.start()
                    worker = pool._workers[0]
                    for _ in range(WORKER_RECYCLE_CALLS + 2):
                        await pool.send("chunk")
                    assert pool._workers[0] is worker
                    assert worker.calls_since_reset == 0
                finally:
                    await pool.shutdown()
            assert len(ep.bodies) == WORKER_RECYCLE_CALLS + 2

    @pytest.mark.asyncio
    async def test_endpoint_change_recycles_the_idle_pool(self):
        async with _serving() as (_ep, base):
            pool = LLMPool(pool_size=1, use_extraction_provider=True)
            first = {
                "knowledge": {
                    "extraction_provider": "openai_compat",
                    "extraction_base_url": base,
                    "extraction_endpoint_model": _MODEL,
                    "pool_idle_ttl_secs": 0,
                }
            }
            second = {
                "knowledge": {**first["knowledge"], "extraction_endpoint_model": "other-model"}
            }
            with patch.object(llm_pool, "_read_config", return_value=first):
                await pool.start()
            try:
                await pool.reconfigure(second)
                assert pool._started is False  # idle, so recycled at once
                with patch.object(llm_pool, "_read_config", return_value=second):
                    await pool.send("chunk")
                assert pool._running_endpoint.model == "other-model"
            finally:
                await pool.shutdown()


def audits_kwargs_loopback(audits: list[tuple]) -> list[bool]:
    return [a[-1]["loopback"] for a in audits]


class _NoHarness:
    """Stands in for AcpWorker and CCWorker: building one fails the test."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("a chunk was routed to the ACP harness")


def _openai_config(base: str, **extra: object) -> dict:
    return {
        "knowledge": {
            "extraction_provider": "openai_compat",
            "extraction_base_url": base,
            "extraction_endpoint_model": _MODEL,
            **extra,
        }
    }


@contextlib.contextmanager
def _counting_sends() -> "contextlib.AbstractContextManager[list[str]]":
    """Record every prompt an OpenAICompatWorker is asked to send."""
    sent: list[str] = []
    original = OpenAICompatWorker.send_message

    async def _counted(self, prompt, timeout=llm_pool.DEFAULT_TIMEOUT, **kwargs):
        sent.append(prompt)
        return await original(self, prompt, timeout, **kwargs)

    with patch.object(OpenAICompatWorker, "send_message", _counted):
        yield sent


class TestBatchStopsOnEndpointFailure:
    async def _extract(self, config: dict, chunks: list[str], *, pool_size: int) -> list[dict]:
        pool = LLMPool(pool_size=pool_size, use_extraction_provider=True)
        with (
            patch.object(llm_pool, "_read_config", return_value=config),
            patch.object(llm_pool, "AcpWorker", _NoHarness),
            patch.object(llm_pool, "CCWorker", _NoHarness),
        ):
            try:
                return await EntityExtractor(pool=pool).extract_batch(chunks)
            finally:
                await pool.shutdown()

    @pytest.mark.asyncio
    async def test_stopped_endpoint_sends_one_request_and_records_every_chunk_failed(self):
        async with _serving() as (_ep, base):
            pass  # the server is gone: connections to its port are refused
        chunks = [f"chunk {i}" for i in range(6)]
        with _counting_sends() as sent:
            results = await self._extract(_openai_config(base), chunks, pool_size=1)
        assert len(sent) == 1
        assert all(extraction_failed(r) for r in results)

    @pytest.mark.asyncio
    async def test_hung_endpoint_costs_one_timeout_per_worker_not_per_chunk(self):
        async with _serving() as (ep, base):
            ep.hold = asyncio.Event()
            chunks = [f"chunk {i}" for i in range(8)]
            pool = LLMPool(pool_size=2, use_extraction_provider=True)
            # Below the config floor on purpose: the test needs a short timeout.
            pool_endpoint = _endpoint(base, timeout_secs=0.3)
            with (
                patch.object(llm_pool, "_read_config", return_value=_openai_config(base)),
                patch.object(llm_pool, "_get_extraction_endpoint", return_value=pool_endpoint),
                patch.object(llm_pool, "AcpWorker", _NoHarness),
                patch.object(llm_pool, "CCWorker", _NoHarness),
            ):
                try:
                    results = await EntityExtractor(pool=pool).extract_batch(chunks)
                finally:
                    await pool.shutdown()
            # Only the requests already in flight when the first one timed out.
            assert 1 <= len(ep.bodies) <= 2
            assert all(extraction_failed(r) for r in results)

    @pytest.mark.asyncio
    async def test_http_error_stops_the_rest_of_the_batch(self):
        async with _serving() as (ep, base):
            ep.status = 503
            results = await self._extract(_openai_config(base), ["a", "b", "c", "d"], pool_size=1)
            assert len(ep.bodies) == 1
            assert all(extraction_failed(r) for r in results)

    @pytest.mark.asyncio
    async def test_refused_endpoint_records_chunks_failed_without_sending(self):
        config = _openai_config("https://llm.lab.example/v1")
        with (
            _counting_sends() as sent,
            patch.object(extraction_endpoint, "_audit_endpoint_decision", lambda *a, **k: None),
        ):
            results = await self._extract(config, ["a", "b", "c"], pool_size=1)
        assert sent == []
        assert all(extraction_failed(r) for r in results)

    @pytest.mark.asyncio
    async def test_an_unparseable_reply_does_not_stop_the_batch(self):
        """A bad reply is that chunk's problem; the endpoint answered."""
        async with _serving() as (ep, base):
            ep.reply_message = {"role": "assistant", "content": "no json here"}
            results = await self._extract(_openai_config(base), ["a", "b", "c"], pool_size=1)
            assert len(ep.bodies) == 3
            assert all(extraction_failed(r) for r in results)

    @pytest.mark.asyncio
    async def test_a_later_batch_tries_the_endpoint_again(self):
        async with _serving() as (ep, base):
            ep.status = 500
            pool = LLMPool(pool_size=1, use_extraction_provider=True)
            with patch.object(llm_pool, "_read_config", return_value=_openai_config(base)):
                try:
                    extractor = EntityExtractor(pool=pool)
                    first = await extractor.extract_batch(["a", "b"])
                    ep.status = 200
                    second = await extractor.extract_batch(["c", "d"])
                finally:
                    await pool.shutdown()
            assert all(extraction_failed(r) for r in first)
            assert [r["entities"] for r in second] == [_REPLY["entities"]] * 2
            assert len(ep.bodies) == 3

    @pytest.mark.asyncio
    async def test_harness_batch_still_sends_every_item_after_a_failure(self):
        """The stop applies to the endpoint only; ACP batches are unchanged."""
        sent: list[str] = []

        class _FlakyAcp:
            stateless = False
            calls_since_reset = 0

            def __init__(self, **kwargs: object) -> None:
                pass

            async def start(self) -> None:
                return None

            async def send_message(self, prompt: str, timeout: float = 0, **kwargs: object) -> str:
                sent.append(prompt)
                if len(sent) == 1:
                    raise RuntimeError("turn failed")
                return json.dumps(_REPLY)

            def is_alive(self) -> bool:
                return True

            def context_pct(self) -> float:
                return 0.0

            async def shutdown(self) -> None:
                return None

        pool = LLMPool(pool_size=1, use_extraction_provider=True)
        with (
            patch.object(llm_pool, "_read_config", return_value={}),
            patch.object(llm_pool, "AcpWorker", _FlakyAcp),
            patch.object(llm_pool, "_get_provider_type", return_value="acp"),
        ):
            try:
                results = await EntityExtractor(pool=pool).extract_batch(["a", "b", "c"])
            finally:
                await pool.shutdown()
        assert len(sent) == 3
        assert [extraction_failed(r) for r in results] == [True, False, False]


class TestHungEndpointDuringIngest:
    @pytest.mark.asyncio
    async def test_an_ingest_costs_one_request_per_worker_and_no_summary_request(self, tmp_path):
        """The whole ingest, not just its batch: a document whose every chunk
        failed sends no source-summary request after the batch."""
        from unittest.mock import MagicMock

        from kiro_crew.knowledge.folder_watcher import FolderWatcher
        from kiro_crew.knowledge.ingestion import IngestionPipeline
        from kiro_crew.knowledge.readers import FileReader
        from kiro_crew.knowledge.store import KnowledgeStore

        def _four_chunks(text, **_kw):
            return [
                {
                    "content": f"{text} part {i}",
                    "chunk_index": i,
                    "section_title": None,
                    "line_start": 0,
                    "line_end": 0,
                }
                for i in range(4)
            ]

        chunker = MagicMock()
        chunker.chunk.side_effect = _four_chunks
        chunker.chunk_markdown.side_effect = _four_chunks
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "first.md").write_text("first", encoding="utf-8")
        store = KnowledgeStore(str(tmp_path / "knowledge.db"))
        async with _serving() as (ep, base):
            pool = LLMPool(pool_size=2, use_extraction_provider=True)
            pipeline = IngestionPipeline(
                store=store,
                extractor=EntityExtractor(pool=pool),
                chunker=chunker,
                reader=FileReader(),
                embedder=None,
            )
            watcher = FolderWatcher(store, pipeline)
            source_id = await asyncio.to_thread(
                store.add_source, "docs", "local_folder", str(folder)
            )
            source = {
                "id": source_id,
                "uri": str(folder),
                "source_type": "local_folder",
                "properties": "{}",
            }
            with (
                patch.object(llm_pool, "_read_config", return_value=_openai_config(base)),
                patch.object(llm_pool, "AcpWorker", _NoHarness),
                patch.object(llm_pool, "CCWorker", _NoHarness),
            ):
                try:
                    # The first file extracts, so the source has chunk summaries
                    # and a summary request would have something to send.
                    await watcher.scan_source(source)
                    assert len(ep.bodies) == 5  # 4 chunks + the source summary
                    ep.bodies.clear()
                    ep.hold = asyncio.Event()
                    await pool.shutdown()
                    (folder / "second.md").write_text("second", encoding="utf-8")
                    # Below the config floor on purpose: the test needs a short timeout.
                    hung = _endpoint(base, timeout_secs=0.3)
                    with patch.object(llm_pool, "_get_extraction_endpoint", return_value=hung):
                        await watcher.scan_source(source)
                finally:
                    await pool.shutdown()
            assert 1 <= len(ep.bodies) <= 2
            assert all("second part" in b["messages"][0]["content"] for b in ep.bodies)
        store.close()


class TestAcpDefaultIsUnchanged:
    @pytest.mark.asyncio
    async def test_default_config_sends_the_same_prompt_to_the_harness(self):
        calls: list[tuple[str, dict]] = []

        class _RecordingAcp:
            stateless = False
            calls_since_reset = 0

            def __init__(self, **kwargs: object) -> None:
                pass

            async def start(self) -> None:
                return None

            async def send_message(self, prompt: str, **kwargs: object) -> str:
                calls.append((prompt, kwargs))
                return json.dumps(_REPLY)

            def is_alive(self) -> bool:
                return True

            def context_pct(self) -> float:
                return 0.0

            async def shutdown(self) -> None:
                return None

        pool = LLMPool(pool_size=1, use_extraction_provider=True)
        with (
            patch.object(llm_pool, "_read_config", return_value={}),
            patch.object(llm_pool, "AcpWorker", _RecordingAcp),
            patch.object(llm_pool, "_get_provider_type", return_value="acp"),
            patch.object(extraction_endpoint, "_audit_endpoint_decision") as audit,
        ):
            try:
                await EntityExtractor(pool=pool).extract_batch(["the chunk"])
            finally:
                await pool.shutdown()
        assert not pool._endpoint.is_openai_compat
        audit.assert_not_called()
        [(prompt, kwargs)] = calls
        nonce = re.search(r"<<<BEGIN_UNTRUSTED_CHUNK_(\w+)>>>", prompt).group(1)
        assert prompt == EXTRACTION_PROMPT.format(
            chunk="the chunk",
            begin_marker=f"<<<BEGIN_UNTRUSTED_CHUNK_{nonce}>>>",
            end_marker=f"<<<END_UNTRUSTED_CHUNK_{nonce}>>>",
        )
        assert kwargs["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_acp_worker_ignores_response_format(self):
        """The harness wire call is the same with or without a reply schema."""
        received: list[tuple] = []

        class _Client:
            is_ready = True

            async def send_message(self, prompt: str, timeout: float | None = None) -> str:
                received.append((prompt, timeout))
                return "ok"

        worker = AcpWorker.__new__(AcpWorker)
        worker._client = _Client()  # type: ignore[assignment]
        await worker.send_message("p", timeout=60.0)
        await worker.send_message("p", timeout=60.0, response_format=EXTRACTION_RESPONSE_FORMAT)
        assert received == [("p", 60.0), ("p", 60.0)]


async def _async_none() -> None:
    return None
