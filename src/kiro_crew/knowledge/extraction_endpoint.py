"""Where knowledge extraction may send document chunks: the endpoint settings and the
egress guard.

``ExtractionEndpoint`` is the extraction pool's resolved ``knowledge.extraction_*``
configuration. ``endpoint_refusal`` decides whether an OpenAI-compatible endpoint may
receive chunks (loopback only unless ``knowledge.extraction_allow_remote``), and
``_audit_endpoint_decision`` writes the SEL row for each decision. ``LLMPool`` reads all
of it; the worker that talks to the endpoint is ``openai_compat_worker.OpenAICompatWorker``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from kiro_crew.config.fields import _safe_int
from kiro_crew.config.sections import (
    EXTRACTION_PROVIDER_ACP,
    EXTRACTION_PROVIDER_OPENAI_COMPAT,
    EXTRACTION_TIMEOUT_SECS_DEFAULT,
    EXTRACTION_TIMEOUT_SECS_MAX,
    EXTRACTION_TIMEOUT_SECS_MIN,
    normalize_extraction_provider,
)

logger = logging.getLogger("kiro_crew.knowledge.llm_pool")

# Config keys that select and describe the extraction endpoint. A change to any
# of them recycles an extraction pool at its next idle boundary.
_EXTRACTION_ENDPOINT_KEYS = (
    "knowledge.extraction_provider",
    "knowledge.extraction_base_url",
    "knowledge.extraction_endpoint_model",
    "knowledge.extraction_timeout_secs",
    "knowledge.extraction_allow_remote",
)
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})
# SEL identity for endpoint decisions: background work, no user session.
_SEL_SESSION_KEY = "_bg"
_SEL_AGENT = "knowledge-extraction"
_SEL_OPERATION = "knowledge.extraction_endpoint"


@dataclass(frozen=True)
class ExtractionEndpoint:
    """Resolved ``knowledge.extraction_*`` settings for the extraction pool."""

    provider: str = EXTRACTION_PROVIDER_ACP
    base_url: str = ""
    model: str = ""
    timeout_secs: int = EXTRACTION_TIMEOUT_SECS_DEFAULT
    allow_remote: bool = False

    @property
    def is_openai_compat(self) -> bool:
        return self.provider == EXTRACTION_PROVIDER_OPENAI_COMPAT

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


class ExtractionEndpointRefused(RuntimeError):
    """The configured extraction endpoint may not receive chunks."""


def _get_extraction_endpoint(config: dict) -> ExtractionEndpoint:
    """Resolve the extraction endpoint from ``knowledge.extraction_*`` of a raw config.

    Mirrors the loader's coercion and clamps, like the pool's other getters, so a
    hand-edited value never reaches the wire unbounded.
    """
    raw = config.get("knowledge", {})
    section = raw if isinstance(raw, dict) else {}

    def _text(key: str) -> str:
        value = section.get(key, "")
        return value.strip() if isinstance(value, str) else ""

    return ExtractionEndpoint(
        provider=normalize_extraction_provider(section.get("extraction_provider")),
        base_url=_text("extraction_base_url"),
        model=_text("extraction_endpoint_model"),
        timeout_secs=_safe_int(
            section.get("extraction_timeout_secs", EXTRACTION_TIMEOUT_SECS_DEFAULT),
            EXTRACTION_TIMEOUT_SECS_DEFAULT,
            EXTRACTION_TIMEOUT_SECS_MIN,
            EXTRACTION_TIMEOUT_SECS_MAX,
        ),
        allow_remote=section.get("extraction_allow_remote") is True,
    )


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and any loopback IP literal (127.0.0.0/8, ``::1``)."""
    host = host.strip().lower().rstrip(".")
    if host in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def endpoint_refusal(endpoint: ExtractionEndpoint) -> Optional[str]:
    """Why *endpoint* may not receive chunks, or ``None`` when it may.

    Chunks are the user's private documents, so a non-loopback host needs the
    explicit ``knowledge.extraction_allow_remote`` opt-in. A name that merely
    resolves to loopback is not trusted as loopback: only ``localhost`` and IP
    literals are, because anything else is a DNS answer someone else controls.
    """
    if not endpoint.base_url:
        return "knowledge.extraction_base_url is not set"
    if not endpoint.model:
        return "knowledge.extraction_endpoint_model is not set"
    parts = urlsplit(endpoint.base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return f"knowledge.extraction_base_url {endpoint.base_url!r} is not an http(s) URL"
    if not is_loopback_host(parts.hostname) and not endpoint.allow_remote:
        return (
            f"knowledge.extraction_base_url {endpoint.base_url!r} is not a loopback "
            "address; set knowledge.extraction_allow_remote to send document chunks "
            "to that host"
        )
    return None


def _audit_endpoint_decision(
    endpoint: ExtractionEndpoint, outcome: str, *, loopback: bool, error: str = ""
) -> None:
    """SEL record for an endpoint decision. Blocking; best-effort."""
    try:
        from kiro_crew.sel import sel

        sel().log_tool_invocation(
            session_key=_SEL_SESSION_KEY,
            agent=_SEL_AGENT,
            tool_name=_SEL_OPERATION,
            outcome=outcome,
            resources=f"base_url={endpoint.base_url} model={endpoint.model}",
            error=error,
            metadata={"allow_remote": endpoint.allow_remote, "loopback": loopback},
        )
    except Exception:
        logger.debug("LLMPool: SEL audit of extraction endpoint failed", exc_info=True)


class EndpointGate:
    """Refuse an endpoint that may not receive chunks, and audit each decision.

    Every decision gets a SEL row: each allow when the pool starts (a start
    follows an idle scale-to-zero or an endpoint change, so these are few),
    loopback included, and each refusal. A refused endpoint is re-checked on
    every checkout, so a refusal repeated for the same endpoint and reason is
    audited and logged once while this gate (one per pool) lives.
    """

    def __init__(self) -> None:
        self._reported_refusal: Optional[tuple[ExtractionEndpoint, str]] = None

    async def authorize(self, endpoint: ExtractionEndpoint) -> None:
        loopback = is_loopback_host(urlsplit(endpoint.base_url).hostname or "")
        refusal = endpoint_refusal(endpoint)
        if refusal is not None:
            if self._reported_refusal != (endpoint, refusal):
                self._reported_refusal = (endpoint, refusal)
                logger.error("LLMPool: extraction endpoint refused: %s", refusal)
                await asyncio.to_thread(
                    _audit_endpoint_decision, endpoint, "denied", loopback=loopback, error=refusal
                )
            raise ExtractionEndpointRefused(refusal)
        self._reported_refusal = None
        if not loopback:
            logger.warning(
                "LLMPool: extraction sends document chunks to remote endpoint %s "
                "(knowledge.extraction_allow_remote=true)",
                endpoint.base_url,
            )
        await asyncio.to_thread(_audit_endpoint_decision, endpoint, "approved", loopback=loopback)
