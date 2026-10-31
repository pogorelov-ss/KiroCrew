"""The stateless OpenAI-compatible worker the extraction pool builds for
``knowledge.extraction_provider = "openai_compat"``.

One ``/chat/completions`` request per prompt, no history, nothing recycled. It refuses an
endpoint ``extraction_endpoint.endpoint_refusal`` refuses, and it never falls back to a
harness or a cloud model. ``LLMPool`` imports this module when it builds a worker, because
the worker subclasses ``llm_pool.Worker``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import aiohttp

from .extraction_endpoint import ExtractionEndpoint, ExtractionEndpointRefused, endpoint_refusal
from .llm_pool import DEFAULT_TIMEOUT, Worker

logger = logging.getLogger("kiro_crew.knowledge.llm_pool")

# Reply-length cap per request to an OpenAI-compatible endpoint. An extraction
# reply is ~600-900 tokens under the capped schema; 1500 leaves headroom without
# letting a runaway reply hold a slot for minutes.
EXTRACTION_MAX_TOKENS = 1500


class OpenAICompatWorker(Worker):
    """Stateless worker: one ``/chat/completions`` request per prompt.

    Every knowledge prompt is self-contained, so this worker sends exactly the
    one user message it is given and keeps no history. There is no transcript to
    recycle and no process to shield from the orphan sweep: ``reset_conversation``
    is a no-op and ``context_pct`` stays 0, so ``_maybe_recycle`` never fires.

    Failures raise. Nothing falls back to the harness or a cloud model when the
    endpoint is down: the pool's batch path stops sending the rest of the batch,
    the extractor records each unsent or failed chunk as a failed extraction (a
    folder sweep retries it later), and the log names the endpoint.
    """

    stateless = True

    def __init__(self, endpoint: ExtractionEndpoint) -> None:
        self._endpoint = endpoint
        self._session: Optional[aiohttp.ClientSession] = None
        # Token counts across this worker's life, for logs and A/B runs. The
        # spend view is estimate-only (knowledge/spend.py derives cost from rows
        # ingestion already writes), and a local endpoint has no cost, so these
        # are not reported there.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._warned_reasoning = False

    async def start(self) -> None:
        refusal = endpoint_refusal(self._endpoint)
        if refusal is not None:
            raise ExtractionEndpointRefused(refusal)
        if self._session is None or self._session.closed:
            # No Authorization header: a loopback server needs no key, and a
            # remote endpoint that does is out of scope until one is needed.
            self._session = aiohttp.ClientSession(headers={"Content-Type": "application/json"})

    def _body(self, prompt: str, response_format: Optional[dict]) -> dict:
        body: dict = {
            "model": self._endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": EXTRACTION_MAX_TOKENS,
            "stream": False,
            # Reasoning text would cost the whole token budget before the JSON
            # starts; templates that support thinking read this flag.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_format is not None:
            body["response_format"] = response_format
        return body

    async def send_message(
        self,
        prompt: str,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        response_format: Optional[dict] = None,
    ) -> str:
        """POST one request and return the reply text.

        The endpoint's configured timeout replaces *timeout*: callers size theirs
        for a harness session, while a local model serving several workers at
        once is slower per request.
        """
        if self._session is None or self._session.closed:
            await self.start()
        assert self._session is not None
        url = self._endpoint.chat_url
        request_timeout = aiohttp.ClientTimeout(total=float(self._endpoint.timeout_secs))
        try:
            async with self._session.post(
                url, json=self._body(prompt, response_format), timeout=request_timeout
            ) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(
                        f"extraction endpoint {url} returned HTTP {resp.status}: {raw[:300]}"
                    )
        except asyncio.TimeoutError as exc:
            # aiohttp's timeout carries no message, so name the endpoint here.
            message = f"{url} timed out after {self._endpoint.timeout_secs}s"
            logger.warning("OpenAICompatWorker: %s", message)
            raise asyncio.TimeoutError(message) from exc
        except aiohttp.ClientError as exc:
            logger.warning("OpenAICompatWorker: %s unreachable: %s", url, exc)
            raise
        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(choice, dict) or not isinstance(message, dict):
                raise TypeError("choice or message is not an object")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"extraction endpoint {url} returned a malformed reply") from exc
        self._record_usage(data.get("usage"))
        if message.get("reasoning_content") and not self._warned_reasoning:
            # The reply is still usable (only ``content`` is read), but the
            # model spent tokens thinking: the template ignored enable_thinking.
            self._warned_reasoning = True
            logger.warning(
                "OpenAICompatWorker: model %s returned reasoning text despite "
                "enable_thinking=false",
                self._endpoint.model,
            )
        if choice.get("finish_reason") == "length":
            logger.warning(
                "OpenAICompatWorker: reply hit max_tokens=%d; it will likely not parse",
                EXTRACTION_MAX_TOKENS,
            )
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _record_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int):
            self.prompt_tokens += prompt
        if isinstance(completion, int):
            self.completion_tokens += completion
        logger.debug(
            "OpenAICompatWorker: usage prompt=%s completion=%s (totals %d/%d)",
            prompt,
            completion,
            self.prompt_tokens,
            self.completion_tokens,
        )

    async def shutdown(self) -> None:
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

    def is_alive(self) -> bool:
        # No process to die. A down endpoint surfaces as a failed request, and
        # replacing the worker would not bring it back.
        return True

    async def reset_conversation(self) -> None:
        self.calls_since_reset = 0
