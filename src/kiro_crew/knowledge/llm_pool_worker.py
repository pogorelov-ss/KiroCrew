"""The worker interface every knowledge pool worker implements.

Its own module so a worker module can subclass it without importing
:mod:`kiro_crew.knowledge.llm_pool`, which imports the workers. ``llm_pool``
re-exports both names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

DEFAULT_TIMEOUT = 60.0


class Worker(ABC):
    """Abstract base for a long-lived LLM worker."""

    # Prompts served since this worker's conversation was last reset. The pool
    # reads it to decide when the transcript has grown far enough to be worth
    # dropping (see WORKER_RECYCLE_CALLS). Declared at class level, not set in an
    # ``__init__``, so a subclass that does not chain up still has the counter
    # (``+=`` rebinds it per instance).
    calls_since_reset: int = 0
    # True for a worker that keeps no transcript, so the pool never recycles it.
    stateless: bool = False

    @abstractmethod
    async def start(self) -> None:
        """Initialize/spawn the worker process."""

    @abstractmethod
    async def send_message(
        self,
        prompt: str,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a prompt and return the text response. *response_format* (an OpenAI
        ``response_format`` object) binds only an endpoint worker; sessions ignore it."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully destroy this worker."""

    @abstractmethod
    def is_alive(self) -> bool:
        """True if this worker can still process messages."""

    def context_pct(self) -> float:
        """Backend-reported context usage for this worker's conversation.

        ``0.0`` when the backend reports nothing (the pool then falls back to the
        call count), so a subclass with no telemetry needs no override.
        """
        return 0.0

    @abstractmethod
    async def reset_conversation(self) -> None:
        """Drop the accumulated transcript, leaving the worker ready to serve.

        Must leave ``is_alive()`` true on success so the pool's dead-worker
        replacement path stays reserved for genuine deaths.
        """
