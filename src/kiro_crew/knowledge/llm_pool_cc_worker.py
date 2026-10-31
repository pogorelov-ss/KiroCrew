"""The external agent CLI worker the knowledge pool runs on the Claude Code provider.

:class:`~kiro_crew.knowledge.llm_pool.LLMPool` builds a :class:`CCWorker` when the
provider is the Claude Code CLI, and ``llm_pool`` re-exports it, so callers keep
importing it from there. The pool module's names are read through it at call time,
so a patch on ``kiro_crew.knowledge.llm_pool`` still reaches this worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any, Optional

from kiro_crew import platform_compat
from kiro_crew.json_line import parse_json_object_line

from .llm_pool_worker import DEFAULT_TIMEOUT, Worker

logger = logging.getLogger("kiro_crew.knowledge.llm_pool")

#: Longest stream-json line :class:`CCWorker` reads; the ACP transports' ceiling
#: (``acp.transport_framing._STDOUT_BUFFER_LIMIT``, pinned equal by a test so
#: this module stays importable without the ACP client).
STDOUT_LINE_LIMIT = 10 * 1024 * 1024


def _pool() -> Any:
    """The pool module, read at call time so a patch on it reaches this worker."""
    from kiro_crew.knowledge import llm_pool

    return llm_pool


class CCWorkerReplyError(RuntimeError):
    """The CLI answered this message with an error result."""


def _text_blocks(container: Any) -> list[str]:
    """The ``text`` of each text block in ``container["content"]``, skipping bad shapes."""
    content = container.get("content") if isinstance(container, dict) else None
    if not isinstance(content, list):
        return []
    return [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]


class CCWorker(Worker):
    """Long-lived external agent CLI subprocess using stream-json I/O."""

    def __init__(self, *, sandbox_mode: Optional[str] = None) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._event_queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
        self._claude_bin: Optional[str] = None
        # The operator's ``agent.sandbox`` tier, pre-resolved by the pool off
        # the event loop -- the same argument, from the same reader
        # (``_get_sandbox_mode``), that ``AcpWorker`` takes. Every agent child
        # answers to the one configured tier, so this worker must receive it
        # rather than take ``wrap_argv``'s own default. ``None`` -> resolve
        # lazily in ``_spawn`` (direct construction outside the pool / tests).
        self._sandbox_mode = sandbox_mode

    async def start(self) -> None:
        self._claude_bin = shutil.which("claude")
        if not self._claude_bin:
            raise RuntimeError("claude CLI not found in PATH")
        await self._spawn()

    async def _spawn(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        try:
            await self._spawn_process()
        except BaseException:
            # The old reader is gone, so the old process can deliver nothing
            # any more: retire it, and the worker reads dead until a spawn
            # succeeds instead of being handed out to a caller it cannot answer.
            await self.shutdown()
            raise

    async def _spawn_process(self) -> None:
        assert self._claude_bin is not None
        cmd = [
            self._claude_bin,
            "-p",
            "--verbose",
            "--model",
            "haiku",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "bypassPermissions",
        ]
        # Optional URL-fetch tool. Empty by default (no built-in remote fetch on a
        # vanilla machine). Users can opt in by setting KIROCREW_KNOWLEDGE_FETCH_TOOLS
        # to a comma-separated list of MCP tool names their backend exposes.
        fetch_tools = os.environ.get("KIROCREW_KNOWLEDGE_FETCH_TOOLS", "").strip()
        if fetch_tools:
            cmd += ["--allowedTools", fetch_tools]
        # Resolved per spawn into a LOCAL, never stored: ``_spawn`` also serves
        # the respawn paths (``send_message``'s recovery, ``reset_conversation``),
        # so a stored value would pin every later worker process to the tier read
        # at the first one. Same expression, same reader, same reason as
        # ``AcpWorker.start``.
        sandbox_mode = (
            self._sandbox_mode
            if self._sandbox_mode is not None
            else await asyncio.to_thread(_pool()._get_sandbox_mode)
        )
        # ``strip_python_env`` for the same reason every other agent spawn
        # passes it: this is a FOREIGN runtime, and any Python it reaches for
        # must not inherit Kiro Crew's interpreter paths.
        wrapped = _pool().cgroup_scope_argv(
            (
                await _pool().wrap_argv_async(
                    cmd,
                    mode=sandbox_mode,
                    strip_python_env=True,
                    _prepare=_pool().wrap_argv,
                )
            )[0]
        )  # cgroup DoS ceiling
        self._proc = await _pool().create_subprocess_limited(
            *wrapped,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Nothing reads it, and an unread pipe both holds what the CLI
            # writes (up to twice the raised ``limit`` below, per worker) and
            # stalls the CLI once it fills.
            stderr=asyncio.subprocess.DEVNULL,
            # Own process group, so the worker's descendants are reachable as a tree.
            # `claude -p` forks helpers (see spine/agent_runner.py's _terminate_group),
            # and without this the child lands in the gateway's OWN group -- which is
            # the one case platform_compat.kill_and_reap deliberately skips its group
            # kill for, so the reap below would degrade to a pid-scoped kill and
            # re-parent those helpers to init.
            #
            # BOTH args, spelled out explicitly, per the recipe in platform_compat:
            # on POSIX start_new_session calls setsid (so killpg reaps the group) and
            # creationflags=0 is a no-op; on Windows start_new_session is silently
            # ignored and CREATE_NEW_PROCESS_GROUP is what makes the child tree
            # `taskkill /T`-reapable. Not **unpacked from a dict -- that defeats
            # mypy's Popen overload resolution on the build fleet.
            start_new_session=platform_compat.IS_POSIX,
            creationflags=platform_compat.CREATE_NEW_PROCESS_GROUP,
            # The agent enforcement point, passed exactly as the agent-CLI
            # spawns in ``dashboard/handlers/sessions.py`` and
            # ``sessions_parts/usage_scrape.py`` pass it. This worker
            # runs with its permission prompts disabled and its inputs are
            # knowledge documents, so gateway-owned credentials must not be
            # readable inside it. The tier does not cover this: no tier below
            # ``strict`` scrubs the agent-denied keys, so the parent side is
            # where they are removed.
            env=_pool().scrub_agent_subprocess_env(),
            # A stream-json line carries a whole event, a fetched page echoed
            # back in a tool_result included; asyncio's 64 KiB default would
            # refuse it. Same ceiling as the ACP transports. stdout is the only
            # pipe, so it is the only reader this limit sizes.
            limit=STDOUT_LINE_LIMIT,
        )
        self._event_queue = asyncio.Queue()
        self._reader_task = asyncio.create_task(self._stdout_reader())

    async def _stdout_reader(self) -> None:
        """Background task: read NDJSON lines from stdout into event queue.

        A line that is not a JSON object, or is over :data:`STDOUT_LINE_LIMIT`,
        is dropped and the read goes on; the reader ends only with the stream.
        """
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                try:
                    line = await self._proc.stdout.readline()
                except ValueError:
                    # Over the limit: asyncio has discarded what it buffered,
                    # and the rest of that line arrives as one more line that
                    # does not parse.
                    continue
                if not line:
                    break
                event = parse_json_object_line(line)
                if event is not None:
                    await self._event_queue.put(event)
        except Exception:
            # A read error ends the reader the same way the end of the stream does.
            pass
        finally:
            await self._event_queue.put(None)

    async def send_message(
        self,
        prompt: str,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        response_format: Optional[dict] = None,
    ) -> str:
        if self._proc is None or self._proc.returncode is not None:
            await self._spawn()
        assert self._proc is not None and self._proc.stdin is not None

        stdin = self._proc.stdin
        msg = json.dumps({"type": "user", "message": {"role": "user", "content": prompt}})

        async def _deliver_and_collect() -> str:
            stdin.write((msg + "\n").encode())
            await stdin.drain()
            return await self._collect_response()

        # The queue may hold only this prompt's events, so the worker survives
        # a message only when its result event was consumed. Anything else (a
        # timeout, a dead reader, a reply that broke off, a cancelled caller)
        # leaves the rest of THIS reply queued, and the next prompt would read
        # it as its own: retire the process instead. The write and its drain
        # sit inside the same guard and the same timeout, because a prompt
        # over the pipe's high-water mark is already delivered while drain()
        # waits for the CLI to read it.
        settled = False
        try:
            reply = await asyncio.wait_for(_deliver_and_collect(), timeout=timeout)
            settled = True
            return reply
        except CCWorkerReplyError:
            settled = True  # the error result is the result event
            raise
        finally:
            if not settled:
                await self.shutdown()

    async def _collect_response(self) -> str:
        """Collect text from events until a result event arrives.

        A value of the wrong type is skipped at its own level. The real CLI's
        ``result`` is the final answer, a string, and is the reply whenever it
        is non-empty: the assistant turns before it may be narration around a
        tool call. The assistant text is joined only when the result carries
        no text of its own. A result with ``is_error: true`` fails this
        message.
        """
        text_parts: list[str] = []
        while True:
            event = await self._event_queue.get()
            if event is None:
                raise RuntimeError("CCWorker process died during message")

            event_type = event.get("type", "")

            if event_type == "assistant":
                text_parts.extend(_text_blocks(event.get("message")))

            elif event_type == "result":
                result = event.get("result")
                if event.get("is_error") is True:
                    detail = result if isinstance(result, str) else event.get("subtype", "")
                    raise CCWorkerReplyError(f"CLI reported an error result: {str(detail)[:500]}")
                if isinstance(result, str):
                    if result:
                        return result
                else:
                    text_parts.extend(_text_blocks(result))
                break

        return "".join(text_parts)

    async def shutdown(self) -> None:
        # Both handles are dropped before the first await, so the worker reads
        # dead from here on even when the reap itself is interrupted.
        reader, self._reader_task = self._reader_task, None
        proc, self._proc = self._proc, None
        if reader is not None:
            reader.cancel()
        if proc is not None:
            try:
                await platform_compat.kill_and_reap(proc)
            except Exception:
                logger.debug("CCWorker shutdown error", exc_info=True)

    def is_alive(self) -> bool:
        # A reader that ended leaves nothing to deliver a reply, even while the
        # process itself is still running.
        reader = self._reader_task
        return (
            self._proc is not None
            and self._proc.returncode is None
            and (reader is None or not reader.done())
        )

    async def reset_conversation(self) -> None:
        """Respawn the CLI subprocess, discarding the accumulated transcript.

        The transcript lives only in the process's own stream-json conversation,
        so replacing the process is the reset. ``_spawn()`` cancels the old
        reader task and installs a fresh event queue, leaving the worker ready.
        """
        old = self._proc
        await self._spawn()
        if old is not None and old is not self._proc:
            try:
                await platform_compat.kill_and_reap(old)
            except Exception:
                logger.debug("CCWorker: stale process reap failed", exc_info=True)
        self.calls_since_reset = 0
