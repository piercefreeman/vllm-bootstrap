"""Subprocess entry point for vLLM model serving.

This module runs in a child process spawned by the manager. It loads a vLLM
model and handles generate/embed/shutdown commands over a multiprocessing
Connection (pipe). Log lines are forwarded to the parent via a Queue.
"""

from __future__ import annotations

import os
import sys
import traceback
from multiprocessing.connection import Connection
from multiprocessing import Queue
from typing import Any


class _QueueStream:
    """File-like wrapper that sends complete lines to a multiprocessing Queue.

    Replaces sys.stdout/sys.stderr in the worker subprocess so that all output
    (logging, print, tqdm, etc.) is forwarded to the parent process.
    """

    def __init__(self, log_queue: Queue, original: Any) -> None:
        self._queue = log_queue
        self._original = original
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                try:
                    self._queue.put_nowait(line)
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            try:
                self._queue.put_nowait(self._buffer)
            except Exception:
                pass
            self._buffer = ""

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return False


def worker_main(
    cmd_conn: Connection,
    log_queue: Queue,
    gpu_ids: list[int],
    model: str,
    task: str,
    extra_kwargs: dict[str, Any],
) -> None:
    """Entry point that runs inside the child subprocess.

    1. Sets CUDA_VISIBLE_DEVICES
    2. Installs log forwarding
    3. Loads vllm.LLM
    4. Sends ("ready",) on the pipe
    5. Loops handling commands until shutdown
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    # Redirect stdout/stderr so all output (logging, print, tqdm, etc.)
    # is forwarded to the parent process via the log queue. vLLM's child
    # loggers set propagate=False and write to stdout via their own
    # StreamHandlers, so a handler on the "vllm" logger alone won't work.
    sys.stdout = _QueueStream(log_queue, sys.stdout)  # type: ignore[assignment]
    sys.stderr = _QueueStream(log_queue, sys.stderr)  # type: ignore[assignment]

    try:
        import vllm

        vllm_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": len(gpu_ids),
        }
        if task == "embed":
            vllm_kwargs["runner"] = "pooling"
            vllm_kwargs["convert"] = "embed"

        vllm_kwargs.update(extra_kwargs)
        llm = vllm.LLM(**vllm_kwargs)
    except Exception:
        cmd_conn.send(("error", traceback.format_exc()))
        return

    cmd_conn.send(("ready",))

    while True:
        try:
            msg = cmd_conn.recv()
        except (EOFError, OSError):
            break

        cmd = msg[0]
        if cmd == "shutdown":
            break
        elif cmd == "generate":
            _handle_generate(llm, cmd_conn, msg)
        elif cmd == "embed":
            _handle_embed(llm, cmd_conn, msg)
        elif cmd == "generate_stream":
            _handle_generate_stream(llm, cmd_conn, msg)
        elif cmd == "embed_stream":
            _handle_embed_stream(llm, cmd_conn, msg)
        else:
            cmd_conn.send(("error", f"Unknown command: {cmd}"))


def _handle_generate(llm: Any, cmd_conn: Connection, msg: tuple) -> None:
    try:
        _, prompts, params_dict = msg

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        kwargs: dict[str, Any] = {}
        for key in ("max_tokens", "temperature", "top_p"):
            if key in params_dict:
                kwargs[key] = params_dict[key]

        structured = params_dict.get("structured_outputs")
        if structured is not None:
            kwargs["structured_outputs"] = StructuredOutputsParams(**structured)

        sampling_params = SamplingParams(**kwargs)
        outputs = llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            results.append(
                {
                    "text": output.outputs[0].text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                }
            )
        cmd_conn.send(("result", results))
    except Exception:
        cmd_conn.send(("error", traceback.format_exc()))


def _handle_embed(llm: Any, cmd_conn: Connection, msg: tuple) -> None:
    try:
        _, texts = msg
        outputs = llm.embed(texts)
        results = [list(output.outputs.embedding) for output in outputs]
        cmd_conn.send(("result", results))
    except Exception:
        cmd_conn.send(("error", traceback.format_exc()))


def _handle_generate_stream(llm: Any, cmd_conn: Connection, msg: tuple) -> None:
    try:
        _, prompts, params_dict = msg

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        kwargs: dict[str, Any] = {}
        for key in ("max_tokens", "temperature", "top_p"):
            if key in params_dict:
                kwargs[key] = params_dict[key]

        structured = params_dict.get("structured_outputs")
        if structured is not None:
            kwargs["structured_outputs"] = StructuredOutputsParams(**structured)

        sampling_params = SamplingParams(**kwargs)

        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            result = {
                "text": output.outputs[0].text,
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
            }
            cmd_conn.send(("stream_item", result))

        cmd_conn.send(("stream_end",))
    except Exception:
        cmd_conn.send(("error", traceback.format_exc()))


def _handle_embed_stream(llm: Any, cmd_conn: Connection, msg: tuple) -> None:
    try:
        _, texts = msg

        outputs = llm.embed(texts)
        for output in outputs:
            result = list(output.outputs.embedding)
            cmd_conn.send(("stream_item", result))

        cmd_conn.send(("stream_end",))
    except Exception:
        cmd_conn.send(("error", traceback.format_exc()))
