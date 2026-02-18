"""Subprocess entry point for vLLM model serving.

This module runs in a child process spawned by the manager. It loads a vLLM
model and handles generate/embed/shutdown commands over a multiprocessing
Connection (pipe). Log lines are forwarded to the parent via a Queue.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from multiprocessing.connection import Connection
from multiprocessing import Queue
from typing import Any


class _QueueLogHandler(logging.Handler):
    """Sends formatted log lines to the parent process via a Queue."""

    def __init__(self, log_queue: Queue) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            self._queue.put_nowait(line)
        except Exception:
            pass


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

    log_handler = _QueueLogHandler(log_queue)
    log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.addHandler(log_handler)
    if vllm_logger.level == logging.NOTSET or vllm_logger.level > logging.DEBUG:
        vllm_logger.setLevel(logging.DEBUG)

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

        for prompt in prompts:
            outputs = llm.generate([prompt], sampling_params)
            output = outputs[0]
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

        for text in texts:
            outputs = llm.embed([text])
            result = list(outputs[0].outputs.embedding)
            cmd_conn.send(("stream_item", result))

        cmd_conn.send(("stream_end",))
    except Exception:
        cmd_conn.send(("error", traceback.format_exc()))
