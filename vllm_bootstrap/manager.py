from __future__ import annotations

import csv
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Callable, Sequence

from .config import Settings

logger = logging.getLogger(__name__)


class LaunchState(str, Enum):
    BOOTSTRAPPING = "bootstrapping"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


TERMINAL_STATES = {LaunchState.STOPPED, LaunchState.FAILED}


@dataclass(slots=True)
class LaunchSnapshot:
    launch_id: str
    model: str
    gpu_ids: list[int]
    task: str
    state: LaunchState
    created_at: float
    updated_at: float
    error: str | None


@dataclass(slots=True)
class LogSnapshot:
    launch_id: str
    offset: int
    next_offset: int
    content: str


@dataclass(slots=True)
class GPUStatsSnapshot:
    gpu_id: int
    uuid: str | None
    name: str
    utilization_percent: float | None
    memory_total_mib: int | None
    memory_used_mib: int | None
    memory_free_mib: int | None
    temperature_c: int | None
    power_draw_watts: float | None
    power_limit_watts: float | None


@dataclass(slots=True)
class SystemStatsSnapshot:
    collected_at: float
    load_avg_1m: float | None
    load_avg_5m: float | None
    load_avg_15m: float | None
    cpu_count: int | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    memory_used_bytes: int | None
    memory_utilization_percent: float | None
    host_memory_error: str | None
    gpu_count: int
    gpus: list[GPUStatsSnapshot]
    nvidia_smi_error: str | None


@dataclass(slots=True)
class _LaunchRecord:
    launch_id: str
    model: str
    gpu_ids: list[int]
    task: str
    state: LaunchState
    created_at: float
    updated_at: float
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None
    _process: multiprocessing.Process | None = None
    _cmd_conn: Connection | None = None
    _log_queue: multiprocessing.Queue | None = None
    _pipe_lock: threading.Lock | None = None
    _log_thread: threading.Thread | None = None
    _monitor_thread: threading.Thread | None = None


class LaunchManagerError(Exception):
    pass


class LaunchNotFoundError(LaunchManagerError):
    pass


class LaunchConflictError(LaunchManagerError):
    pass


class LaunchValidationError(LaunchManagerError):
    pass


_NVIDIA_SMI_GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)
_NVIDIA_SMI_GPU_QUERY_COMMAND = [
    "nvidia-smi",
    f"--query-gpu={','.join(_NVIDIA_SMI_GPU_QUERY_FIELDS)}",
    "--format=csv,noheader,nounits",
]
_NA_TOKENS = {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}


def _worker_entry(
    cmd_conn: Connection,
    log_queue: multiprocessing.Queue,
    gpu_ids: list[int],
    model: str,
    task: str,
    extra_kwargs: dict[str, Any],
) -> None:
    """Default subprocess entry point. Lazily imports worker_main."""
    from .worker import worker_main

    worker_main(cmd_conn, log_queue, gpu_ids, model, task, extra_kwargs)


class VLLMEnvironmentManager:
    def __init__(
        self,
        settings: Settings,
        *,
        _worker_fn: Callable[..., None] | None = None,
    ) -> None:
        self._settings = settings
        self._settings.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._launches: dict[str, _LaunchRecord] = {}
        self._gpu_owners: dict[int, str] = {}
        self._worker_fn = _worker_fn or _worker_entry

    def launch(
        self,
        *,
        model: str,
        gpu_ids: Sequence[int] | None,
        task: str = "generate",
        extra_kwargs: dict[str, Any] | None = None,
    ) -> LaunchSnapshot:
        selected_model = model.strip()
        if not selected_model:
            raise LaunchValidationError("model must be provided")

        if task not in ("generate", "embed"):
            raise LaunchValidationError("task must be 'generate' or 'embed'")

        with self._lock:
            available_gpu_ids = self._discover_gpu_ids()
            selected_gpu_ids = self._resolve_gpu_ids(available_gpu_ids, gpu_ids)
            self._assert_gpu_availability(selected_gpu_ids)
            launch_id = str(uuid.uuid4())

            created_at = time.time()
            record = _LaunchRecord(
                launch_id=launch_id,
                model=selected_model,
                gpu_ids=selected_gpu_ids,
                task=task,
                state=LaunchState.BOOTSTRAPPING,
                created_at=created_at,
                updated_at=created_at,
            )
            record._pipe_lock = threading.Lock()
            self._launches[launch_id] = record
            for gpu_id in selected_gpu_ids:
                self._gpu_owners[gpu_id] = launch_id

            monitor_thread = threading.Thread(
                target=self._launch_subprocess,
                args=(record, extra_kwargs or {}),
                daemon=True,
                name=f"vllm-monitor-{launch_id}",
            )
            record._monitor_thread = monitor_thread
            monitor_thread.start()

            return self._snapshot(record)

    def _launch_subprocess(
        self, record: _LaunchRecord, extra_kwargs: dict[str, Any]
    ) -> None:
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        log_queue: multiprocessing.Queue = ctx.Queue()

        record._cmd_conn = parent_conn
        record._log_queue = log_queue

        process = ctx.Process(
            target=self._worker_fn,
            args=(
                child_conn,
                log_queue,
                record.gpu_ids,
                record.model,
                record.task,
                extra_kwargs,
            ),
            daemon=False,
        )
        record._process = process
        process.start()
        child_conn.close()

        # Start log drainer
        log_thread = threading.Thread(
            target=self._drain_logs,
            args=(record,),
            daemon=True,
            name=f"vllm-logdrain-{record.launch_id}",
        )
        record._log_thread = log_thread
        log_thread.start()

        # Wait for ready/error from subprocess
        try:
            msg = parent_conn.recv()
        except (EOFError, OSError):
            with self._lock:
                record.error = "Subprocess exited before signaling ready"
                record.state = LaunchState.FAILED
                record.updated_at = time.time()
                self._release_gpus(record)
            return

        if msg[0] == "ready":
            with self._lock:
                if record.state == LaunchState.BOOTSTRAPPING:
                    record.state = LaunchState.READY
                    record.updated_at = time.time()
        elif msg[0] == "error":
            with self._lock:
                record.error = msg[1]
                record.state = LaunchState.FAILED
                record.updated_at = time.time()
                self._release_gpus(record)

    def _drain_logs(self, record: _LaunchRecord) -> None:
        log_queue = record._log_queue
        if log_queue is None:
            return
        while True:
            try:
                line = log_queue.get(timeout=0.5)
            except Exception:
                # Check if process is still alive
                proc = record._process
                if proc is not None and not proc.is_alive():
                    # Drain remaining items
                    while True:
                        try:
                            line = log_queue.get_nowait()
                            record.log_lines.append(line)
                            try:
                                sys.stdout.write(line + "\n")
                                sys.stdout.flush()
                            except (BrokenPipeError, OSError, ValueError):
                                pass
                        except Exception:
                            break
                    break
                continue
            record.log_lines.append(line)
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def generate(
        self, launch_id: str, prompts: list[str], params_dict: dict[str, Any]
    ) -> list[dict]:
        with self._lock:
            record = self._require_record(launch_id)
            if record.state != LaunchState.READY:
                raise LaunchConflictError(
                    f"Launch {launch_id} is not ready (state={record.state.value})"
                )
            if record.task != "generate":
                raise LaunchConflictError(
                    f"Launch {launch_id} has task '{record.task}', expected 'generate'"
                )

        return self._send_command(record, ("generate", prompts, params_dict))

    def embed(self, launch_id: str, texts: list[str]) -> list[list[float]]:
        with self._lock:
            record = self._require_record(launch_id)
            if record.state != LaunchState.READY:
                raise LaunchConflictError(
                    f"Launch {launch_id} is not ready (state={record.state.value})"
                )
            if record.task != "embed":
                raise LaunchConflictError(
                    f"Launch {launch_id} has task '{record.task}', expected 'embed'"
                )

        return self._send_command(record, ("embed", texts))

    def _send_command(self, record: _LaunchRecord, message: tuple) -> Any:
        pipe_lock = record._pipe_lock
        cmd_conn = record._cmd_conn
        if pipe_lock is None or cmd_conn is None:
            raise LaunchConflictError(
                f"Launch {record.launch_id} subprocess is not available"
            )

        with pipe_lock:
            try:
                cmd_conn.send(message)
                response = cmd_conn.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                with self._lock:
                    record.error = f"Subprocess crashed: {exc}"
                    record.state = LaunchState.FAILED
                    record.updated_at = time.time()
                    self._release_gpus(record)
                raise LaunchConflictError(
                    f"Launch {record.launch_id} subprocess crashed"
                ) from exc

        if response[0] == "result":
            return response[1]
        elif response[0] == "error":
            raise RuntimeError(f"Subprocess error: {response[1]}")
        else:
            raise RuntimeError(f"Unexpected subprocess response: {response}")

    def get_status(self, launch_id: str) -> LaunchSnapshot:
        with self._lock:
            record = self._require_record(launch_id)
            return self._snapshot(record)

    def list_launches(self, *, include_terminal: bool = False) -> list[LaunchSnapshot]:
        with self._lock:
            snapshots: list[LaunchSnapshot] = []
            records = sorted(
                self._launches.values(), key=lambda record: record.created_at
            )
            for record in records:
                if not include_terminal and record.state in TERMINAL_STATES:
                    continue
                snapshots.append(self._snapshot(record))
            return snapshots

    def read_logs(self, launch_id: str, offset: int) -> LogSnapshot:
        if offset < 0:
            raise LaunchValidationError("offset must be >= 0")

        with self._lock:
            record = self._require_record(launch_id)
            all_content = "\n".join(record.log_lines)
            if record.log_lines:
                all_content += "\n"

        content_bytes = all_content.encode("utf-8")
        file_size = len(content_bytes)
        effective_offset = min(offset, file_size)
        chunk = content_bytes[
            effective_offset : effective_offset + self._settings.log_read_chunk_bytes
        ]
        next_offset = effective_offset + len(chunk)

        return LogSnapshot(
            launch_id=launch_id,
            offset=effective_offset,
            next_offset=next_offset,
            content=chunk.decode("utf-8", errors="replace"),
        )

    def stop(self, launch_id: str) -> LaunchSnapshot:
        with self._lock:
            record = self._require_record(launch_id)
            if record.state in TERMINAL_STATES:
                return self._snapshot(record)

            logger.info("Starting shutdown for launch_id=%s", launch_id)
            record.state = LaunchState.STOPPING
            record.updated_at = time.time()

        self._cleanup_subprocess(record)

        with self._lock:
            record = self._require_record(launch_id)
            if record.state not in TERMINAL_STATES:
                record.state = LaunchState.STOPPED
                record.updated_at = time.time()
                self._release_gpus(record)
            return self._snapshot(record)

    def _cleanup_subprocess(self, record: _LaunchRecord) -> None:
        cmd_conn = record._cmd_conn
        process = record._process
        timeout = self._settings.stop_timeout_seconds

        # Send shutdown command
        if cmd_conn is not None:
            try:
                cmd_conn.send(("shutdown",))
            except (EOFError, BrokenPipeError, OSError):
                pass

        # Wait for process to exit
        if process is not None and process.is_alive():
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

        # Close pipe
        if cmd_conn is not None:
            try:
                cmd_conn.close()
            except OSError:
                pass
            record._cmd_conn = None

        record._process = None

    def stop_all(self) -> None:
        with self._lock:
            launch_ids = [
                launch_id
                for launch_id, record in self._launches.items()
                if record.state not in TERMINAL_STATES
            ]

        for launch_id in launch_ids:
            try:
                self.stop(launch_id)
            except LaunchManagerError:
                continue

    def get_system_stats(self) -> SystemStatsSnapshot:
        collected_at = time.time()

        load_avg_1m: float | None = None
        load_avg_5m: float | None = None
        load_avg_15m: float | None = None
        try:
            load_avg_1m, load_avg_5m, load_avg_15m = os.getloadavg()
        except (AttributeError, OSError):
            pass

        cpu_count = os.cpu_count()
        (
            memory_total_bytes,
            memory_available_bytes,
            memory_used_bytes,
            memory_utilization_percent,
            host_memory_error,
        ) = self._collect_host_memory_stats()
        gpu_stats, nvidia_smi_error = self._collect_gpu_stats()

        return SystemStatsSnapshot(
            collected_at=collected_at,
            load_avg_1m=load_avg_1m,
            load_avg_5m=load_avg_5m,
            load_avg_15m=load_avg_15m,
            cpu_count=cpu_count,
            memory_total_bytes=memory_total_bytes,
            memory_available_bytes=memory_available_bytes,
            memory_used_bytes=memory_used_bytes,
            memory_utilization_percent=memory_utilization_percent,
            host_memory_error=host_memory_error,
            gpu_count=len(gpu_stats),
            gpus=gpu_stats,
            nvidia_smi_error=nvidia_smi_error,
        )

    def _resolve_gpu_ids(
        self, available_gpu_ids: Sequence[int], requested_gpu_ids: Sequence[int] | None
    ) -> list[int]:
        if not available_gpu_ids:
            raise LaunchValidationError("No GPUs available for launch")

        if requested_gpu_ids is None:
            return sorted(set(available_gpu_ids))

        normalized = sorted(set(requested_gpu_ids))
        if not normalized:
            raise LaunchValidationError("gpu_ids cannot be empty")

        missing = [gpu_id for gpu_id in normalized if gpu_id not in available_gpu_ids]
        if missing:
            raise LaunchValidationError(f"Requested GPUs are not available: {missing}")
        return normalized

    def _assert_gpu_availability(self, requested_gpu_ids: Sequence[int]) -> None:
        conflicts: list[str] = []
        for gpu_id in requested_gpu_ids:
            owner_launch_id = self._gpu_owners.get(gpu_id)
            if owner_launch_id is None:
                continue

            owner_record = self._launches.get(owner_launch_id)
            if owner_record is None:
                self._gpu_owners.pop(gpu_id, None)
                continue

            if owner_record.state in TERMINAL_STATES:
                self._gpu_owners.pop(gpu_id, None)
                continue

            conflicts.append(f"gpu {gpu_id} -> {owner_launch_id}")

        if conflicts:
            message = ", ".join(conflicts)
            raise LaunchConflictError(f"Requested GPUs are in use ({message})")

    def _release_gpus(self, record: _LaunchRecord) -> None:
        for gpu_id in record.gpu_ids:
            if self._gpu_owners.get(gpu_id) == record.launch_id:
                self._gpu_owners.pop(gpu_id, None)

    def _discover_gpu_ids(self) -> list[int]:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices and cuda_visible_devices.lower() != "all":
            return self._parse_gpu_ids(
                cuda_visible_devices, source="CUDA_VISIBLE_DEVICES"
            )

        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as error:
            raise LaunchValidationError(
                "Could not discover GPUs from connected host hardware because nvidia-smi is unavailable."
            ) from error
        except subprocess.CalledProcessError as error:
            raise LaunchValidationError(
                f"Failed to discover GPUs from nvidia-smi: {error.output.strip()}"
            ) from error

        gpu_ids = []
        for line in output.splitlines():
            token = line.strip()
            if not token:
                continue
            gpu_ids.append(int(token))

        return sorted(set(gpu_ids))

    def _parse_gpu_ids(self, raw: str, *, source: str) -> list[int]:
        gpu_ids: list[int] = []
        for token in raw.split(","):
            candidate = token.strip()
            if not candidate:
                continue
            try:
                gpu_id = int(candidate)
            except ValueError as error:
                raise LaunchValidationError(
                    f"{source} must contain comma-separated integer GPU ids, got {raw!r}"
                ) from error

            if gpu_id < 0:
                raise LaunchValidationError(
                    f"{source} GPU ids must be >= 0, got {gpu_id}"
                )
            gpu_ids.append(gpu_id)

        if not gpu_ids:
            raise LaunchValidationError(f"{source} did not contain any GPU ids")

        return sorted(set(gpu_ids))

    def _collect_gpu_stats(self) -> tuple[list[GPUStatsSnapshot], str | None]:
        try:
            output = subprocess.check_output(
                _NVIDIA_SMI_GPU_QUERY_COMMAND,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            return [], "nvidia-smi is unavailable on this host."
        except subprocess.CalledProcessError as error:
            message = error.output.strip() if error.output else str(error)
            return [], f"Failed to query GPU stats from nvidia-smi: {message}"

        gpu_stats = self._parse_gpu_stats(output)
        if not gpu_stats and output.strip():
            return [], "Could not parse GPU stats from nvidia-smi output."
        return gpu_stats, None

    def _parse_gpu_stats(self, raw: str) -> list[GPUStatsSnapshot]:
        gpu_stats: list[GPUStatsSnapshot] = []
        rows = csv.reader(raw.splitlines(), skipinitialspace=True)
        for row in rows:
            if len(row) < len(_NVIDIA_SMI_GPU_QUERY_FIELDS):
                continue

            gpu_id = self._parse_optional_int(row[0])
            if gpu_id is None:
                continue

            gpu_stats.append(
                GPUStatsSnapshot(
                    gpu_id=gpu_id,
                    uuid=row[1].strip() or None,
                    name=row[2].strip() or "unknown",
                    utilization_percent=self._parse_optional_float(row[3]),
                    memory_total_mib=self._parse_optional_int(row[4]),
                    memory_used_mib=self._parse_optional_int(row[5]),
                    memory_free_mib=self._parse_optional_int(row[6]),
                    temperature_c=self._parse_optional_int(row[7]),
                    power_draw_watts=self._parse_optional_float(row[8]),
                    power_limit_watts=self._parse_optional_float(row[9]),
                )
            )
        return sorted(gpu_stats, key=lambda snapshot: snapshot.gpu_id)

    def _collect_host_memory_stats(
        self,
    ) -> tuple[int | None, int | None, int | None, float | None, str | None]:
        memory_total_bytes: int | None = None
        memory_available_bytes: int | None = None
        memory_error: str | None = None

        if sys.platform.startswith("linux"):
            try:
                memory_total_bytes, memory_available_bytes = (
                    self._collect_host_memory_stats_linux()
                )
            except (OSError, ValueError) as error:
                memory_error = f"Failed to parse /proc/meminfo: {error}"

        if memory_total_bytes is None:
            try:
                fallback_total, fallback_available = (
                    self._collect_host_memory_stats_sysconf()
                )
                if fallback_total is not None:
                    memory_total_bytes = fallback_total
                if fallback_available is not None:
                    memory_available_bytes = fallback_available
            except (OSError, ValueError, AttributeError) as error:
                if memory_error is None:
                    memory_error = f"Failed to inspect host memory: {error}"

        memory_used_bytes: int | None = None
        memory_utilization_percent: float | None = None
        if memory_total_bytes is not None and memory_available_bytes is not None:
            memory_used_bytes = max(memory_total_bytes - memory_available_bytes, 0)
            if memory_total_bytes > 0:
                memory_utilization_percent = (
                    memory_used_bytes / memory_total_bytes
                ) * 100.0

        return (
            memory_total_bytes,
            memory_available_bytes,
            memory_used_bytes,
            memory_utilization_percent,
            memory_error,
        )

    def _collect_host_memory_stats_linux(self) -> tuple[int, int]:
        meminfo: dict[str, int] = {}
        from pathlib import Path

        with Path("/proc/meminfo").open("r", encoding="utf-8") as meminfo_file:
            for raw_line in meminfo_file:
                key, _, remainder = raw_line.partition(":")
                if not remainder:
                    continue
                number_token = remainder.strip().split(maxsplit=1)[0]
                try:
                    value_kib = int(number_token)
                except ValueError:
                    continue
                meminfo[key.strip()] = value_kib

        total_kib = meminfo.get("MemTotal")
        available_kib = meminfo.get("MemAvailable", meminfo.get("MemFree"))
        if total_kib is None or available_kib is None:
            raise ValueError("MemTotal or MemAvailable is missing")

        return total_kib * 1024, available_kib * 1024

    def _collect_host_memory_stats_sysconf(
        self,
    ) -> tuple[int | None, int | None]:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size == -1 or total_pages == -1:
            return None, None

        total_bytes = int(page_size) * int(total_pages)
        available_bytes: int | None = None

        try:
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            available_pages = -1

        if available_pages != -1:
            available_bytes = int(page_size) * int(available_pages)

        return total_bytes, available_bytes

    @staticmethod
    def _parse_optional_int(token: str) -> int | None:
        value = token.strip()
        if value in _NA_TOKENS:
            return None

        match = re.search(r"-?\d+", value)
        if match is None:
            return None
        return int(match.group())

    @staticmethod
    def _parse_optional_float(token: str) -> float | None:
        value = token.strip()
        if value in _NA_TOKENS:
            return None

        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match is None:
            return None
        return float(match.group())

    def _require_record(self, launch_id: str) -> _LaunchRecord:
        record = self._launches.get(launch_id)
        if record is None:
            raise LaunchNotFoundError(f"Unknown launch_id: {launch_id}")
        return record

    def _snapshot(self, record: _LaunchRecord) -> LaunchSnapshot:
        return LaunchSnapshot(
            launch_id=record.launch_id,
            model=record.model,
            gpu_ids=list(record.gpu_ids),
            task=record.task,
            state=record.state,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error=record.error,
        )
