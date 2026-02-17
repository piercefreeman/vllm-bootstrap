from __future__ import annotations

import csv
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Sequence

from .config import Settings

_PROCESS_LOG_CHUNK_BYTES = 8192


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
    port: int
    state: LaunchState
    created_at: float
    updated_at: float
    return_code: int | None
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
    port: int
    state: LaunchState
    created_at: float
    updated_at: float
    process: subprocess.Popen[bytes]
    log_path: Path
    return_code: int | None = None
    error: str | None = None
    ready_scan_offset: int = 0
    ready_scan_tail: str = ""


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


class VLLMEnvironmentManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._settings.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._system_output_lock = threading.Lock()
        self._launches: dict[str, _LaunchRecord] = {}
        self._gpu_owners: dict[int, str] = {}
        self._max_ready_marker_len = max(
            len(marker) for marker in self._settings.ready_markers
        )

    def launch(
        self,
        *,
        model: str,
        gpu_ids: Sequence[int] | None,
        port: int | None,
        extra_args: Sequence[str],
    ) -> LaunchSnapshot:
        selected_model = model.strip()
        if not selected_model:
            raise LaunchValidationError("model must be provided")

        with self._lock:
            available_gpu_ids = self._discover_gpu_ids()
            selected_gpu_ids = self._resolve_gpu_ids(available_gpu_ids, gpu_ids)
            self._assert_gpu_availability(selected_gpu_ids)
            selected_port = self._select_port(port)
            launch_id = str(uuid.uuid4())

            log_path = self._settings.log_dir / f"{launch_id}.log"
            command = self._build_command(
                model=selected_model,
                port=selected_port,
                tensor_parallel_size=len(selected_gpu_ids),
                extra_args=extra_args,
            )
            log_path.touch(exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu_id) for gpu_id in selected_gpu_ids
            )

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
            if process.stdout is None:
                raise LaunchManagerError(
                    "Failed to capture vLLM process output stream."
                )
            threading.Thread(
                target=self._forward_process_output,
                args=(process.stdout, log_path),
                daemon=True,
                name=f"vllm-log-forwarder-{launch_id}",
            ).start()

            created_at = time.time()
            record = _LaunchRecord(
                launch_id=launch_id,
                model=selected_model,
                gpu_ids=selected_gpu_ids,
                port=selected_port,
                state=LaunchState.BOOTSTRAPPING,
                created_at=created_at,
                updated_at=created_at,
                process=process,
                log_path=log_path,
            )
            self._launches[launch_id] = record
            for gpu_id in selected_gpu_ids:
                self._gpu_owners[gpu_id] = launch_id

            self._refresh_record(record)
            return self._snapshot(record)

    def get_status(self, launch_id: str) -> LaunchSnapshot:
        with self._lock:
            record = self._require_record(launch_id)
            self._refresh_record(record)
            return self._snapshot(record)

    def list_launches(self, *, include_terminal: bool = False) -> list[LaunchSnapshot]:
        with self._lock:
            snapshots: list[LaunchSnapshot] = []
            records = sorted(
                self._launches.values(), key=lambda record: record.created_at
            )
            for record in records:
                self._refresh_record(record)
                if not include_terminal and record.state in TERMINAL_STATES:
                    continue
                snapshots.append(self._snapshot(record))
            return snapshots

    def read_logs(self, launch_id: str, offset: int) -> LogSnapshot:
        if offset < 0:
            raise LaunchValidationError("offset must be >= 0")

        with self._lock:
            record = self._require_record(launch_id)
            self._refresh_record(record)
            log_path = record.log_path

        if not log_path.exists():
            return LogSnapshot(
                launch_id=launch_id, offset=offset, next_offset=offset, content=""
            )

        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            file_size = log_file.tell()
            effective_offset = min(offset, file_size)
            log_file.seek(effective_offset)
            content = log_file.read(self._settings.log_read_chunk_bytes)
            next_offset = log_file.tell()

        return LogSnapshot(
            launch_id=launch_id,
            offset=effective_offset,
            next_offset=next_offset,
            content=content.decode("utf-8", errors="replace"),
        )

    def stop(self, launch_id: str) -> LaunchSnapshot:
        with self._lock:
            record = self._require_record(launch_id)
            self._refresh_record(record)
            if record.state in TERMINAL_STATES:
                return self._snapshot(record)

            record.state = LaunchState.STOPPING
            record.updated_at = time.time()
            process = record.process

        self._terminate_process(process)

        with self._lock:
            record = self._require_record(launch_id)
            self._refresh_record(record)
            if record.state not in TERMINAL_STATES:
                record.state = LaunchState.STOPPED
                record.return_code = process.poll()
                record.updated_at = time.time()
                self._release_gpus(record)
            return self._snapshot(record)

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

    def _build_command(
        self,
        *,
        model: str,
        port: int,
        tensor_parallel_size: int,
        extra_args: Sequence[str],
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            self._settings.launch_host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ]
        command.extend(extra_args)
        return command

    def _select_port(self, requested_port: int | None) -> int:
        active_ports = {
            record.port
            for record in self._launches.values()
            if record.state not in TERMINAL_STATES and record.process.poll() is None
        }

        if requested_port is not None:
            if requested_port in active_ports or not self._is_bindable_port(
                requested_port
            ):
                raise LaunchConflictError(f"Port {requested_port} is not available")
            return requested_port

        for port in range(
            self._settings.launch_port_start, self._settings.launch_port_end + 1
        ):
            if port in active_ports:
                continue
            if self._is_bindable_port(port):
                return port

        raise LaunchConflictError(
            f"No available ports in range {self._settings.launch_port_start}-{self._settings.launch_port_end}"
        )

    def _is_bindable_port(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((self._settings.launch_host, port))
                return True
            except OSError:
                return False

    def _refresh_record(self, record: _LaunchRecord) -> None:
        return_code = record.process.poll()
        if return_code is not None:
            record.return_code = return_code
            if record.state == LaunchState.STOPPING:
                record.state = LaunchState.STOPPED
            elif record.state != LaunchState.STOPPED:
                record.state = LaunchState.FAILED
                if record.error is None:
                    record.error = f"vLLM exited with code {return_code}"
            record.updated_at = time.time()
            self._release_gpus(record)
            return

        if record.state == LaunchState.BOOTSTRAPPING and self._has_ready_marker(record):
            record.state = LaunchState.READY
            record.updated_at = time.time()

    def _forward_process_output(self, source: IO[bytes], log_path: Path) -> None:
        read_chunk = getattr(source, "read1", source.read)
        try:
            with log_path.open("ab", buffering=0) as log_file:
                while True:
                    chunk = read_chunk(_PROCESS_LOG_CHUNK_BYTES)
                    if not chunk:
                        break
                    log_file.write(chunk)
                    self._write_to_system_output(chunk)
        finally:
            source.close()

    def _write_to_system_output(self, chunk: bytes) -> None:
        with self._system_output_lock:
            stdout_buffer = getattr(sys.stdout, "buffer", None)
            if stdout_buffer is not None:
                try:
                    stdout_buffer.write(chunk)
                    stdout_buffer.flush()
                    return
                except (BrokenPipeError, OSError, ValueError):
                    return

            try:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            except (BrokenPipeError, OSError, ValueError):
                return

    def _has_ready_marker(self, record: _LaunchRecord) -> bool:
        if not record.log_path.exists():
            return False

        with record.log_path.open("rb") as log_file:
            log_file.seek(record.ready_scan_offset)
            chunk = log_file.read(self._settings.ready_scan_chunk_bytes)
            record.ready_scan_offset = log_file.tell()

        if not chunk:
            return False

        text = chunk.decode("utf-8", errors="replace")
        combined = record.ready_scan_tail + text
        marker_found = any(
            marker in combined for marker in self._settings.ready_markers
        )

        tail_len = max(0, self._max_ready_marker_len - 1)
        if tail_len:
            record.ready_scan_tail = combined[-tail_len:]
        else:
            record.ready_scan_tail = ""

        return marker_found

    def _terminate_process(self, process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            process.wait(timeout=self._settings.stop_timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

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

            self._refresh_record(owner_record)
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
            port=record.port,
            state=record.state,
            created_at=record.created_at,
            updated_at=record.updated_at,
            return_code=record.return_code,
            error=record.error,
        )
