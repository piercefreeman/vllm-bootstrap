from __future__ import annotations

import sys
import time
import types
from concurrent import futures
from unittest.mock import MagicMock

import grpc
import pytest

from vllm_bootstrap.config import Settings
from vllm_bootstrap.generated import inference_pb2, inference_pb2_grpc
from vllm_bootstrap.grpc_server import InferenceServicer
from vllm_bootstrap.manager import (
    LaunchState,
    VLLMEnvironmentManager,
    _LaunchRecord,
)


def _make_manager(tmp_path) -> VLLMEnvironmentManager:
    settings = Settings(
        log_dir=tmp_path / "logs",
        stop_timeout_seconds=2.0,
        log_read_chunk_bytes=64 * 1024,
        grpc_port=0,
    )
    return VLLMEnvironmentManager(settings=settings)


def _inject_ready_launch(
    manager: VLLMEnvironmentManager,
    launch_id: str,
    task: str,
    llm: MagicMock,
) -> None:
    """Directly inject a ready launch record into the manager for testing."""
    now = time.time()
    record = _LaunchRecord(
        launch_id=launch_id,
        model="test-model",
        gpu_ids=[0],
        task=task,
        state=LaunchState.READY,
        created_at=now,
        updated_at=now,
        llm=llm,
    )
    manager._launches[launch_id] = record
    manager._gpu_owners[0] = launch_id


@pytest.fixture(autouse=True)
def _fake_vllm():
    """Install a fake vllm module so gRPC servicer can import SamplingParams."""
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: MagicMock(**kwargs)
    fake_vllm.LLM = MagicMock()
    old = sys.modules.get("vllm")
    sys.modules["vllm"] = fake_vllm
    yield
    if old is None:
        sys.modules.pop("vllm", None)
    else:
        sys.modules["vllm"] = old


@pytest.fixture
def grpc_channel(tmp_path):
    """Create a gRPC server+channel pair for testing."""
    manager = _make_manager(tmp_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = InferenceServicer(manager)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel, manager
    server.stop(grace=0)
    channel.close()


def test_embed_returns_vectors(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    mock_llm = MagicMock()
    mock_output_1 = MagicMock()
    mock_output_1.outputs.embedding = [0.1, 0.2, 0.3]
    mock_output_2 = MagicMock()
    mock_output_2.outputs.embedding = [0.4, 0.5, 0.6]
    mock_llm.embed.return_value = [mock_output_1, mock_output_2]

    _inject_ready_launch(manager, "embed-1", "embed", mock_llm)

    response = stub.Embed(
        inference_pb2.EmbedRequest(launch_id="embed-1", texts=["hello", "world"])
    )
    assert len(response.embeddings) == 2
    assert list(response.embeddings[0].values) == pytest.approx([0.1, 0.2, 0.3])
    assert list(response.embeddings[1].values) == pytest.approx([0.4, 0.5, 0.6])
    mock_llm.embed.assert_called_once_with(["hello", "world"])


def test_embed_rejects_generate_task(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    mock_llm = MagicMock()
    _inject_ready_launch(manager, "gen-1", "generate", mock_llm)

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Embed(inference_pb2.EmbedRequest(launch_id="gen-1", texts=["hello"]))
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "expected 'embed'" in exc_info.value.details()


def test_embed_returns_not_found(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Embed(inference_pb2.EmbedRequest(launch_id="missing", texts=["hello"]))
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


def test_complete_returns_text(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    mock_llm = MagicMock()
    mock_completion = MagicMock()
    mock_completion.outputs[0].text = "Hello world!"
    mock_completion.prompt_token_ids = [1, 2, 3]
    mock_completion.outputs[0].token_ids = [4, 5]
    mock_llm.generate.return_value = [mock_completion]

    _inject_ready_launch(manager, "gen-1", "generate", mock_llm)

    response = stub.Complete(
        inference_pb2.CompleteRequest(
            launch_id="gen-1",
            prompt="Say hello",
            max_tokens=100,
        )
    )
    assert response.text == "Hello world!"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2


def test_complete_rejects_embed_task(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    mock_llm = MagicMock()
    _inject_ready_launch(manager, "embed-1", "embed", mock_llm)

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="embed-1",
                prompt="Say hello",
                max_tokens=100,
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "expected 'generate'" in exc_info.value.details()


def test_complete_returns_not_found(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="missing",
                prompt="hello",
                max_tokens=10,
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
