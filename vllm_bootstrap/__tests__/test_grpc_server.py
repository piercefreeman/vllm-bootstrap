from __future__ import annotations

import time
from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import pytest

from vllm_bootstrap.auth import build_access_key_interceptor
from vllm_bootstrap.config import Settings
from vllm_bootstrap.generated import inference_pb2, inference_pb2_grpc
from vllm_bootstrap.grpc_server import InferenceServicer
from vllm_bootstrap.manager import (
    LaunchConflictError,
    LaunchNotFoundError,
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
    )
    manager._launches[launch_id] = record
    manager._gpu_owners[0] = launch_id


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

    _inject_ready_launch(manager, "embed-1", "embed")

    with patch.object(
        manager, "embed", return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ) as mock_embed:
        response = stub.Embed(
            inference_pb2.EmbedRequest(launch_id="embed-1", texts=["hello", "world"])
        )
        assert len(response.embeddings) == 2
        assert list(response.embeddings[0].values) == pytest.approx([0.1, 0.2, 0.3])
        assert list(response.embeddings[1].values) == pytest.approx([0.4, 0.5, 0.6])
        mock_embed.assert_called_once_with("embed-1", ["hello", "world"])


def test_embed_rejects_generate_task(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "gen-1", "generate")

    with patch.object(
        manager,
        "embed",
        side_effect=LaunchConflictError(
            "Launch gen-1 has task 'generate', expected 'embed'"
        ),
    ):
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

    _inject_ready_launch(manager, "gen-1", "generate")

    with patch.object(
        manager,
        "generate",
        return_value=[
            {"text": "Hello world!", "prompt_tokens": 3, "completion_tokens": 2}
        ],
    ) as mock_generate:
        response = stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="gen-1",
                prompts=["Say hello"],
                max_tokens=100,
            )
        )
        assert len(response.completions) == 1
        assert response.completions[0].text == "Hello world!"
        assert response.completions[0].prompt_tokens == 3
        assert response.completions[0].completion_tokens == 2


def test_complete_batch(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "gen-1", "generate")

    with patch.object(
        manager,
        "generate",
        return_value=[
            {"text": "Hello!", "prompt_tokens": 2, "completion_tokens": 2},
            {"text": "Goodbye!", "prompt_tokens": 3, "completion_tokens": 1},
        ],
    ) as mock_generate:
        response = stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="gen-1",
                prompts=["Say hello", "Say goodbye"],
                max_tokens=100,
            )
        )
        assert len(response.completions) == 2
        assert response.completions[0].text == "Hello!"
        assert response.completions[0].prompt_tokens == 2
        assert response.completions[0].completion_tokens == 2
        assert response.completions[1].text == "Goodbye!"
        assert response.completions[1].prompt_tokens == 3
        assert response.completions[1].completion_tokens == 1

        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        assert call_args[0][1] == ["Say hello", "Say goodbye"]


def test_complete_rejects_empty_prompts(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "gen-1", "generate")

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="gen-1",
                prompts=[],
                max_tokens=100,
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "prompts must not be empty" in exc_info.value.details()


def test_complete_rejects_embed_task(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "embed-1", "embed")

    with patch.object(
        manager,
        "generate",
        side_effect=LaunchConflictError(
            "Launch embed-1 has task 'embed', expected 'generate'"
        ),
    ):
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.Complete(
                inference_pb2.CompleteRequest(
                    launch_id="embed-1",
                    prompts=["Say hello"],
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
                prompts=["hello"],
                max_tokens=10,
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


def test_complete_with_json_schema(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "gen-1", "generate")

    with patch.object(
        manager,
        "generate",
        return_value=[
            {"text": '{"name": "Alice"}', "prompt_tokens": 2, "completion_tokens": 3}
        ],
    ) as mock_generate:
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        response = stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="gen-1",
                prompts=["Generate a person"],
                max_tokens=100,
                guided_decoding=inference_pb2.GuidedDecodingParams(
                    json_schema=schema,
                ),
            )
        )
        assert len(response.completions) == 1
        assert response.completions[0].text == '{"name": "Alice"}'

        call_args = mock_generate.call_args
        params_dict = call_args[0][2]
        assert "structured_outputs" in params_dict
        assert params_dict["structured_outputs"]["json"] == schema


def test_complete_with_regex(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "gen-1", "generate")

    with patch.object(
        manager,
        "generate",
        return_value=[{"text": "42", "prompt_tokens": 1, "completion_tokens": 2}],
    ) as mock_generate:
        response = stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="gen-1",
                prompts=["Give me a number"],
                max_tokens=10,
                guided_decoding=inference_pb2.GuidedDecodingParams(
                    regex=r"\d+",
                ),
            )
        )
        assert len(response.completions) == 1
        assert response.completions[0].text == "42"

        call_args = mock_generate.call_args
        params_dict = call_args[0][2]
        assert "structured_outputs" in params_dict
        assert params_dict["structured_outputs"]["regex"] == r"\d+"


def test_complete_with_choice(grpc_channel) -> None:
    channel, manager = grpc_channel
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "gen-1", "generate")

    with patch.object(
        manager,
        "generate",
        return_value=[{"text": "yes", "prompt_tokens": 2, "completion_tokens": 1}],
    ) as mock_generate:
        response = stub.Complete(
            inference_pb2.CompleteRequest(
                launch_id="gen-1",
                prompts=["Is this good?"],
                max_tokens=10,
                guided_decoding=inference_pb2.GuidedDecodingParams(
                    choice=["yes", "no"],
                ),
            )
        )
        assert len(response.completions) == 1
        assert response.completions[0].text == "yes"

        call_args = mock_generate.call_args
        params_dict = call_args[0][2]
        assert "structured_outputs" in params_dict
        assert params_dict["structured_outputs"]["choice"] == ["yes", "no"]


# -- Auth interceptor tests --

TEST_ACCESS_KEY = "test-secret-key-12345"


@pytest.fixture
def grpc_channel_with_auth(tmp_path):
    """Create a gRPC server+channel pair with auth interceptor enabled."""
    manager = _make_manager(tmp_path)
    interceptor = build_access_key_interceptor(
        access_key_getter=lambda: TEST_ACCESS_KEY,
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[interceptor],
    )
    servicer = InferenceServicer(manager)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel, manager
    server.stop(grace=0)
    channel.close()


def test_auth_rejects_missing_metadata(grpc_channel_with_auth) -> None:
    channel, manager = grpc_channel_with_auth
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "embed-1", "embed")

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Embed(inference_pb2.EmbedRequest(launch_id="embed-1", texts=["hello"]))
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_auth_rejects_wrong_key(grpc_channel_with_auth) -> None:
    channel, manager = grpc_channel_with_auth
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "embed-1", "embed")

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.Embed(
            inference_pb2.EmbedRequest(launch_id="embed-1", texts=["hello"]),
            metadata=[("authorization", "Bearer wrong-key")],
        )
    assert exc_info.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_auth_accepts_valid_bearer_token(grpc_channel_with_auth) -> None:
    channel, manager = grpc_channel_with_auth
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "embed-1", "embed")

    with patch.object(manager, "embed", return_value=[[0.1, 0.2]]):
        response = stub.Embed(
            inference_pb2.EmbedRequest(launch_id="embed-1", texts=["hello"]),
            metadata=[("authorization", f"Bearer {TEST_ACCESS_KEY}")],
        )
        assert len(response.embeddings) == 1
        assert list(response.embeddings[0].values) == pytest.approx([0.1, 0.2])


def test_auth_accepts_valid_basic_auth(grpc_channel_with_auth) -> None:
    import base64

    channel, manager = grpc_channel_with_auth
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "embed-1", "embed")

    basic_cred = base64.b64encode(f"user:{TEST_ACCESS_KEY}".encode()).decode()

    with patch.object(manager, "embed", return_value=[[0.3, 0.4]]):
        response = stub.Embed(
            inference_pb2.EmbedRequest(launch_id="embed-1", texts=["hello"]),
            metadata=[("authorization", f"Basic {basic_cred}")],
        )
        assert len(response.embeddings) == 1
        assert list(response.embeddings[0].values) == pytest.approx([0.3, 0.4])


@pytest.fixture
def grpc_channel_no_auth(tmp_path):
    """Create a gRPC server+channel with auth interceptor but no key configured."""
    manager = _make_manager(tmp_path)
    interceptor = build_access_key_interceptor(
        access_key_getter=lambda: None,
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[interceptor],
    )
    servicer = InferenceServicer(manager)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel, manager
    server.stop(grace=0)
    channel.close()


def test_auth_passes_when_no_key_configured(grpc_channel_no_auth) -> None:
    channel, manager = grpc_channel_no_auth
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    _inject_ready_launch(manager, "embed-1", "embed")

    with patch.object(manager, "embed", return_value=[[0.5, 0.6]]):
        # No metadata at all — should pass through since no key is configured
        response = stub.Embed(
            inference_pb2.EmbedRequest(launch_id="embed-1", texts=["hello"]),
        )
        assert len(response.embeddings) == 1
        assert list(response.embeddings[0].values) == pytest.approx([0.5, 0.6])
