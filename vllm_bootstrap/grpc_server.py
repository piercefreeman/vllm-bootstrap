from __future__ import annotations

import logging
from concurrent import futures
from typing import Any

import grpc

from .auth import AccessKeyGetter, build_access_key_interceptor
from .generated import inference_pb2, inference_pb2_grpc
from .manager import (
    LaunchConflictError,
    LaunchNotFoundError,
    VLLMEnvironmentManager,
)

logger = logging.getLogger(__name__)


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, manager: VLLMEnvironmentManager) -> None:
        self._manager = manager

    def Embed(
        self, request: inference_pb2.EmbedRequest, context: grpc.ServicerContext
    ) -> inference_pb2.EmbedResponse:
        texts = list(request.texts)
        if not texts:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("texts must not be empty")
            return inference_pb2.EmbedResponse()

        try:
            results = self._manager.embed(request.launch_id, texts)
        except LaunchNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return inference_pb2.EmbedResponse()
        except LaunchConflictError as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return inference_pb2.EmbedResponse()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding failed: {exc}")
            return inference_pb2.EmbedResponse()

        embeddings = []
        for data in results:
            embeddings.append(inference_pb2.Embedding(values=data))

        return inference_pb2.EmbedResponse(embeddings=embeddings)

    def Complete(
        self, request: inference_pb2.CompleteRequest, context: grpc.ServicerContext
    ) -> inference_pb2.CompleteResponse:
        prompts = list(request.prompts)
        if not prompts:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("prompts must not be empty")
            return inference_pb2.CompleteResponse()

        params_dict: dict[str, Any] = {}
        if request.max_tokens > 0:
            params_dict["max_tokens"] = request.max_tokens
        if request.HasField("temperature"):
            params_dict["temperature"] = request.temperature
        if request.HasField("top_p"):
            params_dict["top_p"] = request.top_p

        if request.HasField("guided_decoding"):
            gd = request.guided_decoding
            gd_kwargs: dict[str, Any] = {}
            if gd.choice:
                gd_kwargs["choice"] = list(gd.choice)
            else:
                oneof_field = gd.WhichOneof("kind")
                if oneof_field == "json_schema":
                    gd_kwargs["json"] = gd.json_schema
                elif oneof_field == "regex":
                    gd_kwargs["regex"] = gd.regex
                elif oneof_field == "grammar":
                    gd_kwargs["grammar"] = gd.grammar
            if gd_kwargs:
                params_dict["structured_outputs"] = gd_kwargs

        try:
            results = self._manager.generate(request.launch_id, prompts, params_dict)
        except LaunchNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return inference_pb2.CompleteResponse()
        except LaunchConflictError as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return inference_pb2.CompleteResponse()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Generation failed: {exc}")
            return inference_pb2.CompleteResponse()

        completions = []
        for result in results:
            completions.append(
                inference_pb2.Completion(
                    text=result["text"],
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                )
            )

        return inference_pb2.CompleteResponse(completions=completions)

    def EmbedStream(
        self, request: inference_pb2.EmbedRequest, context: grpc.ServicerContext
    ):
        texts = list(request.texts)
        if not texts:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "texts must not be empty")
            return

        try:
            for data in self._manager.embed_stream(request.launch_id, texts):
                yield inference_pb2.Embedding(values=data)
        except LaunchNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except LaunchConflictError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, f"Embedding stream failed: {exc}")

    def CompleteStream(
        self, request: inference_pb2.CompleteRequest, context: grpc.ServicerContext
    ):
        prompts = list(request.prompts)
        if not prompts:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "prompts must not be empty")
            return

        params_dict: dict[str, Any] = {}
        if request.max_tokens > 0:
            params_dict["max_tokens"] = request.max_tokens
        if request.HasField("temperature"):
            params_dict["temperature"] = request.temperature
        if request.HasField("top_p"):
            params_dict["top_p"] = request.top_p

        if request.HasField("guided_decoding"):
            gd = request.guided_decoding
            gd_kwargs: dict[str, Any] = {}
            if gd.choice:
                gd_kwargs["choice"] = list(gd.choice)
            else:
                oneof_field = gd.WhichOneof("kind")
                if oneof_field == "json_schema":
                    gd_kwargs["json"] = gd.json_schema
                elif oneof_field == "regex":
                    gd_kwargs["regex"] = gd.regex
                elif oneof_field == "grammar":
                    gd_kwargs["grammar"] = gd.grammar
            if gd_kwargs:
                params_dict["structured_outputs"] = gd_kwargs

        try:
            for result in self._manager.generate_stream(
                request.launch_id, prompts, params_dict
            ):
                yield inference_pb2.Completion(
                    text=result["text"],
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                )
        except LaunchNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except LaunchConflictError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, f"Generation stream failed: {exc}")


def create_grpc_server(
    manager: VLLMEnvironmentManager,
    port: int,
    *,
    access_key_getter: AccessKeyGetter | None = None,
) -> grpc.Server:
    interceptors = []
    if access_key_getter is not None:
        interceptors.append(
            build_access_key_interceptor(access_key_getter=access_key_getter)
        )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=interceptors,
    )
    servicer = InferenceServicer(manager)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    return server
