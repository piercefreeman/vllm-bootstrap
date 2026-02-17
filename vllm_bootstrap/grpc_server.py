from __future__ import annotations

import logging
from concurrent import futures

import grpc

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
        try:
            llm, task = self._manager.get_llm(request.launch_id)
        except LaunchNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return inference_pb2.EmbedResponse()
        except LaunchConflictError as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return inference_pb2.EmbedResponse()

        if task != "embed":
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                f"Launch {request.launch_id} has task '{task}', expected 'embed'"
            )
            return inference_pb2.EmbedResponse()

        texts = list(request.texts)
        if not texts:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("texts must not be empty")
            return inference_pb2.EmbedResponse()

        try:
            outputs = llm.embed(texts)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding failed: {exc}")
            return inference_pb2.EmbedResponse()

        embeddings = []
        for output in outputs:
            data = output.outputs.embedding
            embeddings.append(inference_pb2.Embedding(values=data))

        return inference_pb2.EmbedResponse(embeddings=embeddings)

    def Complete(
        self, request: inference_pb2.CompleteRequest, context: grpc.ServicerContext
    ) -> inference_pb2.CompleteResponse:
        try:
            llm, task = self._manager.get_llm(request.launch_id)
        except LaunchNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return inference_pb2.CompleteResponse()
        except LaunchConflictError as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return inference_pb2.CompleteResponse()

        if task != "generate":
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                f"Launch {request.launch_id} has task '{task}', expected 'generate'"
            )
            return inference_pb2.CompleteResponse()

        from vllm import SamplingParams

        kwargs: dict = {}
        if request.max_tokens > 0:
            kwargs["max_tokens"] = request.max_tokens
        if request.HasField("temperature"):
            kwargs["temperature"] = request.temperature
        if request.HasField("top_p"):
            kwargs["top_p"] = request.top_p

        sampling_params = SamplingParams(**kwargs)

        try:
            outputs = llm.generate([request.prompt], sampling_params)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Generation failed: {exc}")
            return inference_pb2.CompleteResponse()

        output = outputs[0]
        generated_text = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)

        return inference_pb2.CompleteResponse(
            text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def create_grpc_server(manager: VLLMEnvironmentManager, port: int) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = InferenceServicer(manager)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    return server
