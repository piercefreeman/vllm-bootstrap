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

        prompts = list(request.prompts)
        if not prompts:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("prompts must not be empty")
            return inference_pb2.CompleteResponse()

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        kwargs: dict = {}
        if request.max_tokens > 0:
            kwargs["max_tokens"] = request.max_tokens
        if request.HasField("temperature"):
            kwargs["temperature"] = request.temperature
        if request.HasField("top_p"):
            kwargs["top_p"] = request.top_p

        if request.HasField("guided_decoding"):
            gd = request.guided_decoding
            gd_kwargs: dict = {}
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
                kwargs["structured_outputs"] = StructuredOutputsParams(**gd_kwargs)

        sampling_params = SamplingParams(**kwargs)

        try:
            outputs = llm.generate(prompts, sampling_params)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Generation failed: {exc}")
            return inference_pb2.CompleteResponse()

        completions = []
        for output in outputs:
            completions.append(
                inference_pb2.Completion(
                    text=output.outputs[0].text,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(output.outputs[0].token_ids),
                )
            )

        return inference_pb2.CompleteResponse(completions=completions)


def create_grpc_server(manager: VLLMEnvironmentManager, port: int) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = InferenceServicer(manager)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    return server
