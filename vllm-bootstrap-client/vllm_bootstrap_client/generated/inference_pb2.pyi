from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EmbedRequest(_message.Message):
    __slots__ = ("launch_id", "texts")
    LAUNCH_ID_FIELD_NUMBER: _ClassVar[int]
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    launch_id: str
    texts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self, launch_id: _Optional[str] = ..., texts: _Optional[_Iterable[str]] = ...
    ) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("embeddings",)
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    embeddings: _containers.RepeatedCompositeFieldContainer[Embedding]
    def __init__(
        self, embeddings: _Optional[_Iterable[_Union[Embedding, _Mapping]]] = ...
    ) -> None: ...

class Embedding(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class GuidedDecodingParams(_message.Message):
    __slots__ = ("json_schema", "regex", "grammar", "choice")
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    REGEX_FIELD_NUMBER: _ClassVar[int]
    GRAMMAR_FIELD_NUMBER: _ClassVar[int]
    CHOICE_FIELD_NUMBER: _ClassVar[int]
    json_schema: str
    regex: str
    grammar: str
    choice: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        json_schema: _Optional[str] = ...,
        regex: _Optional[str] = ...,
        grammar: _Optional[str] = ...,
        choice: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class Completion(_message.Message):
    __slots__ = ("text", "prompt_tokens", "completion_tokens")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    text: str
    prompt_tokens: int
    completion_tokens: int
    def __init__(
        self,
        text: _Optional[str] = ...,
        prompt_tokens: _Optional[int] = ...,
        completion_tokens: _Optional[int] = ...,
    ) -> None: ...

class CompleteRequest(_message.Message):
    __slots__ = (
        "launch_id",
        "prompts",
        "max_tokens",
        "temperature",
        "top_p",
        "guided_decoding",
    )
    LAUNCH_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPTS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    GUIDED_DECODING_FIELD_NUMBER: _ClassVar[int]
    launch_id: str
    prompts: _containers.RepeatedScalarFieldContainer[str]
    max_tokens: int
    temperature: float
    top_p: float
    guided_decoding: GuidedDecodingParams
    def __init__(
        self,
        launch_id: _Optional[str] = ...,
        prompts: _Optional[_Iterable[str]] = ...,
        max_tokens: _Optional[int] = ...,
        temperature: _Optional[float] = ...,
        top_p: _Optional[float] = ...,
        guided_decoding: _Optional[_Union[GuidedDecodingParams, _Mapping]] = ...,
    ) -> None: ...

class CompleteResponse(_message.Message):
    __slots__ = ("completions",)
    COMPLETIONS_FIELD_NUMBER: _ClassVar[int]
    completions: _containers.RepeatedCompositeFieldContainer[Completion]
    def __init__(
        self, completions: _Optional[_Iterable[_Union[Completion, _Mapping]]] = ...
    ) -> None: ...
