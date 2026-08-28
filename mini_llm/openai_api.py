"""Typed translation between OpenAI Chat Completions and the local runtime."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Annotated, Any, Literal, Mapping, Self, Sequence
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from mini_llm.chat import validate_chat_messages
from mini_llm.generation import FinishReason
from mini_llm.interfaces import ChatMessage
from mini_llm.sampling import SamplingConfig


OpenAIFinishReason = Literal["stop", "length"]


class TextContentPart(BaseModel):
    """One supported text item from an OpenAI message content array."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class ChatCompletionMessageInput(BaseModel):
    """A text-only system, user, or assistant input message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str | Annotated[list[TextContentPart], Field(min_length=1)]

    def to_runtime_message(self) -> ChatMessage:
        if isinstance(self.content, str):
            content = self.content
        else:
            content = "".join(part.text for part in self.content)
        return ChatMessage(role=self.role, content=content)


class ChatCompletionRequest(BaseModel):
    """Supported subset of an OpenAI Chat Completions request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatCompletionMessageInput] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    n: int = Field(default=1, strict=True)

    @field_validator("stream")
    @classmethod
    def reject_streaming(cls, value: bool) -> bool:
        if value:
            raise ValueError("streaming is not supported")
        return value

    @field_validator("n")
    @classmethod
    def require_one_choice(cls, value: int) -> int:
        if value != 1:
            raise ValueError("only n=1 is supported")
        return value

    @model_validator(mode="after")
    def validate_supported_request(self) -> Self:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError(
                "max_tokens and max_completion_tokens cannot both be supplied"
            )
        validate_chat_messages(
            self.to_runtime_messages(), add_generation_prompt=True
        )
        return self

    @property
    def output_token_limit(self) -> int:
        """Resolve the two OpenAI token-limit spellings to one runtime value."""

        return self.max_completion_tokens or self.max_tokens or 128

    def to_runtime_messages(self) -> list[ChatMessage]:
        return [message.to_runtime_message() for message in self.messages]

    def to_sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )


@dataclass(frozen=True, slots=True)
class PreparedChatCompletionRequest:
    """Transport-independent arguments needed by ``Engine.generate``."""

    messages: list[ChatMessage]
    max_new_tokens: int
    sampling: SamplingConfig


def prepare_chat_completion_request(
    request: ChatCompletionRequest, *, served_model: str
) -> PreparedChatCompletionRequest:
    """Check the requested model and translate fields to runtime arguments."""

    if request.model != served_model:
        raise OpenAIRequestError(
            f"The model {request.model!r} does not exist",
            status_code=404,
            param="model",
            code="model_not_found",
        )
    return PreparedChatCompletionRequest(
        messages=request.to_runtime_messages(),
        max_new_tokens=request.output_token_limit,
        sampling=request.to_sampling_config(),
    )


class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @classmethod
    def from_counts(cls, prompt_tokens: int, completion_tokens: int) -> Self:
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: Literal[0] = 0
    message: AssistantMessage
    logprobs: None = None
    finish_reason: OpenAIFinishReason


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


def create_chat_completion_response(
    *,
    model: str,
    text: str,
    finish_reason: FinishReason,
    prompt_tokens: int,
    completion_tokens: int,
    completion_id: str | None = None,
    created: int | None = None,
) -> ChatCompletionResponse:
    """Build one non-streaming response from a completed generation."""

    return ChatCompletionResponse(
        id=completion_id or f"chatcmpl-{uuid4().hex}",
        created=int(time.time()) if created is None else created,
        model=model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(content=text),
                finish_reason=to_openai_finish_reason(finish_reason),
            )
        ],
        usage=CompletionUsage.from_counts(prompt_tokens, completion_tokens),
    )


def to_openai_finish_reason(reason: FinishReason) -> OpenAIFinishReason:
    if reason == "eos":
        return "stop"
    if reason in ("max_new_tokens", "context_length"):
        return "length"
    raise ValueError(f"unsupported runtime finish reason: {reason!r}")


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


class OpenAIRequestError(ValueError):
    """A request failure carrying its HTTP status and OpenAI error body."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = OpenAIErrorResponse(
            error=OpenAIErrorDetail(
                message=message,
                type=error_type,
                param=param,
                code=code,
            )
        )


def validation_error_response(error: ValidationError) -> OpenAIErrorResponse:
    """Convert the first schema failure to the standard OpenAI error envelope."""

    return validation_errors_response(error.errors(include_url=False))


def validation_errors_response(
    errors: Sequence[Mapping[str, Any]],
) -> OpenAIErrorResponse:
    """Convert framework-provided validation details to an OpenAI envelope."""

    first_error = errors[0]
    location = first_error.get("loc", ())
    if location and location[0] == "body":
        location = location[1:]
    param = ".".join(str(part) for part in location) or None
    message = str(first_error["msg"])
    if message.startswith("Value error, "):
        message = message.removeprefix("Value error, ")
    return OpenAIErrorResponse(
        error=OpenAIErrorDetail(
            message=message,
            type="invalid_request_error",
            param=param,
            code=None,
        )
    )
