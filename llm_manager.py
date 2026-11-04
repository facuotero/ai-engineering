"""Validated, asynchronous text streaming for OpenAI and Anthropic."""
from collections.abc import AsyncIterator
import os
from typing import Literal

import anthropic
import httpx2
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    prompt: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=1024, gt=0)

    @field_validator('prompt', 'model')
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('El texto no puede estar vacío.')
        return value


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    provider: Literal['openai', 'anthropic']
    model: str = Field(min_length=1)
    api_key: SecretStr

    @field_validator('model')
    @classmethod
    def clean_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('LLM_MODEL es obligatorio.')
        return value.strip()

    @field_validator('api_key')
    @classmethod
    def nonempty_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError('Falta la clave del proveedor seleccionado.')
        return value

    @classmethod
    def from_env(cls) -> 'Settings':
        provider = os.getenv('LLM_PROVIDER', '').strip().lower()
        if provider not in ('openai', 'anthropic'):
            raise ValueError('LLM_PROVIDER debe ser openai o anthropic.')
        key_name = 'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY'
        if not os.getenv(key_name, '').strip():
            raise ValueError(f'Falta {key_name}.')
        return cls(provider=provider, model=os.getenv('LLM_MODEL', ''), api_key=os.environ[key_name])


class LLMError(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str
    message: str
    retryable: bool = False


# Conservative allowlist: unknown/new models work with temperature omitted.
OPENAI_TEMPERATURE_MODELS = frozenset({
    'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4.1-2025-04-14',
    'gpt-4.1-mini-2025-04-14', 'gpt-4.1-nano-2025-04-14',
    'gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20',
    'gpt-4o-mini-2024-07-18',
})


class AsyncLLMManager:
    """Use with async with; explicitly aclose generators when stopping early."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    async def __aenter__(self) -> 'AsyncLLMManager':
        if self._client is not None:
            raise RuntimeError('El gestor ya está abierto.')
        factory = AsyncOpenAI if self.settings.provider == 'openai' else AsyncAnthropic
        self._client = factory(api_key=self.settings.api_key.get_secret_value(), timeout=60.0, max_retries=0)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None

    async def generate(self, params: GenerationParams) -> AsyncIterator[str | LLMError]:
        if self._client is None:
            yield LLMError(code='configuration', message='Usa el gestor dentro de async with.')
            return
        try:
            # Revalidate even objects made with model_construct/model_copy.
            params = GenerationParams.model_validate(params.model_dump())
        except (ValidationError, AttributeError):
            yield LLMError(code='validation', message='Parámetros de generación inválidos.')
            return
        if params.temperature is not None and (
            self.settings.provider == 'anthropic' or params.model not in OPENAI_TEMPERATURE_MODELS
        ):
            yield LLMError(code='validation', message='Temperatura no admitida por este adaptador para el modelo seleccionado; omítela.')
            return
        try:
            if self.settings.provider == 'openai':
                options = {} if params.temperature is None else {'temperature': params.temperature}
                stream = await self._client.responses.create(
                    model=params.model, input=params.prompt, max_output_tokens=params.max_tokens,
                    stream=True, **options,
                )
                async with stream:
                    async for event in stream:
                        if event.type == 'response.output_text.delta' and event.delta:
                            yield event.delta
                        elif event.type in ('error', 'response.failed'):
                            yield LLMError(code='api_error', message='El proveedor informó un error durante la generación.')
                            return
                        elif event.type == 'response.incomplete':
                            yield LLMError(code='incomplete', message='La respuesta terminó incompleta; revisa el límite de tokens.')
                            return
            else:
                async with self._client.messages.stream(
                    model=params.model, messages=[{'role': 'user', 'content': params.prompt}],
                    max_tokens=params.max_tokens,
                ) as stream:
                    async for text in stream.text_stream:
                        if text:
                            yield text
        except (openai.RateLimitError, anthropic.RateLimitError):
            yield LLMError(code='rate_limit', message='Límite de solicitudes o cuota alcanzado. Intenta más tarde o revisa tu cuota.', retryable=True)
        except (openai.APITimeoutError, anthropic.APITimeoutError, httpx2.TimeoutException):
            yield LLMError(code='timeout', message='Se agotó el tiempo de espera.', retryable=True)
        except (openai.APIConnectionError, anthropic.APIConnectionError, httpx2.RequestError):
            yield LLMError(code='connection', message='No se pudo mantener la conexión con el proveedor.', retryable=True)
        except (openai.AuthenticationError, anthropic.AuthenticationError):
            yield LLMError(code='authentication', message='La API rechazó las credenciales.')
        except (openai.APIStatusError, anthropic.APIStatusError) as exc:
            yield LLMError(code='api_error', message='La API rechazó la solicitud.', retryable=exc.status_code >= 500)
        except (openai.APIError, anthropic.APIError):
            yield LLMError(code='api_error', message='El proveedor devolvió una respuesta inválida.')
