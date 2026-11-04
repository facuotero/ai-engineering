# AsyncLLMManager

Ejemplo Python 3.12 de generación asíncrona con OpenAI Responses o Anthropic Messages. Valida las entradas con Pydantic y entrega fragmentos `str` mediante `yield`. Ante un fallo esperado, entrega un único `LLMError` y termina el generador.

## Instalación

Desde la raíz del proyecto, con Python 3.12 instalado:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En este equipo se instaló Python 3.12 mediante Homebrew (`brew install python@3.12`) y ya existe `.venv`. Si `python3.12` no está en el PATH, usa `/opt/homebrew/bin/python3.12`. En Windows, crea el entorno con `py -3.12 -m venv .venv` y actívalo con `.venv\Scripts\Activate.ps1`.

`requirements.txt` fija las versiones directas y transitivas verificadas: OpenAI 3.44.0, Anthropic 2.4.0, Pydantic 2.14.2 y python-dotenv 1.2.4.

## Configuración

```bash
cp .env.example .env
```

Edita `.env` con el proveedor, un modelo disponible en tu cuenta y su clave:

| Variable | Uso |
| --- | --- |
| `LLM_PROVIDER` | Obligatoria: `openai` o `anthropic`. |
| `LLM_MODEL` | Obligatoria: identificador del modelo para ese proveedor. |
| `OPENAI_API_KEY` | Obligatoria solo para OpenAI. |
| `ANTHROPIC_API_KEY` | Obligatoria solo para Anthropic. |

Ejemplo OpenAI:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=tu-clave
```

Para Anthropic, cambia `LLM_PROVIDER=anthropic`, establece `LLM_MODEL` con un identificador de Claude habilitado en tu cuenta y completa `ANTHROPIC_API_KEY`. No necesita la clave de OpenAI. El script carga el `.env` junto a `demo.py`; las variables ya exportadas tienen precedencia. `.env` está excluido de Git.

## Ejecutar el script de prueba

```bash
source .venv/bin/activate
python demo.py
python demo.py "Explica asyncio con un ejemplo breve" --max-tokens 512
# Solo con un modelo OpenAI compatible con temperatura:
python demo.py "Escribe un haiku" --temperature 0.7
```

El texto se imprime conforme llega, con `flush=True`. Los errores se escriben en stderr sin traceback ni credenciales: salida 1 para errores del gestor, 2 para configuración o validación y 130 para Ctrl+C. Las llamadas reales requieren conexión, acceso al modelo y crédito de API; pueden generar cargos.

## Validación y contrato

`GenerationParams` exige `prompt` y `model` no vacíos, `max_tokens` entero positivo (1024 por defecto) y `temperature` opcional entre 0 y 2. Rechaza campos desconocidos, booleanos en campos numéricos y valores no finitos. Los límites de tokens específicos del modelo se validan en la API; no se recortan silenciosamente.

La temperatura se omite por defecto. El adaptador de OpenAI permite configurarla únicamente para los identificadores enumerados en `OPENAI_TEMPERATURE_MODELS` (familias GPT-4.1 y GPT-4o). Para otros modelos devuelve un error de validación; esta lista conservadora no implica que todos los demás modelos carezcan de soporte. El SDK Anthropic instalado ya no expone ese parámetro: este adaptador lo rechaza explícitamente, incluso para modelos antiguos. Consulta la [migración del SDK Anthropic](https://github.com/anthropics/anthropic-sdk-python/blob/main/MIGRATION.md) antes de ampliar esa compatibilidad.

`AsyncLLMManager.generate(params)` produce `AsyncIterator[str | LLMError]`. `LLMError` contiene `code`, `message` y `retryable`. Los códigos incluyen `validation`, `configuration`, `connection`, `timeout`, `rate_limit`, `authentication`, `api_error` e `incomplete`. Si falla después de emitir texto, conserva esos fragmentos y emite el error terminal; el consumidor debe tratar esa respuesta como parcial.

```python
import asyncio
from contextlib import aclosing
from dotenv import load_dotenv
from llm_manager import AsyncLLMManager, GenerationParams, LLMError, Settings

async def main():
    load_dotenv('.env', override=False)
    settings = Settings.from_env()
    params = GenerationParams(prompt='Hola', model=settings.model)
    async with AsyncLLMManager(settings) as manager:
        async with aclosing(manager.generate(params)) as stream:
            async for item in stream:
                if isinstance(item, LLMError):
                    print(f'Error: {item.code}: {item.message}')
                    break
                print(item, end='', flush=True)

asyncio.run(main())
```

El modelo de `params` es el enviado a la API; `settings.model` proporciona el valor configurado para construir esos parámetros. Usa `async with` para cerrar el cliente y `aclosing` cuando puedas abandonar el generador anticipadamente. La cancelación se propaga. El timeout de red es de 60 segundos por operación, no un límite total de duración de la generación. Los reintentos están desactivados para evitar duplicar texto; `retryable` es una indicación para el consumidor, no un reintento automático.

OpenAI transmite deltas siguiendo su [guía de streaming](https://developers.openai.com/api/docs/guides/streaming-responses); Anthropic usa `messages.stream().text_stream`. El helper de Anthropic acumula internamente el mensaje aunque el gestor entregue cada fragmento inmediatamente.

## Pruebas sin API

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Las pruebas no requieren claves reales ni conexión. Cubren validación, configuración, ambos proveedores, fragmentos incrementales, cierre anticipado, cancelación y errores de conexión, timeout, autenticación y rate limiting antes y después del primer fragmento. También ejecutan los SDK reales con HTTP/SSE simulado para verificar el mapeo de parámetros, lectura de eventos y ausencia de reintentos ante HTTP 429.

Las pruebas locales no certifican la disponibilidad del modelo ni las credenciales de tu cuenta. Ejecuta `python demo.py` después de configurar `.env` para comprobar la integración real.
