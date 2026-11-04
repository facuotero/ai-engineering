"""Run a single streaming request using the configuration in .env."""
import argparse
import asyncio
from contextlib import aclosing
from pathlib import Path
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_manager import AsyncLLMManager, GenerationParams, LLMError, Settings


async def main() -> int:
    parser = argparse.ArgumentParser(description='Streaming asíncrono con OpenAI o Anthropic')
    parser.add_argument('prompt', nargs='?', default='Explica qué es un generador de Python en tres frases.')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max-tokens', type=int, default=1024)
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name('.env'), override=False)
    try:
        settings = Settings.from_env()
        params = GenerationParams(prompt=args.prompt, model=settings.model,
                                  temperature=args.temperature, max_tokens=args.max_tokens)
    except ValidationError as exc:
        fields = ', '.join('.'.join(map(str, e['loc'])) for e in exc.errors())
        print(f'Configuración o parámetros inválidos: {fields}.', file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    async with AsyncLLMManager(settings) as manager:
        async with aclosing(manager.generate(params)) as output:
            async for item in output:
                if isinstance(item, LLMError):
                    print(f'\n[{item.code}] {item.message}', file=sys.stderr)
                    return 1
                print(item, end='', flush=True)
    print()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print('\nCancelado.', file=sys.stderr)
        raise SystemExit(130)
