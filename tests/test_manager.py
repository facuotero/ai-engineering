import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import anthropic
import httpx2 as httpx
import openai
from pydantic import ValidationError

from llm_manager import AsyncLLMManager, GenerationParams, LLMError, Settings


class ValidationTests(unittest.TestCase):
    def test_parameters(self):
        for value in (0, 2, 0.5, None):
            GenerationParams(prompt='hello', model='model', temperature=value)
        for changes in ({'temperature': -1}, {'temperature': 3}, {'temperature': float('nan')},
                        {'max_tokens': 0}, {'max_tokens': True}, {'max_tokens': 1.5},
                        {'prompt': '  '}, {'model': ''}, {'unknown': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                GenerationParams.model_validate(dict(prompt='hi', model='model') | changes)

    def test_configuration(self):
        for env in ({}, {'LLM_PROVIDER': 'other', 'LLM_MODEL': 'x'},
                    {'LLM_PROVIDER': 'openai', 'LLM_MODEL': 'x'}):
            with patch.dict('os.environ', env, clear=True), self.assertRaises(ValueError):
                Settings.from_env()
        with patch.dict('os.environ', {'LLM_PROVIDER': 'anthropic', 'LLM_MODEL': 'x',
                                      'ANTHROPIC_API_KEY': 'secret'}, clear=True):
            settings = Settings.from_env()
            self.assertEqual(settings.provider, 'anthropic')
            self.assertNotIn('secret', repr(settings))


class FakeStream:
    def __init__(self, items, error=None):
        self.items, self.error, self.closed = items, error, False
        self.read = 0
        self.text_stream = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.read < len(self.items):
            item = self.items[self.read]
            self.read += 1
            return item
        if self.error:
            raise self.error
        raise StopAsyncIteration


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def setup_client(self, provider, stream, opening_error=None):
        client = NS(close=AsyncMock())
        if provider == 'openai':
            client.responses = NS(create=AsyncMock(return_value=stream, side_effect=opening_error))
        else:
            client.messages = NS(stream=Mock(return_value=stream, side_effect=opening_error))
        return client

    async def run_case(self, provider, items, error=None, opening=False):
        stream = FakeStream(items, None if opening else error)
        client = self.setup_client(provider, stream, error if opening else None)
        constructor = 'AsyncOpenAI' if provider == 'openai' else 'AsyncAnthropic'
        with patch(f'llm_manager.{constructor}', return_value=client):
            async with AsyncLLMManager(Settings(provider=provider, model='test', api_key='fake')) as manager:
                result = [item async for item in manager.generate(GenerationParams(prompt='hi', model='test'))]
        client.close.assert_awaited_once()
        if not opening:
            self.assertTrue(stream.closed)
        return result

    async def test_both_providers(self):
        self.assertEqual(await self.run_case('openai', [NS(type='response.created'),
            NS(type='response.output_text.delta', delta='Hola'),
            NS(type='response.output_text.delta', delta='!'), NS(type='response.completed')]), ['Hola', '!'])
        self.assertEqual(await self.run_case('anthropic', ['Hola', '!']), ['Hola', '!'])

    async def test_network_errors_before_and_after_text(self):
        request = httpx.Request('POST', 'https://example.com')
        for provider, sdk in [('openai', openai), ('anthropic', anthropic)]:
            errors = [(sdk.APIConnectionError(request=request), 'connection'),
                      (sdk.APITimeoutError(request=request), 'timeout'),
                      (sdk.RateLimitError('secret', response=httpx.Response(429, request=request), body=None), 'rate_limit'),
                      (sdk.AuthenticationError('secret', response=httpx.Response(401, request=request), body=None), 'authentication')]
            for error, code in errors:
                for opening in (True, False):
                    with self.subTest(provider=provider, code=code, opening=opening):
                        items = ['hello'] if provider == 'anthropic' else [NS(type='response.output_text.delta', delta='hello')]
                        result = await self.run_case(provider, items, error, opening)
                        self.assertEqual(len(result), 1 if opening else 2)
                        self.assertIsInstance(result[-1], LLMError)
                        self.assertEqual(result[-1].code, code)
                        self.assertNotIn('secret', result[-1].message)

    async def test_incremental_and_early_close(self):
        stream = FakeStream([NS(type='response.output_text.delta', delta='a'), NS(type='response.output_text.delta', delta='b')])
        client = self.setup_client('openai', stream)
        with patch('llm_manager.AsyncOpenAI', return_value=client):
            async with AsyncLLMManager(Settings(provider='openai', model='x', api_key='fake')) as manager:
                gen = manager.generate(GenerationParams(prompt='hi', model='x'))
                self.assertEqual(await anext(gen), 'a')
                self.assertEqual(stream.read, 1)
                await gen.aclose()
                self.assertTrue(stream.closed)

    async def test_cancellation_propagates(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.run_case('anthropic', [], asyncio.CancelledError())

    async def test_temperature_rejected_before_request(self):
        for provider, model, temperature in [('anthropic', 'claude-new', 0.5), ('openai', 'unknown', 1.0)]:
            constructor = 'AsyncOpenAI' if provider == 'openai' else 'AsyncAnthropic'
            client = self.setup_client(provider, FakeStream([]))
            with patch(f'llm_manager.{constructor}', return_value=client):
                async with AsyncLLMManager(Settings(provider=provider, model=model, api_key='fake')) as manager:
                    result = [x async for x in manager.generate(GenerationParams(prompt='hi', model=model, temperature=temperature))]
                self.assertEqual(result[0].code, 'validation')

    async def test_failed_event(self):
        result = await self.run_case('openai', [NS(type='response.failed')])
        self.assertEqual(result[0].code, 'api_error')
