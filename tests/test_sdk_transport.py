"""Exercise the real SDKs with offline HTTP/SSE responses."""
import json
import unittest
from unittest.mock import patch

import httpx2 as httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from llm_manager import AsyncLLMManager, GenerationParams, Settings


def sse(events):
    return ''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n' for e in events)


class SDKTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, provider, handler, temperature=None):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        factory = AsyncOpenAI if provider == 'openai' else AsyncAnthropic
        client = factory(api_key='offline-test', http_client=http, max_retries=0)
        with patch(f'llm_manager.{factory.__name__}', return_value=client):
            async with AsyncLLMManager(Settings(provider=provider, model='gpt-4.1-mini', api_key='offline-test')) as manager:
                result = [x async for x in manager.generate(GenerationParams(
                    prompt='Hi', model='gpt-4.1-mini', temperature=temperature, max_tokens=17))]
        self.assertTrue(http.is_closed)
        return result

    async def test_openai_sse_and_mapping(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['max_output_tokens'], 17)
            self.assertEqual(body['temperature'], 0.5)
            self.assertTrue(body['stream'])
            self.assertEqual(body['input'], 'Hi')
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=sse([
                {'type': 'response.output_text.delta', 'delta': 'Hola', 'item_id': 'msg_1',
                 'output_index': 0, 'content_index': 0, 'sequence_number': 1},
                {'type': 'response.output_text.delta', 'delta': '!', 'item_id': 'msg_1',
                 'output_index': 0, 'content_index': 0, 'sequence_number': 2},
            ]))
        self.assertEqual(await self.collect('openai', handler, 0.5), ['Hola', '!'])

    async def test_anthropic_sse_and_mapping(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['max_tokens'], 17)
            self.assertNotIn('temperature', body)
            self.assertEqual(body['messages'], [{'role': 'user', 'content': 'Hi'}])
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=sse([
                {'type': 'message_start', 'message': {'id': 'msg_1', 'type': 'message', 'role': 'assistant',
                 'model': 'test', 'content': [], 'stop_reason': None, 'stop_sequence': None,
                 'usage': {'input_tokens': 1, 'output_tokens': 0}}},
                {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hola'}},
                {'type': 'content_block_stop', 'index': 0},
                {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                 'usage': {'output_tokens': 1}},
                {'type': 'message_stop'},
            ]))
        self.assertEqual(await self.collect('anthropic', handler), ['Hola'])

    async def test_http_429_is_not_retried(self):
        for provider in ('openai', 'anthropic'):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(429, json={'error': {'type': 'rate_limit_error', 'message': 'private'}})
            result = await self.collect(provider, handler)
            self.assertEqual(result[0].code, 'rate_limit')
            self.assertEqual(len(calls), 1)
            self.assertNotIn('private', result[0].message)
