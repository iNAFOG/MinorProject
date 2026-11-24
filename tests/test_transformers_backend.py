"""Parsing/formatting for the local Transformers backend; no weights are loaded."""
import json
from agent_harness.config import ModelConfig
from agent_harness.models import build_model
from agent_harness.models.openai_compatible import INVALID_ARGUMENTS
from agent_harness.models.transformers_local import TransformersModel, parse_tool_calls, to_chat


def test_parses_qwen_tool_calls_and_content():
    text = 'Searching.\n<tool_call>\n{"name": "fire", "arguments": {"cells": [1, 2, 3]}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == 'Searching.'
    assert calls[0]['function']['name'] == 'fire'
    assert json.loads(calls[0]['function']['arguments']) == {'cells': [1, 2, 3]}


def test_malformed_and_truncated_calls_become_failed_actions():
    _, calls = parse_tool_calls('<tool_call>{"name": "fire", "arguments": {"cells": [1,</tool_call>')
    assert calls[0]['function']['name'] == 'invalid_tool_call'
    assert INVALID_ARGUMENTS in json.loads(calls[0]['function']['arguments'])
    _, calls = parse_tool_calls('<tool_call>\n{"name": "fire", "arguments": {"cells": [4, 5, 6]}}')  # cut at limit
    assert calls[0]['function']['name'] == 'fire'
    _, calls = parse_tool_calls('<tool_call>{"name": "fire", "arguments": "{\\"cells\\": [7, 8, 9]}"}</tool_call>')
    assert json.loads(calls[0]['function']['arguments']) == {'cells': [7, 8, 9]}


def test_plain_text_has_no_calls():
    assert parse_tool_calls('Game complete.') == ('Game complete.', [])


def test_wire_messages_convert_to_template_messages():
    chat = to_chat([{'role': 'system', 'content': 's'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'id': 'call_0', 'type': 'function', 'function': {'name': 'fire', 'arguments': '{"cells": [1, 2, 3]}'}}]},
                    {'role': 'tool', 'tool_call_id': 'call_0', 'name': 'fire', 'content': '{"hits": 1}'},
                    {'role': 'tool', 'content': 'legacy observation'}])
    assert chat[1] == {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'fire', 'arguments': {'cells': [1, 2, 3]}}}]}
    assert chat[2] == {'role': 'tool', 'content': '{"hits": 1}'}
    assert chat[3]['role'] == 'user'


def test_backend_is_registered_without_loading_weights():
    assert isinstance(build_model(ModelConfig(backend='transformers', model='any/model')), TransformersModel)
