from dataclasses import replace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
import json
import pytest
from agent_harness.config import ModelConfig
from agent_harness.models.local import LocalModel
from agent_harness.models.openai_compatible import OpenAICompatibleModel


def test_openai_compatible_and_local_wire_protocol():
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body={'choices':[{'message':{'content':None,'tool_calls':[{'id':'provider-call-1','type':'function',
                'extra_content':{'google':{'thought_signature':'opaque-test-signature'}},'function':{
                'name':'read','arguments':'{"path":"solution.py"}'}}]}}],
                'usage':{'prompt_tokens':17,'completion_tokens':9}}
            raw=json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self,*args):
            pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        cfg=ModelConfig(backend='local',base_url=f'http://127.0.0.1:{server.server_port}/v1',
                        input_cost_per_million=1,output_cost_per_million=2)
        response=LocalModel(cfg).generate([{'role':'tool','content':'result'}],
            [{'name':'read','parameters':{'type':'object'}}],{'max_output_tokens':25})
        assert response.actions[0].arguments=={'path':'solution.py'}
        assert response.input_tokens==17 and response.output_tokens==9
        assert response.estimated_cost==pytest.approx(35/1e6)
        assert received[0]['max_tokens']==25
        assert received[0]['messages'][0]['role']=='user'
        assert received[0]['tools'][0]['type']=='function'
        assert response.tool_calls[0]['extra_content']['google']['thought_signature']=='opaque-test-signature'
        LocalModel(cfg).generate([
            {'role':'assistant','content':None,'tool_calls':list(response.tool_calls)},
            {'role':'tool','tool_call_id':'provider-call-1','name':'read','content':'{"result":"source"}'},
        ],context={'max_output_tokens':25})
        assert received[1]['messages'][0]['tool_calls'][0]==response.tool_calls[0]
        assert received[1]['messages'][1]['role']=='tool'
        assert received[1]['messages'][1]['tool_call_id']=='provider-call-1'
    finally:
        server.shutdown()
        server.server_close()
    with pytest.raises(ValueError):
        LocalModel(replace(cfg,base_url='https://example.com'))
