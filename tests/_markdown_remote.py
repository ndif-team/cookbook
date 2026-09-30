"""Isolated NDIF protocol fixture: real serialization and inference; no network."""
from contextlib import contextmanager
from unittest.mock import patch
import io, json, linecache
from types import SimpleNamespace
import httpx, torch, zstandard
from nnsight import CONFIG
from nnsight.intervention.backends.remote import RemoteBackend
from nnsight.schema.request import RequestModel
from nnsight.schema.response import ResponseModel, Status
from nnsight.tracing.tracer import _saves

@contextmanager
def remote_transport(model):
    payloads=[]; sockets=[]; result_blob=None
    class Socket:
        def __init__(self): self.messages=[json.dumps({'session_id':'fixture-session'})];self.closed=False
        def recv(self): return self.messages.pop(0)
        def close(self): self.closed=True
        def settimeout(self, timeout): pass
    def connect(*args, **kwargs):
        socket=Socket();sockets.append(socket);return socket
    def post(backend, request, blob):
        nonlocal result_blob
        assert blob and request.model_key
        payloads.append(blob)
        snapshot=dict(linecache.cache)
        try:
            restored=RequestModel.deserialize(blob, model._remoteable_persistent_objects(),compress=backend.compress)
        finally:
            linecache.cache.clear();linecache.cache.update(snapshot)
        restored.execute(restored.info.code)
        saves={k:v for k,v in restored.info.frame.f_locals.items() if id(v) in _saves()}
        assert saves, 'The serialized intervention must produce saved values'
        buffer=io.BytesIO();torch.save(saves, buffer)
        result_blob=buffer.getvalue()
        if backend.compress:result_blob=zstandard.ZstdCompressor().compress(result_blob)
        for socket in sockets:
            socket.messages.append(ResponseModel(id='fixture-job',status=Status.COMPLETED,data=result_blob).pickle())
        return ResponseModel(id='fixture-job',status=Status.RECEIVED)
    def handle(request):
        if '/response/' in str(request.url):
            return httpx.Response(200,json=ResponseModel(id='fixture-job',status=Status.COMPLETED,data='https://fixture.test/result').model_dump())
        assert str(request.url)=='https://fixture.test/result', str(request.url)
        return httpx.Response(200,content=result_blob)
    client=httpx.Client
    def factory(*args,**kwargs):return client(*args,transport=httpx.MockTransport(handle),**kwargs)
    old_key=CONFIG.API.APIKEY;old_host=CONFIG.API.HOST
    CONFIG.API.APIKEY='';CONFIG.API.HOST='https://fixture.test'
    try:
        with patch('huggingface_hub.HfApi.model_info',lambda self, repo_id, **kw:SimpleNamespace(id=repo_id)),patch.object(RemoteBackend,'_post',post),patch('httpx.Client',factory),patch('websocket.create_connection',connect):
            yield payloads
        assert payloads
        assert all(s.closed for s in sockets)
    finally:
        CONFIG.API.APIKEY=old_key;CONFIG.API.HOST=old_host
