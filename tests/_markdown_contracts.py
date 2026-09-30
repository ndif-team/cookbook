"""Exercise definition-only and implementation-excerpt Markdown blocks."""
import asyncio, io
import torch
from nnsight import NNsight
from nnsight.intervention.backends.remote import AsyncRemoteBackend
from nnsight.schema.response import ResponseModel, Status

def check_contract(row, ns):
    key=(row['file'],row['index'])
    if key==('DESIGN_DECISIONS.md',1):
        weights=torch.arange(12).reshape(4,3)
        assert torch.equal(ns['CallableEmbed'](weights)(torch.tensor([1,3])),weights[[1,3]])
    elif key in [('NNsight.md',3),('NNsight.md',60)]:
        model=NNsight(torch.nn.Linear(5,2))
        with model.trace(torch.ones(1,5), backend=ns['Backend']()):
            result=model.output.save()
        assert result.shape==(1,2)
    elif key==('NNsight.md',6):
        from nnsight.intervention.util import first_input, replace_first_input
        from nnsight.intervention.interleaver import Mediator
        ns.update(first_input=first_input,replace_first_input=replace_first_input,Mediator=Mediator)
        views=type('ExcerptViews',(NNsight,),{k:ns[k] for k in ['input','inputs','output']})
        model=views(torch.nn.Linear(5,2))
        with model.trace(torch.ones(1,5)):
            original=model.input.save()
            model.input=torch.zeros(1,5)
            result=model.output.save()
        assert torch.equal(original,torch.ones(1,5))
        assert torch.allclose(result,model._module.bias.unsqueeze(0))
    elif key==('NNsight.md',49):
        buffer=io.BytesIO();torch.save({'value':torch.tensor(3)},buffer)
        messages=[ResponseModel(id='fixture',status=Status.RUNNING).model_dump_json(),
                  ResponseModel(id='fixture',status=Status.COMPLETED,data=buffer.getvalue()).pickle()]
        class Socket:
            closed=False
            def recv(self):return messages.pop(0)
            def close(self):self.closed=True
        backend=AsyncRemoteBackend('fixture',api_key='fixture',host='https://fixture.test')
        backend.compress=False;backend.connection=Socket();connection=backend.connection
        result=asyncio.run(ns['collect_updates'](backend))
        assert result['value'].item()==3 and connection.closed
    elif key==('NNsight.md',53):
        model=ns['MyModel'](torch.nn.Sequential(torch.nn.Identity(),torch.nn.Linear(5,2)))
        with model.trace(torch.ones(1,5)):
            lens=model.logit_lens(model[0].output).save()
            result=model.output.save()
        assert torch.equal(lens,result)
    elif key==('NNsight.md',57):
        class Producer(torch.nn.Module):
            def forward(self, x):return type(model).telemetry.provide(model,x)
        model=ns['MyModel'](Producer())
        with model.trace(torch.ones(1,5)):
            telemetry=model.telemetry.save()
            model.telemetry=torch.zeros(1,5)
            result=model.output.save()
        assert torch.equal(telemetry,torch.ones(1,5)) and torch.equal(result,torch.zeros(1,5))
    elif key==('NNsight.md',58):
        assert torch.equal(ns['head_output'].view(1,2,12,2)[:,:,5],torch.zeros(1,2,2))
    elif key==('NNsight.md',59):
        model=ns['model']
        assert isinstance(model.transformer.h[0].mlp,ns['Heads'])
        with model.trace("Hello world"):
            model.transformer.h[0].mlp.heads[:,5]=0
            result=model.transformer.h[0].mlp.output.save()
        assert torch.equal(result.view(1,-1,12,64)[:,:,5],torch.zeros_like(result.view(1,-1,12,64)[:,:,5]))
