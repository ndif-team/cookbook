"""Execute the literal Markdown blocks with explicit, documented prerequisites."""
import argparse
import ast
import copy
import gc
import inspect
import json
import linecache
import os
from pathlib import Path
import re
import textwrap
import traceback
import warnings
import tempfile
import sys
from _markdown_shell import check_shell
from contextlib import nullcontext

OUT = Path(os.environ.get('MARKDOWN_RESULTS_DIR', tempfile.mkdtemp(prefix='cookbook-md-')))
OUT.mkdir(parents=True,exist_ok=True)
os.environ['NNSIGHT_CONFIG'] = str(OUT / 'config.yaml')
os.environ.pop('NDIF_API_KEY',None)
import torch
import nnsight
from nnsight import NNsight, TransformersModel, Envoy, CONFIG
from nnsight.intervention.eproperty import eproperty
from nnsight.tracing.tracer import Tracer, BLOCKS, push_result
from nnsight.tracing.util import Scope
from transformers import AutoModelForCausalLM, AutoTokenizer
from _markdown_remote import remote_transport
from _markdown_contracts import check_contract

ROOT = Path(__file__).resolve().parents[1]

torch.set_num_threads(2)

def inventory():
    rows=[]
    for path in sorted(ROOT.glob('*.md')):
        for i,m in enumerate(re.finditer(r'^([ ]*)```([^\n]*)\n(.*?)^\1```[^\n]*$', path.read_text(), re.M|re.S),1):
            source=path.read_text()
            headings=re.findall(r'^#{1,6} .+$',source[:m.start()],re.M)
            rows.append(dict(file=path.name,index=i,line=source[:m.start()].count('\n')+1,
                             lang=m[2],heading=headings[-1] if headings else '',section=(re.findall(r'^## .+$',source[:m.start()],re.M) or [''])[-1],code=textwrap.dedent(m[3])))
    return rows

def namespace(net,tokenizer):
    model=TransformersModel(copy.deepcopy(net),tokenizer=tokenizer,device='cpu')
    ns=dict(__name__='__markdown_example__',torch=torch,nn=torch.nn,nnsight=nnsight,
            NNsight=NNsight,TransformersModel=TransformersModel,Envoy=Envoy,CONFIG=CONFIG,
            Tracer=Tracer,eproperty=eproperty,Scope=Scope,push_result=push_result,
            model=model,llm=model,tokenizer=tokenizer,prompt='Hello world',os=os,Path=Path,
            torch_model=torch.nn.Sequential(torch.nn.Linear(5,10),torch.nn.Linear(10,2)),
            my_pytorch_model=torch.nn.Linear(5,2),REMOTE='local',Any=object,
            BLOCKS=BLOCKS,self=Tracer(),frame=inspect.currentframe(),code=compile('value = 1','<internal-example>','exec'))
    linecache.cache['<internal-example>']=(10,None,['value = 1\n'],'<internal-example>')
    ns['tensor']=torch.tensor([[1,2,3]])
    ns['mask']=torch.ones_like(ns['tensor'])
    ns['some_tensor']=torch.randn(3,requires_grad=True)
    ns['loss']=(ns['some_tensor']**2).sum()
    return ns

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--only');parser.add_argument('--start',type=int,default=0)
    args=parser.parse_args(argv)
    rows=inventory()
    (OUT/'inventory.json').write_text(json.dumps(rows,indent=2))
    net=AutoModelForCausalLM.from_pretrained('openai-community/gpt2',attn_implementation='eager').eval()
    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    results=[]
    for row in rows:
        if args.only and not re.search(args.only, row['file']+':'+str(row['index'])):continue
        if row['index']<args.start:continue
        result={k:v for k,v in row.items() if k!='code'}
        key=row['file']+':'+str(row['index'])
        if row['lang']!='python':
            if row['lang']=='bash':
                try:
                    result.update(status='passed-shell',output=check_shell(row,OUT))
                except Exception as error:
                    result.update(status='failed',error=str(error))
            else:
                result['status']='illustration'
            results.append(result);continue
        path=OUT/(row['file'].replace('.md','')+'_'+str(row['index'])+'.py')
        path.write_text(row['code'])
        try:ast.parse(row['code'])
        except Exception as e:
            result.update(status='syntax-error',error=str(e));results.append(result);print(key,'SYNTAX',e,flush=True);continue
        if 'from nnsight.modeling.vllm' in row['code'] or row['section']=='## vLLM Integration':
            result['status']='needs-cuda';results.append(result);print(key,'GPU',flush=True);continue
        is_remote=bool(re.search(r'(?m)^[^#\n]*remote=True',row['code']) or 'AsyncRemoteBackend' in row['code'])
        if row['file']=='DESIGN_DECISIONS.md' and row['index']==2:
            pass
        ns=namespace(net,tokenizer)
        if row['file']=='DESIGN_DECISIONS.md' and row['index']==2:
            import sys
            sys.path.insert(0,str(ROOT / "arena-ch1"))
            from nnterp_compat import StandardizedTransformer
            ns['model']=StandardizedTransformer(copy.deepcopy(net),tokenizer=tokenizer)
        if row['file']=='NNsight.md' and row['index']==56: ns['self']=ns['model']
        if row['file']=='NNsight.md' and row['index']==59:
            previous=next(r for r in rows if r['file']=='NNsight.md' and r['index']==58)
            nodes=ast.parse(previous['code']).body
            definitions=[n for n in nodes if isinstance(n,(ast.ClassDef,ast.Import,ast.ImportFrom))]
            exec(compile(ast.Module(body=definitions,type_ignores=[]),'<Heads prerequisite>','exec'),ns)
        saved_debug=CONFIG.APP.DEBUG;saved_pymount=CONFIG.APP.PYMOUNT
        warnings_seen=[]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                # Breakpoint example gets a recording, noninteractive hook.
                hits=[]
                ns['breakpoint']=lambda:hits.append('breakpoint reached')
                with remote_transport(ns['model']) if is_remote else nullcontext():
                    exec(compile(row['code'],str(path),'exec'),ns)
                    check_contract(row,ns)
                warnings_seen=[str(w.message) for w in caught]
            result.update(status='protocol-tested' if is_remote else 'passed',warnings=warnings_seen)
            print(key,'PASS',flush=True)
        except BaseException as e:
            result.update(status='failed',error=type(e).__name__+': '+str(e)[:1800])
            print(key,'FAIL',type(e).__name__,str(e)[:350].replace('\n',' '),flush=True)
        finally:
            CONFIG.APP.DEBUG=saved_debug;CONFIG.APP.PYMOUNT=saved_pymount
            del ns;gc.collect()
        results.append(result)
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}
    print(json.dumps(counts),flush=True)
    failures=[r for r in results if r['status'] in ('failed','syntax-error')]
    assert not failures, json.dumps(failures,indent=2)
    print('Report:', OUT / 'results.json',flush=True)
    return results

def test_markdown_examples():
    """Run all CPU and offline-protocol examples; report CUDA requirements explicitly."""
    results=main([])
    assert sum(r['status']=='needs-cuda' for r in results)==8
    assert sum(r['status']=='protocol-tested' for r in results)==3

if __name__=='__main__':main()
