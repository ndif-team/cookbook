"""Run shell examples in the active isolated Python environment."""
import os, subprocess, sys
from pathlib import Path

def check_shell(row, out):
    script=out/(row['file'].replace('.md','')+'_'+str(row['index'])+'.sh')
    script.write_text(row['code'])
    subprocess.run(['bash','-n',str(script)],check=True,capture_output=True,text=True)
    fixture=out/'your_script.py'
    fixture.write_text('import torch\nfrom nnsight import NNsight, CONFIG\nassert CONFIG.APP.DEBUG\nmodel=NNsight(torch.nn.Linear(5,2))\nwith model.trace(torch.ones(1,5)):\n    output=model.output.save()\nassert output.shape==(1,2)\n')
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    return subprocess.run(['bash',str(script)],cwd=out,env=env,check=True,capture_output=True,text=True).stdout
