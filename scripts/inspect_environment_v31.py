"""Record the actual runtime and selected dependency revisions without secrets."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Environment evidence must use a new output path')
    import torch
    def command(arguments):
        result=subprocess.run(arguments,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return {'returncode':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip()}
    dependencies={}
    for name in ('WAN_SOURCE_ROOT','HMR2_SOURCE_ROOT','DIVIBODY_SOURCE_ROOT'):
        location=os.environ.get(name)
        if location:
            path=Path(location).resolve()
            dependencies[name]={'path':str(path),'git_commit':command(['git','-C',str(path),'rev-parse','HEAD'])}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
        'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,
        'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu':command(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,utilization.gpu','--format=csv']),
        'packages':sorted({(d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name')}),
        'dependency_sources':dependencies,
        'note':'Observed environment inventory, not an independently validated installation lockfile.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({'output':str(args.output),'packages':len(result['packages'])}))

if __name__=='__main__':
    main()
