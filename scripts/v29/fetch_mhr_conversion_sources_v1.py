"""Fetch pinned official conversion code/assets into a new isolated directory."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def get(url,limit):
    request=urllib.request.Request(url,headers={'User-Agent':'v29-research-source-audit'})
    with urllib.request.urlopen(request,timeout=30) as r:
        data=r.read(limit+1)
    if len(data)>limit:raise ValueError('Finite download limit exceeded: '+url)
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    commit=json.loads(get('https://api.github.com/repos/facebookresearch/MHR/commits/main',2*1024**2))['sha']
    tree=json.loads(get(f'https://api.github.com/repos/facebookresearch/MHR/git/trees/{commit}?recursive=1',8*1024**2))
    if tree.get('truncated'):raise ValueError('Incomplete official repository tree')
    items=[x for x in tree['tree'] if x['type']=='blob' and (
        x['path'].startswith('tools/mhr_smpl_conversion/') or x['path'] in ('LICENSE','mhr/mhr.py','mhr/__init__.py'))]
    total=sum(x.get('size',0) for x in items)
    if total>64*1024**2:raise ValueError('Conversion sources exceed finite 64 MiB budget')
    rows=[]
    try:
        for item in items:
            name=item['path'];target=(a.output/name).resolve()
            if not target.is_relative_to(a.output.resolve()):raise ValueError('Unsafe official asset path')
            data=get(f'https://raw.githubusercontent.com/facebookresearch/MHR/{commit}/{name}',32*1024**2)
            if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=item['sha']:
                raise ValueError('Official Git blob fingerprint differs: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
            rows.append(dict(path=name,bytes=len(data),git_blob_sha1=item['sha'],sha256=hashlib.sha256(data).hexdigest(),
                lfs_pointer=data.startswith(b'version https://git-lfs.github.com/spec/v1')))
            print('Fetched',name,len(data),flush=True)
        result=dict(status='complete',repository='https://github.com/facebookresearch/MHR',commit=commit,
            files=rows,dependencies_installed=False,conversion_not_yet_run=True)
    except Exception as error:
        result=dict(status='failed',commit=commit,files=rows,error=repr(error),dependencies_installed=False)
        raise
    finally:
        if 'result' in locals():
            with (a.output/'source_manifest.json').open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
