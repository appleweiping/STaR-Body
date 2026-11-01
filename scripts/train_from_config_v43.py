"""C11 (v43): translate a JSON config into train_target60_v43; --dry-run checks arguments and asset paths without torch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true',help='Validate CLI and local asset paths without importing torch or starting a run')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    if not isinstance(config,dict) or not config:
        raise ValueError('Nonempty training configuration required')
    path_keys={'train-vace-cache','train-hmr-sidecar','val-vace-cache','val-hmr-sidecar',
        'pose-train-vace-dir','pose-train-hmr-dir','pose-val-vace-dir','pose-val-hmr-dir',
        'init-checkpoint','smpl-model-root','joint-regressor','output-dir','validation-manifest',
        'coco-train-manifest','resume-checkpoint','motion-pretrained',
        'train-track-prototypes','val-track-prototypes','initial-reference-report','parent-validation-report',
        'train-external-keypoints','val-external-keypoints'}
    cli=[]
    for key,value in config.items():
        if not isinstance(key,str) or not key or key.startswith('-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in key):
            raise ValueError('Invalid CLI option name')
        if key=='v43-router-identity' and value is False:
            key,value='v43-router-no-identity',True  # store_false flag spelled positively in JSON
        if value is None or value is False:
            continue
        if key=='local-radius-schedule' and isinstance(value,list):
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in value):
                raise ValueError('Local radius schedule requires a list of numbers')
            value=','.join(str(v) for v in value)
        cli.append('--'+key)
        if value is True:
            continue
        if not isinstance(value,(str,int,float)):
            raise ValueError('Scalar CLI value required: '+key)
        if key in path_keys:
            path=Path(value).expanduser()
            if not path.is_absolute():
                path=root/path
            value=str(path.resolve())
        cli.append(str(value))
    sys.path.insert(0,str(root/'video_VACE'))
    from divibody_training.target60_contract_v43 import parse_training_args,inspect_assets
    args=parse_training_args(cli)
    audit=inspect_assets(args,fingerprint=False)
    command=[sys.executable,'-u','-m','divibody_training.v28d_gpu_entry',
             'divibody_training.train_target60_v43',*cli]
    if a.dry_run:
        print(json.dumps(dict(dry_run=True,command=command,
            normalized_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            asset_preflight=audit),indent=2))
        raise SystemExit(0 if audit['passed'] else 2)
    if not audit['passed']:
        print(json.dumps(dict(asset_preflight=audit),indent=2),file=sys.stderr)
        raise SystemExit(2)
    # C11: sibling lock '<output-dir>.running' so the matrix runner never moves a live run aside.
    import os,socket
    output_dir=Path(args.output_dir).resolve();lock=output_dir.with_name(output_dir.name+'.running')
    if lock.exists():
        try:
            info=json.loads(lock.read_text());os.kill(int(info['pid']),0)
            if info.get('host')==socket.gethostname():
                print(json.dumps(dict(error='run already active',lock=str(lock),**info)),file=sys.stderr);raise SystemExit(3)
        except (ProcessLookupError,ValueError,KeyError,json.JSONDecodeError):
            pass  # stale lock from a dead process
    lock.parent.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),output_dir=str(output_dir))))
    try:
        completed=subprocess.run(command,cwd=root)
    finally:
        lock.unlink(missing_ok=True)
    raise SystemExit(completed.returncode)


if __name__=='__main__':
    main()
