"""Translate a V31 JSON config; --dry-run checks arguments and asset paths without torch."""
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
        'coco-train-manifest','resume-checkpoint'}
    cli=[]
    for key,value in config.items():
        if not isinstance(key,str) or not key or key.startswith('-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in key):
            raise ValueError('Invalid CLI option name')
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
    from divibody_training.v31_training_contract import parse_training_args,inspect_assets
    args=parse_training_args(cli)
    audit=inspect_assets(args,fingerprint=False)
    command=[sys.executable,'-u','-m','divibody_training.v28d_gpu_entry',
             'divibody_training.train_spatiotemporal_v31',*cli]
    if a.dry_run:
        print(json.dumps(dict(dry_run=True,command=command,
            normalized_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            asset_preflight=audit),indent=2))
        raise SystemExit(0 if audit['passed'] else 2)
    if not audit['passed']:
        print(json.dumps(dict(asset_preflight=audit),indent=2),file=sys.stderr)
        raise SystemExit(2)
    completed=subprocess.run(command,cwd=root)
    raise SystemExit(completed.returncode)


if __name__=='__main__':
    main()
