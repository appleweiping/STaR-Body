"""Launch the original V30 through additive, provenance-bound rebuild infrastructure."""
import argparse
import json
import os
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
        'coco-train-manifest','resume-checkpoint','baseline-report'}
    cli=[]
    for key,value in config.items():
        if not isinstance(key,str) or not key or key.startswith('-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in key):
            raise ValueError('Invalid CLI option name')
        if key in {'frame-shape-delta-scale','local-radius-schedule','deduplicate-shape-consistency'}:
            raise ValueError('V31 model/loss switches are unsupported by the original V30 rebuild: '+key)
        if value is None or value is False:
            continue
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
    from divibody_training.v30_rebuild_training_contract import parse_training_args,inspect_assets
    args=parse_training_args(cli)
    audit=inspect_assets(args,fingerprint=False)
    command=[sys.executable,'-u','-m','divibody_training.v28d_gpu_entry',
             'divibody_training.train_spatiotemporal_v30_rebuild',*cli]
    if a.dry_run:
        print(json.dumps(dict(dry_run=True,command=command,
            normalized_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            asset_preflight=audit),indent=2))
        raise SystemExit(0 if audit['passed'] else 2)
    if not audit['passed']:
        print(json.dumps(dict(asset_preflight=audit),indent=2),file=sys.stderr)
        raise SystemExit(2)
    env=os.environ.copy()
    paths=[str(root/'video_VACE'),str(root/'video_VACE/divibody_training'),str(root)]
    paths.extend(env[key] for key in ('WAN_SOURCE_ROOT','HMR2_SOURCE_ROOT') if env.get(key))
    if env.get('PYTHONPATH'):
        paths.extend(env['PYTHONPATH'].split(os.pathsep))
    shadow=(root/'video_VACE/vace/models').resolve()
    paths=[path for path in paths if path and Path(path).resolve()!=shadow]
    env['PYTHONPATH']=os.pathsep.join(dict.fromkeys(paths))
    env['PYTHONDONTWRITEBYTECODE']='1'
    completed=subprocess.run(command,cwd=root,env=env)
    raise SystemExit(completed.returncode)


if __name__=='__main__':
    main()
