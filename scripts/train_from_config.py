"""Translate a portable JSON config into the preserved V30 training CLI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    config=json.loads(a.config.read_text(encoding='utf-8'))
    if not isinstance(config,dict) or not config:raise ValueError('Nonempty training configuration required')
    path_keys={'train-vace-cache','train-hmr-sidecar','val-vace-cache','val-hmr-sidecar',
        'pose-train-vace-dir','pose-train-hmr-dir','pose-val-vace-dir','pose-val-hmr-dir',
        'init-checkpoint','smpl-model-root','joint-regressor','output-dir','validation-manifest',
        'coco-train-manifest','resume-checkpoint'}
    command=[sys.executable,'-u','-m','divibody_training.v28d_gpu_entry',
        'divibody_training.train_spatiotemporal_v30_v3']
    for key,value in config.items():
        if not isinstance(key,str) or not key or key.startswith('-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in key):
            raise ValueError('Invalid CLI option name')
        if value is None or value is False:continue
        command.append('--'+key)
        if value is True:continue
        if not isinstance(value,(str,int,float)):raise ValueError('Scalar CLI value required: '+key)
        if key in path_keys:
            path=Path(value).expanduser()
            if not path.is_absolute():path=root/path
            value=str(path)
        command.append(str(value))
    # Preserved parser enforces valid options and existing-run/resume contracts.
    completed=subprocess.run(command,cwd=root)
    raise SystemExit(completed.returncode)


if __name__=='__main__':main()
