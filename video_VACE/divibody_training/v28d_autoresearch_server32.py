"""Copied V28D suite for server32 with bounded shared-GPU execution."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from divibody_training.v28d_experiment_utils import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('DIVIBODY_SOURCE_ROOT', str(ROOT.parent/'divibody_vace')))


def default_options():
    runs = SOURCE/'training_runs'
    collection = runs/'ram_posetrack_sharded_v2_full_20260902'
    return {
        'train-vace-cache': str(runs/'continuous_vace_v5_full_zero_text/features_train.pt'),
        'train-hmr-sidecar': str(runs/'hmr2a_anchor_v11_raw25/features_train.pt'),
        'val-vace-cache': str(runs/'continuous_vace_v6_stride4_zero_text/features_validation.pt'),
        'val-hmr-sidecar': str(runs/'hmr2a_anchor_v11_raw25/features_validation_stride4.pt'),
        'pose-train-vace-dir': str(collection/'train'), 'pose-train-hmr-dir': str(collection/'hmr_train'),
        'pose-val-vace-dir': str(collection/'val'), 'pose-val-hmr-dir': str(collection/'hmr_val'),
        'init-checkpoint': str(runs/'autoresearch_v27_geomcoupled_g2e5_p05_c02_20260926/run/best.pt'),
        'smpl-model-root': str(SOURCE/'checkpoints/4DHumans/extracted/data/smpl'),
        'joint-regressor': str(SOURCE/'body_models/J_regressor_h36m.npy'),
        'coco-train-manifest': str(ROOT/'training_runs/coco_train_v28_10k_cache/manifest_coco_train_mixed.json'),
        'device': 'cuda:0', 'batch-size': 4, 'pose-batch-size': 8, 'val-batch-size': 16,
        'num-workers': 0, 'seed': 2026, 'geometry-lr': 2e-5, 'local-lr': 1e-5,
        'iterations': 2, 'diagnostic-every': 100, 'max-steps': 1000, 'eval-every': 200,
        'coco-warmup-steps': 0,
    }


def load_options(config):
    options = default_options()
    if config:
        custom = json.loads(Path(config).read_text())
        if not isinstance(custom, dict):
            raise ValueError('Configuration must be one JSON object')
        options.update({key.replace('_', '-'): value for key, value in custom.items()})
    return options


def doctor(options):
    assets = {}
    for name, value in options.items():
        if name.endswith(('-cache', '-sidecar', '-dir', '-root', '-manifest', '-checkpoint')) or name == 'joint-regressor':
            p = Path(value)
            assets[name] = dict(path=str(p), exists=p.exists())
    manifest = Path(options['coco-train-manifest'])
    if manifest.exists():
        data = json.loads(manifest.read_text())
        paths = [manifest.parent/p for p in data.get('files', [])]
        assets['coco-train-manifest'].update(samples=len(paths), missing=sum(not p.is_file() for p in paths),
                                          sha256=sha256(manifest), split=data.get('split'))
    return dict(assets=assets, ready=all(item['exists'] and not item.get('missing',0) for item in assets.values()))


def run_suite(args, options):
    if Path(args.run_name).name != args.run_name or args.run_name in ('.','..'):
        raise ValueError('Use a single new run directory name')
    out = ROOT/'training_runs'/args.run_name
    out.mkdir(exist_ok=False)
    assets = doctor(options)
    atomic_json(out/'assets.json', assets)
    if not assets['ready']:
        raise FileNotFoundError('Required assets missing; see assets.json')
    if not args.allow_partial_validation and 'validation-manifest' not in options:
        raise ValueError('Supply audited validation-manifest or explicit development validation flag')
    variants = [('pt002_coco0',.02,0.), ('pt002_coco1',.02,1.),
                ('pt1_coco0',1.,0.), ('pt1_coco1',1.,1.)]
    if args.smoke:
        options.update({'max-steps':2,'eval-every':2,'max-train-windows':8,
                        'max-val-windows':8,'max-pose-train-windows':16,'max-pose-val-windows':8,
                        'diagnostic-every':1})
    lock_path = ROOT/'training_runs'/f'.v28d_gpu{args.gpu}.lock'
    commands = []
    for label, pose_weight, coco_weight in variants:
        settings = options | {'output-dir': str(out/label), 'pose-2d-weight':pose_weight,'coco-weight':coco_weight}
        command = [sys.executable,'-u','-m','divibody_training.v28d_gpu_entry',
                   'divibody_training.train_parts_v28d_balanced']
        for key, value in settings.items():
            if value is True:
                command.append('--'+key)
            elif value is not False and value is not None:
                command.extend(['--'+key,str(value)])
        if args.allow_partial_validation:
            command.append('--allow-partial-validation')
        commands.append(dict(label=label,command=command,settings=settings))
    atomic_json(out/'suite.json', dict(variants=commands,gpu=args.gpu,shared_seed=options['seed'],
                smoke=args.smoke,development_validation=args.allow_partial_validation,
                source_hashes=json.loads((ROOT/'v28d_upgrade_source_hashes.json').read_text())))
    if args.plan_only:
        print(json.dumps(dict(plan=str(out/'suite.json')))); return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    env['V28D_MEMORY_LIMIT_GIB']='10'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'video_VACE'),str(ROOT/'video_VACE/divibody_training'),
        str(SOURCE/'dependencies/Wan2.1'),str(ROOT/'video_VACE/vace/models'),
        str(SOURCE/'dependencies/4D-Humans'),str(ROOT/'runtime_packages'),
        str(SOURCE.parent.parent/'Backup/4D-Humans'),
        env.get('PYTHONPATH','')])
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            samples = []
            for _ in range(2):
                value = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                    '--query-gpu=memory.used,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True)
                memory,util,free = map(int,value.strip().split(',')); samples.append((memory,util,free))
                if len(samples)==1:
                    atomic_json(out/'status.json',dict(status='waiting_for_gpu_resources',gpu=args.gpu,snapshot=samples,
                                                      shared_gpu=args.allow_shared_gpu,memory_limit_gib=10))
                    time.sleep(30)
            if all(free >= 12288 and (args.allow_shared_gpu or (memory < 100 and util < 5))
                   for memory,util,free in samples):
                break
            time.sleep(30)
        for variant in commands:
            atomic_json(out/'status.json',dict(status='running',variant=variant['label'],gpu=args.gpu))
            with (out/(variant['label']+'.log')).open('x') as log:
                process=subprocess.Popen(variant['command'],cwd=ROOT/'video_VACE',env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(out/'active_process.json',dict(pid=process.pid,variant=variant['label']))
                code=process.wait()
            if code:
                atomic_json(out/'status.json',dict(status='failed',variant=variant['label'],exit_code=code))
                raise RuntimeError(f"Variant {variant['label']} failed ({code})")
        atomic_json(out/'status.json',dict(status='complete',gpu=args.gpu))
    report(out)


def report(out):
    rows=[]
    for config in sorted(Path(out).glob('*/run_config.json')):
        run=config.parent
        complete=json.loads((run/'completion.json').read_text()) if (run/'completion.json').exists() else {}
        metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()] if (run/'metrics.jsonl').exists() else []
        for row in metrics:
            if row.get('variant')=='student':
                rows.append(dict(run=run.name,step=row['step'],mpjpe=row['validation']['mpjpe_mm'],
                    pve=row['validation']['pve_mm'],pck05=row['pose_validation']['projected_PCKn@0.05'],
                    pck10=row['pose_validation']['projected_PCKn@0.10'],accepted=row['accepted'],
                    reasons=row['selection']['reasons'],coverage=row['coverage'],
                    baseline_retained=complete.get('baseline_retained')))
    atomic_json(Path(out)/'comparison.json',rows)
    lines=['# V28D 实验结果','', '| 实验 | 步数 | MPJPE mm ↓ | PVE mm ↓ | 投影 PCK05 ↑ | 投影 PCK10 ↑ | 通过选模 | 拒绝原因 |',
           '|---|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        lines.append(f"| {row['run']} | {row['step']} | {row['mpjpe']:.4f} | {row['pve']:.4f} | {row['pck05']:.6f} | {row['pck10']:.6f} | {row['accepted']} | {', '.join(row['reasons'])} |")
    lines += ['', 'baseline_retained=true 表示 best.pt 仍为初始权重。覆盖率只统计实际参与优化的样本。',
              '开发验证与官方测试需分别标注；本表不构成与论文中不同协议结果的比较。']
    (Path(out)/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(report=str(Path(out)/'REPORT.md'),rows=len(rows))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['doctor','suite','report'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--run-name')
    parser.add_argument('--gpu',type=int,choices=range(4),default=3)
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--allow-partial-validation',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args(); options=load_options(args.config)
    if args.action=='doctor':
        print(json.dumps(doctor(options),indent=2)); return
    if not args.run_name:
        parser.error('--run-name is required')
    if args.action=='report':
        report(ROOT/'training_runs'/args.run_name)
    else:
        run_suite(args,options)


if __name__=='__main__':
    main()
