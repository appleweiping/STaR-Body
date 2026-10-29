"""V34 step-budgeted training with versioned losses and resume provenance."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from divibody_training.target60_runtime_v34 import forward_batch, pose_losses
from divibody_training.v29_experiment_utils import (
    MapStream, PoseStream, atomic_json, gradients, subtract, gradient_report,
    sha256)
from divibody_training.target60_policy_v34 import selection_report
from divibody_training.target60_boundary_v34 import build_boundary_pool,boundary_training_loss
from divibody_training.target60_partition_v34 import geometric_losses,audit_boundary_partition as audit_partition


def execute_run(model, optimizer, ema, train_loader, pose_loader, coco_loader,
                models, official_smpl, regressor, evaluate, args, device, metadata):
    out = args.output_dir
    geometry_stream = MapStream(train_loader.dataset, args.batch_size, args.seed)
    boundary_pool=build_boundary_pool(train_loader.dataset)
    partition=audit_partition(train_loader.dataset,boundary_pool)
    args.motion_train_windows=len(train_loader.dataset)
    args.motion_canonical_within_triplets=partition['canonical_within_triplets']
    if not boundary_pool.triplets:raise ValueError('No train boundary triplets')
    atomic_json(out/'boundary_partition.json',partition)
    atomic_json(out/'boundary_audit.json',boundary_pool.audit|dict(identity_sha256=boundary_pool.identity_sha256))
    pose_stream = PoseStream(pose_loader) if pose_loader is not None else None
    coco_stream = MapStream(coco_loader.dataset, args.batch_size, args.seed + 2) if coco_loader else None
    streams = {'geometry': geometry_stream}
    if pose_stream is not None:
        streams['pose'] = pose_stream
    if coco_stream:
        streams['coco'] = coco_stream
    manifest_hash = sha256(args.coco_train_manifest) if args.coco_train_manifest else None
    invariants = {key: value for key, value in metadata['args'].items()
                  if key not in ('resume_checkpoint', 'stop_after', 'output_dir', 'num_workers')}
    invariants.update(coco_manifest_hash=manifest_hash,
                      init_sha256=metadata['source_sha256'],
                      validation_reference=metadata['coverage'].get('manifest_sha256'),
                      engine_sha256=sha256(__file__),
                      boundary_sha256=sha256(Path(__file__).with_name('target60_boundary_v34.py')),
                      boundary_pool_sha256=boundary_pool.identity_sha256,
                      partition_code_sha256=sha256(Path(__file__).with_name('target60_partition_v34.py')),
                      policy_sha256=sha256(Path(__file__).with_name('target60_policy_v34.py')),
                      evaluator_sha256=sha256(Path(__file__).with_name('target60_validation_v34.py')),
                      utils_sha256=sha256(Path(__file__).with_name('v29_experiment_utils.py')),
                      source_files_sha256=metadata['source_files_sha256'],
                      asset_manifest=metadata['asset_manifest'],
                      cache_hash_policy=metadata['cache_hash_policy'],
                      environment=metadata['environment'],context_audit=metadata['context_audit'])
    step = best_step = stale = 0
    best = None
    baseline_geometry = baseline_pose = None
    validation = pose_validation = None
    started = time.monotonic()
    history = []

    def record(filename, row):
        with (out / filename).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def coverage():
        return {name: stream.coverage() for name, stream in streams.items()}|dict(boundary=dict(
            pool=len(boundary_pool.triplets),optimized_triplet_draws=step*args.gradient_accumulation*args.boundary_triplets_per_step,
            unique_optimized=min(step*args.gradient_accumulation*args.boundary_triplets_per_step,len(boundary_pool.triplets)),
            fraction=min(1.,step*args.gradient_accumulation*args.boundary_triplets_per_step/len(boundary_pool.triplets)),
            supervised_split='train'))

    def save(filename, network, variant, resumable=False):
        state = metadata | dict(model=network.state_dict(), step=step, epoch=geometry_stream.cycle + 1 if step else 0,
                                variant=variant, validation=validation, pose_validation=pose_validation,
                                selection_metric='Target60 minimax joint errors with initial V30 PVE and acceleration guards',
                                initial_fallback_only=step==0)
        if resumable:
            state.update(optimizer=optimizer.state_dict(), ema_model=ema.model.state_dict(),
                         ema_updates=ema.updates, stream_states={k: v.state_dict() for k, v in streams.items()},
                         baseline_geometry=baseline_geometry, baseline_pose=baseline_pose,
                         best_score=best, best_step=best_step, stale_evaluations=stale,
                         invariant_config=invariants, history=history,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        temporary = out / (filename + '.tmp')
        torch.save(state, temporary)
        temporary.replace(out / filename)

    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        if state.get('invariant_config') != invariants:
            raise ValueError('Resume configuration/data/initial checkpoint differs from the saved run')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        ema.model.load_state_dict(state['ema_model'], strict=True)
        ema.updates = state['ema_updates']
        for name, stream in streams.items():
            stream.load_state_dict(state['stream_states'][name])
        step, best_step, best, stale = state['step'], state['best_step'], state['best_score'], state['stale_evaluations']
        baseline_geometry, baseline_pose = state['baseline_geometry'], state['baseline_pose']
        validation, pose_validation, history = state['validation'], state['pose_validation'], state['history']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        # Preserve interrupted logs, then discard uncheckpointed records before replay.
        for filename in ('metrics.jsonl','diagnostics.jsonl','samples.jsonl'):
            path=out/filename
            if not path.exists(): continue
            retained=[]
            for line in path.read_text().splitlines():
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('step',0)<=step: retained.append(line)
            corrected=''.join(line+'\n' for line in retained)
            if corrected!=path.read_text():
                path.rename(out/(filename+f'.before_resume_{time.time_ns()}'))
                path.write_text(corrected)
        record('events.jsonl', dict(event='resumed', step=step, coverage=coverage()))
    else:
        model.baseline_only=True
        try:
            baseline_geometry,baseline_pose=evaluate(model)
        finally:
            model.baseline_only=False
        if args.derive_acceleration_ceiling:
            args.acceleration_ceiling_mm=baseline_geometry['acceleration_error_mm_per_frame2']
        validation,pose_validation=evaluate(model)
        if args.initial_reference_report and not metadata['development_only']:
            reference=json.loads(args.initial_reference_report.read_text())['metrics']
            for key in ('mpjpe_mm','pa_mpjpe_mm','pve_mm','acceleration_error_mm_per_frame2'):
                if abs(baseline_geometry[key]-reference[key])>args.initial_reproduction_atol_mm:
                    raise ValueError('Initial V30 reproduction failed: '+key)
            if not metadata['development_only'] and baseline_geometry['frames']!=reference['frames']:
                raise ValueError('Initial V30 coverage differs')
        for key in ('mpjpe_mm','pa_mpjpe_mm','pve_mm','acceleration_error_mm_per_frame2'):
            if abs(validation[key]-baseline_geometry[key])>args.initial_reproduction_atol_mm:
                raise ValueError('Zero-init adapter changed native V30 baseline: '+key)
        atomic_json(out/'initial_v30_baseline.json',dict(geometry=baseline_geometry,pose=baseline_pose,
            acceleration_ceiling_mm=args.acceleration_ceiling_mm,force_anchor=True))
        best = -max(validation['mpjpe_mm']/args.target_mpjpe_mm,validation['pa_mpjpe_mm']/args.target_pa_mpjpe_mm)
        save('best.pt', model, 'initial')
        record('metrics.jsonl',dict(step=0,variant='initial',validation=validation,
            pose_validation=pose_validation,accepted=False,initial_fallback_only=True,
            initial_v30_baseline=baseline_geometry,
            development_only=metadata['development_only']))

    if args.derive_acceleration_ceiling:
        args.acceleration_ceiling_mm=baseline_geometry['acceleration_error_mm_per_frame2']

    if args.eval_only:
        atomic_json(out / 'completion.json', dict(completed_steps=0, eval_only=True,
                    complete=True, development_only=metadata['development_only']))
        return

    def evaluate_step():
        nonlocal validation, pose_validation, best, best_step, stale
        improved = False
        for variant, network in (('student', model), ('ema', ema.model)):
            validation, pose_validation = evaluate(network)
            selection = selection_report(validation, pose_validation, baseline_geometry, baseline_pose, args, best)
            row = dict(step=step, variant=variant, validation=validation, pose_validation=pose_validation,
                       selection=selection, accepted=selection['accepted'], coverage=coverage(),
                       development_only=metadata['development_only'])
            history.append(row)
            record('metrics.jsonl', row)
            if selection['accepted']:
                best, best_step, improved = selection['score'], step, True
                save('best.pt', network, variant)
        stale = 0 if improved else stale + 1
        # Latest must describe student weights, even if EMA was evaluated last.
        student = history[-2]
        validation, pose_validation = student['validation'], student['pose_validation']
        save('latest.pt', model, 'student', resumable=True)
        atomic_json(out / 'progress.json', dict(step=step, target_steps=args.max_steps,
                    best_step=best_step, initial_checkpoint_retained=best_step == 0, complete=False,
                    coverage=coverage(), elapsed_seconds=time.monotonic() - started))

    from divibody_training.target60_optim_v34 import step_lr
    while step < args.max_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_lr(optimizer,step,args.max_steps,args.warmup_steps)
        microbatches=[]
        remaining=len(geometry_stream.order)-geometry_stream.cursor
        cursor=geometry_stream.cursor
        windows=0
        for _ in range(args.gradient_accumulation):
            if cursor==len(geometry_stream.order):cursor=0
            count=min(args.batch_size,len(geometry_stream.order)-cursor);windows+=count;cursor+=count
        diagnose=step==0 or (args.diagnostic_every>0 and step%args.diagnostic_every==0)
        values=[];sampled=[];task_vectors={}
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        for micro in range(args.gradient_accumulation):
            batch,ids=geometry_stream.next_batch();microbatches.append(ids)
            weight=len(ids)/windows
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(args.seed+(step*args.gradient_accumulation+micro)*3)
                pred=forward_batch(model,batch,device,args.amp_dtype)
                losses=geometric_losses(pred,batch,models,regressor,args,device)
                weighted=losses['total']*weight
                if not torch.isfinite(weighted):raise FloatingPointError('Nonfinite geometry loss')
                weighted.backward()
            if diagnose:
                current=gradients(optimizer)
            boundary_step=step*args.gradient_accumulation+micro
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(args.seed+boundary_step*7+6)
                boundary_loss,info=boundary_training_loss(model,train_loader.dataset,boundary_pool,models,regressor,
                    device,args.amp_dtype,step=boundary_step,seed=args.seed,
                    triplets_per_step=args.boundary_triplets_per_step)
                boundary=boundary_loss*args.boundary_weight/args.gradient_accumulation
                if not torch.isfinite(boundary):raise FloatingPointError('Nonfinite boundary loss')
                boundary.backward()
            if diagnose:
                after=gradients(optimizer)
                bv=subtract(after,current)
                if micro==0:before={k:torch.zeros_like(v) for k,v in current.items()}
                gv=subtract(current,before)
                for name,vector in (('geometry',gv),('boundary',bv)):
                    if name not in task_vectors:task_vectors[name]=vector
                    else:task_vectors[name]={k:task_vectors[name][k]+v for k,v in vector.items()}
                before=after
            values.append(dict(micro=micro,windows=len(ids),geometry_loss=float(losses['total'].detach()),
                weighted_geometry=float(weighted.detach()),weighted_boundary=float(boundary.detach()),
                pa_auxiliary_m=float(losses['pa_auxiliary_m'].detach()),
                refinement_regularization=float(losses['refinement_regularization'].detach())))
            sampled.append(dict(geometry=ids,boundary=info))
            # Release the forward graph before materializing the next microbatch.
            del pred,losses,weighted,boundary,boundary_loss,batch
        if step<args.adapter_warmup_steps:
            for name,parameter in model.named_parameters():
                if name.startswith('backbone.'):parameter.grad=None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
        optimizer.step();ema.update(model)
        for ids in microbatches:geometry_stream.record(ids)
        step+=1
        record('samples.jsonl',dict(step=step,microbatches=sampled,effective_windows=windows))
        if diagnose:
            record('diagnostics.jsonl',dict(step=step,micro_losses=values,
                weighted_gradient_groups=gradient_report(task_vectors),grad_norm=float(norm),
                optimizer_steps=step,microbatch_draws=step*args.gradient_accumulation,
                coverage=coverage()))
        should_evaluate=step==args.max_steps or step%args.eval_every==0 or (args.stop_after and step>=args.stop_after)
        if should_evaluate:evaluate_step()
        if args.stop_after and step>=args.stop_after and step<args.max_steps:
            atomic_json(out/'paused.json',dict(step=step,reason='stop_after',resumable=True))
            return
    atomic_json(out / 'progress.json', dict(step=step, target_steps=args.max_steps,
                best_step=best_step, initial_checkpoint_retained=best_step == 0, complete=True, coverage=coverage()))
    atomic_json(out / 'completion.json', dict(completed_steps=step, complete=True,
                best_step=best_step, initial_checkpoint_retained=best_step == 0, coverage=coverage(),
                development_only=metadata['development_only'], elapsed_seconds=time.monotonic() - started))
    (out / 'paused.json').unlink(missing_ok=True)
