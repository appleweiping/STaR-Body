"""Step-budgeted V28D training; frozen model architecture is reused unchanged."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from divibody_training.ram_parts_runtime_v29 import forward_batch, geometric_losses, pose_losses
from divibody_training.v29_experiment_utils import (
    MapStream, PoseStream, atomic_json, gradients, subtract, gradient_report,
    sha256)
from divibody_training.v29_geometry_policy_stable import selection_report, merge_auxiliary
from divibody_training.v29_boundary_training import build_boundary_pool,boundary_training_loss


def execute_run(model, optimizer, ema, train_loader, pose_loader, coco_loader,
                models, official_smpl, regressor, evaluate, args, device, metadata):
    out = args.output_dir
    geometry_stream = MapStream(train_loader.dataset, args.batch_size, args.seed)
    boundary_pool=build_boundary_pool(train_loader.dataset)
    if not boundary_pool.triplets:raise ValueError('No train boundary triplets')
    atomic_json(out/'boundary_audit.json',boundary_pool.audit|dict(identity_sha256=boundary_pool.identity_sha256))
    pose_stream = PoseStream(pose_loader)
    coco_stream = MapStream(coco_loader.dataset, args.batch_size, args.seed + 2) if coco_loader else None
    streams = {'geometry': geometry_stream, 'pose': pose_stream}
    if coco_stream:
        streams['coco'] = coco_stream
    manifest_hash = sha256(args.coco_train_manifest) if args.coco_train_manifest else None
    invariants = {key: value for key, value in metadata['args'].items()
                  if key not in ('resume_checkpoint', 'stop_after', 'output_dir', 'num_workers')}
    invariants.update(coco_manifest_hash=manifest_hash,
                      init_sha256=metadata['source_sha256'],
                      validation_reference=metadata['coverage'].get('manifest_sha256'),
                      engine_sha256=sha256(__file__),
                      boundary_sha256=sha256(Path(__file__).with_name('v29_boundary_training.py')),
                      boundary_pool_sha256=boundary_pool.identity_sha256,
                      policy_sha256=sha256(Path(__file__).with_name('v29_geometry_policy_stable.py')),
                      evaluator_sha256=sha256(Path(__file__).with_name('v29_sequence_evaluation_report.py')),
                      utils_sha256=sha256(Path(__file__).with_name('v29_experiment_utils.py')))
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
            pool=len(boundary_pool.triplets),optimized_triplet_draws=step*args.boundary_triplets_per_step,
            unique_optimized=min(step*args.boundary_triplets_per_step,len(boundary_pool.triplets)),
            fraction=min(1.,step*args.boundary_triplets_per_step/len(boundary_pool.triplets)),
            supervised_split='train'))

    def save(filename, network, variant, resumable=False):
        state = metadata | dict(model=network.state_dict(), step=step, epoch=geometry_stream.cycle + 1 if step else 0,
                                variant=variant, validation=validation, pose_validation=pose_validation,
                                selection_metric='strict 3D improvement with frozen validation acceleration ceiling')
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
        validation, pose_validation = baseline_geometry, baseline_pose = evaluate(model)
        best = -3.0
        save('best.pt', model, 'initial')
        record('metrics.jsonl', dict(step=0, variant='initial', validation=validation,
                                    pose_validation=pose_validation, accepted=True, initial_fallback_only=True,
                                    satisfies_acceleration_ceiling=validation['acceleration_error_mm_per_frame2']<=args.acceleration_ceiling_mm,
                                    development_only=metadata['development_only']))

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
                    best_step=best_step, baseline_retained=best_step == 0, complete=False,
                    coverage=coverage(), elapsed_seconds=time.monotonic() - started))

    from divibody_training.parts_training_utils_v29 import step_lr
    while step < args.max_steps:
        model.train()
        batch, geometry_ids = geometry_stream.next_batch()
        auxiliary_active=step >= args.geometry_only_steps
        pose_batch = pose_stream.next_batch() if auxiliary_active and args.pose_2d_weight else None
        coco_batch, coco_ids = coco_stream.next_batch() if coco_stream and auxiliary_active and args.coco_weight else (None,None)
        diagnose = step == 0 or (args.diagnostic_every > 0 and step % args.diagnostic_every == 0)
        optimizer.zero_grad(set_to_none=True)
        step_lr(optimizer, step, args.max_steps, args.warmup_steps)
        task_vectors = {}
        values = {}

        def backward_task(name, sample, loss_function, weight, task_id):
            # Each task has its own deterministic RNG, independent of which branches are enabled.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(args.seed + step * 3 + task_id)
                prediction = forward_batch(model, sample, device, args.amp_dtype)
                losses = loss_function(prediction, sample)
                total = losses['total'] if name == 'geometry' else losses['pose_2d']
                weighted = total * weight
                if not torch.isfinite(weighted):
                    raise FloatingPointError('Nonfinite ' + name + ' loss')
                weighted.backward()
                values[name] = dict(raw_loss=float(total.detach()), weight=weight,
                                    weighted_loss=float(weighted.detach()))
                if name == 'geometry':
                    values[name]['weighted_update_regularization'] = float(losses['update_regularization'].detach()) * args.update_regularization
            if diagnose:
                cumulative = gradients(optimizer)
                previous = {k: torch.zeros_like(v) for k, v in cumulative.items()}
                for earlier_name,earlier in task_vectors.items():
                    if name!='geometry' and earlier_name in ('geometry','boundary'): continue
                    previous = {k: previous[k] + earlier[k] for k in previous}
                task_vectors[name] = subtract(cumulative, previous)

        backward_task('geometry', batch,
                      lambda p, b: geometric_losses(p, b, models, regressor, args, device), 1., 0)
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed+step*7+6)
            boundary_loss,boundary_info=boundary_training_loss(model,train_loader.dataset,boundary_pool,models,regressor,
                device,args.amp_dtype,step=step,seed=args.seed,triplets_per_step=args.boundary_triplets_per_step)
            weighted_boundary=boundary_loss*args.boundary_weight
            weighted_boundary.backward()
        values['boundary']=dict(raw_loss=float(boundary_loss.detach()),weight=args.boundary_weight,
                                weighted_loss=float(weighted_boundary.detach()))
        if diagnose:task_vectors['boundary']=subtract(gradients(optimizer),task_vectors['geometry'])
        protected={p:p.grad.detach().clone() if p.grad is not None else None for group in optimizer.param_groups for p in group['params']}
        optimizer.zero_grad(set_to_none=True)
        ramp=min(1.,max(0.,(step-args.geometry_only_steps+1)/max(1,args.auxiliary_ramp_steps)))
        pose_weight=args.pose_2d_weight*ramp
        if pose_batch is not None:
            backward_task('posetrack',pose_batch,
                          lambda p,b:pose_losses(p,b,official_smpl,args,device),pose_weight,1)
        coco_weight = args.coco_weight * ramp
        if coco_batch is not None and coco_weight > 0:
            backward_task('coco', coco_batch,
                          lambda p, b: pose_losses(p, b, official_smpl, args, device), coco_weight, 2)
        auxiliary={p:p.grad.detach().clone() if p.grad is not None else None for p in protected}
        policy=merge_auxiliary(optimizer,protected,auxiliary,args.auxiliary_grad_ratio)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step(); ema.update(model)
        geometry_stream.record(geometry_ids)
        if pose_batch is not None: pose_stream.record(pose_batch)
        if coco_stream and coco_weight > 0:
            coco_stream.record(coco_ids)
        step += 1
        with (out/'samples.jsonl').open('a') as sample_log:
            sample_log.write(json.dumps(dict(step=step,geometry=geometry_ids,
                pose=pose_batch['window_uid'] if pose_batch is not None else [],coco=coco_ids,boundary=boundary_info))+ '\n')
        if diagnose:
            record('diagnostics.jsonl', dict(step=step, task_losses=values,
                    weighted_gradient_groups=gradient_report(task_vectors),geometry_policy=policy, grad_norm=float(norm),
                    optimized_total=sum(x['weighted_loss'] for x in values.values()), coverage=coverage()))
        boundary = step == args.max_steps or step % args.eval_every == 0 or (args.stop_after and step >= args.stop_after)
        if boundary:
            evaluate_step()
        if args.stop_after and step >= args.stop_after:
            atomic_json(out / 'paused.json', dict(step=step, reason='stop_after', resumable=True))
            return
    atomic_json(out / 'progress.json', dict(step=step, target_steps=args.max_steps,
                best_step=best_step, baseline_retained=best_step == 0, complete=True, coverage=coverage()))
    atomic_json(out / 'completion.json', dict(completed_steps=step, complete=True,
                best_step=best_step, baseline_retained=best_step == 0, coverage=coverage(),
                development_only=metadata['development_only'], elapsed_seconds=time.monotonic() - started))
    (out / 'paused.json').unlink(missing_ok=True)
