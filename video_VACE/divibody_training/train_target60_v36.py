"""Native V30 Target60 fine-tuning. New optimizer phase; exact source state preserved."""
from dataclasses import asdict
import hashlib,json,platform,random,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from divibody_training.target60_contract_v36 import parse_training_args,inspect_assets,source_fingerprints
from divibody_training.train_divibody_rgb_vace_hmr2a_v8 import load_pair
from divibody_training.train_divibody_rgb_vace_hmr2a_v9_small import spread_subset
from divibody_training.target60_runtime_v34 import V10Dataset
from models.human_decoder_kinematic_v36 import model_from_payload,V36Config,FORMAT
from divibody_training.target60_track_dataset_v34 import TrackPrototypeDataset
from divibody_training.target60_validation_v34 import validate
from divibody_training.target60_optim_v36 import make_optimizer,ModelEMA
from divibody_training.target60_engine_v36 import execute_run
from divibody_training.v29_experiment_utils import validate_coverage
from divibody_training.evaluate_standard_3dpw import create_smpl_models
from models.human_decoder_target60_v34 import FORMAT as PARENT_FORMAT,AdaptationConfig
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    args=parse_training_args();torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(min(1.,args.max_cuda_memory_gib*1024**3/torch.cuda.get_device_properties(device).total_memory),device)
    audit=inspect_assets(args,True)
    if not audit['passed']:raise ValueError('V34 asset preflight failed: '+json.dumps(audit))
    payload=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('format')!=PARENT_FORMAT or not payload.get('step',0)>0 or payload.get('development_only',True):
        raise ValueError('Target60 phase requires an actual trained, full-data native V30 initialization')
    parent_sha=sha256(args.init_checkpoint)
    source_sha=payload['source_sha256']
    if source_sha!=payload.get('asset_manifest',{}).get('init_checkpoint',{}).get('sha256',source_sha):raise ValueError('Parent V30 provenance mismatch')
    parent_report=json.loads(args.parent_validation_report.read_text())
    if parent_report['checkpoint_sha256']!=parent_sha:raise ValueError('Parent report SHA mismatch')
    if parent_report.get('split')!='validation':raise ValueError('Only validation parent selection allowed')
    if (parent_report['metrics'].get('frames'),parent_report['metrics'].get('acceleration_triplets'))!=(10413,10226):raise ValueError('Parent must have complete validation')
    if parent_report['metrics']!=payload['validation']:raise ValueError('Parent report does not match actual selected checkpoint')
    if args.initial_reference_report:
        ref=json.loads(args.initial_reference_report.read_text())
        if ref['assets']['checkpoint']['sha256']!=source_sha:raise ValueError('Reference report checkpoint SHA differs')
        if ref.get('split')!='validation' or ref['config']['amp_dtype']!='none':raise ValueError('Need FP32 validation reference')
    arm=args.v36_arm
    v36=V36Config(arm=arm,heads_unfreeze_step=args.v36_heads_unfreeze_step,late_unfreeze_step=args.v36_late_unfreeze_step,
        temporal=arm in ('R2','R3a','R3b','R4'),skeleton=arm in ('R3a','R3b','R4'),ik=arm in ('R3b','R4'),shape_consensus=arm=='R4',feedback_mode='joint' if args.v37_joint_feedback else 'mean')
    model=model_from_payload(payload,v36).to(device)
    cfg=model.adaptation_config
    # Base architecture always comes from the native checkpoint, never an arbitrary CLI override.
    saved=payload['backbone_spec']['adaptation_config']
    for key,arg in (('hidden_dim',args.model_width),('num_heads',args.model_heads),('depth',args.model_depth),('iterations',args.iterations)):
        if saved[key]!=arg:raise ValueError('CLI cannot override saved native V30 architecture: '+key)
    models=create_smpl_models(args.smpl_model_root,device)
    for body in models.values():body.eval().requires_grad_(False)
    from hmr2.models.smpl_wrapper import SMPL
    official=SMPL(model_path=str(args.smpl_model_root),gender='neutral',num_betas=10,
        joint_regressor_extra=str(args.smpl_model_root.parent/'SMPL_to_J19.pkl')).to(device).eval().requires_grad_(False)
    model.bind_geometry(models[2]).bind_landmarks(official)
    regressor=torch.as_tensor(np.load(args.joint_regressor),dtype=torch.float32,device=device)
    if regressor.shape!=(17,6890) or not torch.isfinite(regressor).all():raise ValueError('Invalid H36M regressor')
    train_vace,train_hmr=load_pair(args.train_vace_cache,args.train_hmr_sidecar,'train')
    val_vace,val_hmr=load_pair(args.val_vace_cache,args.val_hmr_sidecar,'validation')
    coverage=validate_coverage(train_vace,val_vace,args.validation_manifest,args.allow_partial_validation)
    train=TrackPrototypeDataset(V10Dataset(train_vace,train_hmr),
        args.train_track_prototypes if cfg.use_track_shape else None,source_sha)
    val=TrackPrototypeDataset(V10Dataset(val_vace,val_hmr),
        args.val_track_prototypes if cfg.use_track_shape else None,source_sha)
    context_audit=dict(train=train.context_audit,validation=val.context_audit)
    development=not coverage['complete'] or bool(args.max_val_windows)
    if not development and args.initial_reference_report is None:raise ValueError('Formal Target60 requires exact V30 FP32 reference report')
    val_subset=spread_subset(val,args.max_val_windows)
    val_loader=DataLoader(val_subset,batch_size=args.val_batch_size,shuffle=False,num_workers=args.num_workers)
    optimizer,names=make_optimizer(model,args);ema=ModelEMA(model,args.ema_decay)
    metadata=dict(format=FORMAT,model_config=asdict(model.config),adaptation_config=asdict(cfg),
        backbone_spec=model.backbone_spec,reference_spec=model.reference_spec,
        v36_config=asdict(v36),parent_spec=model.parent_spec,
        parent_checkpoint_sha256=parent_sha,parent_step=payload['step'],parent_expected_validation=parent_report['metrics'],
        runner_family='target60-v37-joint-feedback-a5-new-optimizer-phase',selection_split='3DPW validation only',
        initialization=str(args.init_checkpoint),prototype_source_v30_sha256=source_sha,source_format=payload['format'],source_step=payload['step'],
        source_sha256=parent_sha,source_files_sha256=source_fingerprints(),
        asset_manifest=audit['assets'],cache_hash_policy=audit['cache_hash_policy'],
        coverage=coverage,development_only=development,test_used_for_model_selection=False,
        training_windows_3dpw=len(train),validation_windows_3dpw=len(val_subset),
        trainable_count=sum(x.numel() for x in model.parameters() if x.requires_grad),
        trainable_parameters={n:p.numel() for n,p in model.named_parameters() if p.requires_grad},
        context_audit=context_audit,gradient_accumulation=args.gradient_accumulation,
        validation_precision='FP32 network/SMPL/metrics',optimizer_reset_on_warm_start=True,
        native_v30_migration='strict A5 parent state; original GRU weights retained; new optimizer phase',
        target=dict(mpjpe_mm=args.target_mpjpe_mm,pa_mpjpe_mm=args.target_pa_mpjpe_mm),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    metadata['environment']=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        cudnn_deterministic=torch.backends.cudnn.deterministic,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/('resume_config_'+str(__import__('time').time_ns())+'.json' if args.resume_checkpoint else 'run_config.json')
    with path.open('x',encoding='utf-8') as f:json.dump(metadata,f,indent=2)
    print(json.dumps(dict(event='initialized',trainable_count=metadata['trainable_count'],development_only=development)),flush=True)
    def evaluate(network):
        eval_args=SimpleNamespace(**vars(args));eval_args.amp_dtype='none'
        metrics=validate(val_loader,network,models,regressor,eval_args,device)
        if not development and (metrics['frames'],metrics['acceleration_triplets'])!=(10413,10226):raise ValueError('Incomplete validation')
        print(json.dumps(dict(event='geometry_validation_done',metrics=metrics)),flush=True)
        return metrics,{}
    execute_run(model,optimizer,ema,SimpleNamespace(dataset=train),None,None,models,official,regressor,evaluate,args,device,metadata)
if __name__=='__main__':main()
