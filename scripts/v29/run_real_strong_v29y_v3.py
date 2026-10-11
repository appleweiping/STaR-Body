"""Run a supplied trained V27/V28 checkpoint on real VACE/HMR cache inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--vace-cache',type=Path,required=True)
    parser.add_argument('--hmr-cache',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-windows',type=int,default=2)
    parser.add_argument('--strong-manifest',type=Path,required=True)
    args=parser.parse_args()
    if 'smoke_untrained' in args.checkpoint.name:
        raise ValueError('The random smoke checkpoint is not a trained model')
    if any(not path.is_file() for path in (args.checkpoint,args.vace_cache,args.hmr_cache)):
        raise FileNotFoundError('Trained checkpoint and real cache pair must exist')
    torch.set_num_threads(4); started=time.monotonic()
    from divibody_training.evaluate_standard_3dpw import enable_legacy_chumpy_compatibility
    enable_legacy_chumpy_compatibility()
    from divibody_training.cache_hmr2a_rgb_anchor_v8 import validate_vace_source,validate_sidecar
    from divibody_training.ram_parts_runtime_v29 import V10Dataset
    from models.human_decoder_strong_rgb_v29y import model_from_payload, FORMAT
    from divibody_training.v29_strong_features_v1 import StrongFeatureStore,StrongRGBFeatureDataset
    import smplx
    source=Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/Backup/4D-Humans')
    assets=Path('/mnt/32a56f7d-8270-4d4e-b9cb-fb8857bd7e93/testpro/xmj/sam-3d-body/divibody_vace/checkpoints/4DHumans/extracted/data')
    spec=importlib.util.spec_from_file_location('v28d_inference_smpl_wrapper',source/'hmr2/models/smpl_wrapper.py')
    wrapper=importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    neutral=smplx.SMPL(model_path=str(assets/'smpl'),gender='neutral',num_betas=10)
    official=wrapper.SMPL(model_path=str(assets/'smpl'),gender='neutral',num_betas=10,
        joint_regressor_extra=str(assets/'SMPL_to_J19.pkl'))
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if not any(key in payload for key in ('epoch','step','validation','source_sha256')):
        raise ValueError('Checkpoint lacks training provenance; inspect it before using it as a trained model')
    model=model_from_payload(payload).bind_geometry(neutral).bind_landmarks(official).eval()
    vace=torch.load(args.vace_cache,map_location='cpu',weights_only=False,mmap=True)
    hmr=torch.load(args.hmr_cache,map_location='cpu',weights_only=False,mmap=True)
    validate_vace_source(vace); validate_sidecar(hmr,vace)
    store=StrongFeatureStore(args.strong_manifest,vace,args.vace_cache,expected_split='train',expected_dim=model.strong_config.input_dim)
    dataset=StrongRGBFeatureDataset(V10Dataset(vace,hmr),store)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    with torch.inference_mode():
        for index,batch in enumerate(DataLoader(dataset,batch_size=1,shuffle=False)):
            if args.max_windows > 0 and index >= args.max_windows:
                break
            values=[batch[key].float() for key in ('rgb_tokens','vace_features','base_rotmat','base_betas','base_camera')]
            source_ids=torch.tensor(vace['metadata'][index]['target_source_frame_ids'],dtype=torch.long)[None]
            output=model(*values,frame_valid=batch['target_valid'],source_frame_ids=source_ids,strong_rgb_pooled=batch['strong_rgb_pooled'])
            for key in ('rotmat','betas','camera','direct_2d'):
                assert torch.isfinite(output[key]).all(),key
            rotation=output['rotmat']
            assert torch.allclose(rotation@rotation.transpose(-1,-2),torch.eye(3).expand_as(rotation),atol=1e-4)
            assert (output['camera'][...,0]>0).all()
            prediction={key:output[key].cpu() for key in ('rotmat','betas','camera','direct_2d')}
            prediction.update(frame_uids=vace['metadata'][index]['target_frame_uids'],
                frame_valid=batch['target_valid'],frame_owner=batch['frame_owner'])
            torch.save(prediction,args.output_dir/f'prediction_{index:04d}.pt')
            rows.append(dict(window=index,frame_uids=prediction['frame_uids'],
                owned_frames=int(batch['frame_owner'].sum())))
            print('Complete V29 real window',index+1,'/',len(dataset),flush=True)
    result=dict(status='passed',complete_v29_run=True,random_features=False,
        trained_checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_format=payload['format'],native_v29_checkpoint=payload['format']==FORMAT,
        v29_migrated_from_legacy=payload['format']!=FORMAT,
        checkpoint_epoch=payload.get('epoch'),checkpoint_step=payload.get('step'),
        vace_cache=str(args.vace_cache),hmr_cache=str(args.hmr_cache),
        development_only=bool(vace['signature'].get('development_only') or hmr['signature'].get('development_only')
                              or len(rows)<len(dataset)),
        full_dataset_benchmark=False,cpu_compute=True,windows=rows,seconds=time.monotonic()-started)
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
