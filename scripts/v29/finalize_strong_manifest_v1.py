"""Freeze a fully extracted real RGB split into the strict training contract."""
import argparse
import json
from pathlib import Path
import torch
from divibody_training.v29_strong_features_v1 import (
    FORMAT, FEATURE_POLICY, RGB_RESOLVER_POLICY, expected_frame_records,
    sha256_file, uid_digest, _json_digest, StrongFeatureStore)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();directory=a.features.resolve()
    old=json.loads((directory/'manifest.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    if (old['format']!='v29-strong-rgb-feature-cache-v5' or result['manifest']!=old
            or result.get('status')!='passed' or result.get('complete_split_cache') is not True
            or old.get('complete') is not True or old['split'] not in ('train','validation')
            or result.get('real_rgb') is not True or result.get('random_features') is not False):
        raise ValueError('Complete real split-specific extraction is required')
    cache=Path(old['source_cache'])
    if sha256_file(cache)!=old['source_vace_cache_sha256']:raise ValueError('Source cache changed')
    vace=torch.load(cache,map_location='cpu',mmap=True,weights_only=False)
    mapping=expected_frame_records(vace,old['split']);uids=list(mapping)
    if old['selected_uids']!=uids or [row['uid'] for row in result['samples']]!=uids:
        raise ValueError('Extracted UID order/coverage differs from complete split')
    sam=root.parent/'checkpoints/sam-3d-body-dinov3'
    paths=dict(checkpoint=sam/'model.ckpt',model_config=directory/'encoder_config.json',
        backbone_source=Path(old['model_backbone_source']),
        dinov3_hubconf=root/'dependencies/dinov3_source_20261006/hubconf.py',
        extractor_source=root/'scripts/v29/prepare_sam3d_teacher_v6.py',
        mhr_asset=sam/'assets/mhr_model.pt')
    expected=dict(checkpoint=old['checkpoint_sha256'],model_config=old['encoder_config_sha256'],
        backbone_source=old['model_backbone_source_sha256'],dinov3_hubconf=old['dinov3_hubconf_sha256'],
        extractor_source=old['source_script_sha256'],mhr_asset=old['mhr_sha256'])
    assets={}
    for name,path in paths.items():
        actual=sha256_file(path)
        if actual!=expected[name]:raise ValueError('Encoder asset changed: '+name)
        assets[name]=dict(path=str(path),sha256=actual)
    samples=[]
    for row in result['samples']:
        if row['npz_sha256']!=sha256_file(directory/row['file']):raise ValueError('Feature changed')
        samples.append(row|dict(file_sha256=row['npz_sha256']))
    manifest=dict(format=FORMAT,split=old['split'],feature_dim=old['feature_dim'],
        feature_key='strong_rgb_pooled',feature_dtype='float32',feature_policy=FEATURE_POLICY,
        rgb_resolver_policy=RGB_RESOLVER_POLICY,complete=True,real_rgb=True,random_features=False,
        placeholder_features=False,three_d_targets_not_encoder_inputs=True,mhr_3d_used_as_supervision=False,
        test_frames_used=False,encoder_frozen=True,encoder_eval=True,
        input_bbox_policy='existing-vace-context-bbox-xyxy-at-target-position',bbox_annotation_derived=True,
        source_vace_cache_sha256=old['source_vace_cache_sha256'],frame_uids_sha256=uid_digest(uids),
        source_rgb_mapping_sha256=_json_digest(list(mapping.values())),
        encoder=dict(family='sam-3d-body-dinov3',inference_type='body',pooling='global-spatial-mean-float32',assets=assets),
        completed_frames=len(uids),samples=samples,extraction_result_sha256=sha256_file(directory/'result.json'))
    path=directory/'training_manifest_v1.json'
    with path.open('x') as f:json.dump(manifest,f,indent=2)
    store=StrongFeatureStore(path,vace,cache,expected_split=old['split'],expected_dim=old['feature_dim'])
    with (directory/'strict_verification_v1.json').open('x') as f:json.dump(store.audit,f,indent=2)
    print(json.dumps(store.audit))


if __name__=='__main__':main()
