"""Build a validation reference from the original official annotation archive."""
import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile
import torch
from divibody_training.evaluate_standard_3dpw import Standard3DPWDataset, audit_split
from divibody_training.v28d_experiment_utils import atomic_json, sha256, validate_coverage, official_annotation_members


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-archive',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--image-archive',type=Path)
    parser.add_argument('--train-cache',type=Path,required=True)
    parser.add_argument('--val-cache',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    root=args.output_dir/'reference_dataset'
    (root/'imageFiles').mkdir(parents=True,exist_ok=True)
    with ZipFile(args.annotation_archive) as archive:
        records={split:official_annotation_members(archive.namelist(),split)
                 for split in ('train','validation','test')}
        for split in ('train','validation'):
            target=root/'sequenceFiles'/split
            target.mkdir(parents=True,exist_ok=True)
            for member in records[split]:
                path=target/Path(member).name
                content=archive.read(member)
                if path.exists() and path.read_bytes()!=content:
                    raise ValueError('Existing staged annotation differs from archive')
                if not path.exists(): path.write_bytes(content)
    sequence_names={split:sorted(Path(n).stem for n in members) for split,members in records.items()}
    if any(set(sequence_names[a]) & set(sequence_names[b]) for a,b in
           [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Official archive contains overlapping splits')
    wanted=set(sequence_names['train']+sequence_names['validation'])
    missing=[]
    for name in wanted:
        destination=root/'imageFiles'/name
        source=args.image_root/name
        if not destination.exists():
            if source.is_dir(): destination.symlink_to(source.resolve(),target_is_directory=True)
            else: missing.append(name)
    if missing and args.image_archive:
        with ZipFile(args.image_archive) as archive:
            for member in archive.namelist():
                parts=Path(member).parts
                if len(parts)>=3 and parts[-2] in missing and parts[-1].endswith(('.jpg','.png')):
                    destination=root/'imageFiles'/parts[-2]/parts[-1]
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not destination.exists(): destination.write_bytes(archive.read(member))
    audits={split:audit_split(root,split) for split in ('train','validation')}
    val_audit=audits['validation']
    dataset=Standard3DPWDataset(root=root,split='validation',image_size=96,
                              frame_stride=1,max_samples=None,crop_to_person=True)
    uids=sorted(f'{s.sequence}/person_{s.person_index}/frame_{s.frame_index:05d}' for s in dataset.samples)
    independent=(len(uids)==val_audit['expected_valid_person_frames']
                 and val_audit['missing_unique_rgb_frames']==0)
    reference=dict(format='v28d-validation-reference-v1',
        source='https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip',
        archive_path=str(args.annotation_archive.resolve()), archive_sha256=sha256(args.annotation_archive),
        independent_archive_verified=independent,validation_frame_uids=uids,
        **sequence_names)
    reference_path=args.output_dir/'validation_reference.json'
    atomic_json(reference_path,reference)
    train=torch.load(args.train_cache,map_location='cpu',mmap=True,weights_only=False)
    val=torch.load(args.val_cache,map_location='cpu',mmap=True,weights_only=False)
    coverage=validate_coverage(train,val,reference_path,allow_partial=True)
    result=dict(status='complete' if coverage['complete'] else 'incomplete_cache',
                official_sequence_counts={k:len(v) for k,v in sequence_names.items()},
                reference_dataset=str(root.resolve()),audits=audits,coverage=coverage,
                reference=str(reference_path.resolve()))
    atomic_json(args.output_dir/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
