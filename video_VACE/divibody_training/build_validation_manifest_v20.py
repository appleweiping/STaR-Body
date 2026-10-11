"""Build validation frame identities from source annotations, never from feature caches.

--split-list is a JSON with source, train, validation, test names independently
checked against the official archive. No test annotations or metrics are read.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split-list', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.split_list.read_bytes()
    splits = json.loads(raw.decode('utf-8-sig'))
    for name, count in (('train',24), ('validation',12), ('test',24)):
        if len(splits[name]) != count or len(set(splits[name])) != count:
            raise ValueError(f'Invalid reference {name} sequence list')
    if not splits.get('source'):
        raise ValueError('Reference split list must identify its official source')
    from divibody_training.evaluate_standard_3dpw import Standard3DPWDataset, audit_split
    from divibody_training.cache_continuous_3dpw_vace_features import frame_identities_from_samples
    audit = audit_split(args.data_root, 'validation')
    dataset = Standard3DPWDataset(root=args.data_root, split='validation', image_size=96,
                                 frame_stride=1, max_samples=None, crop_to_person=True)
    frames = frame_identities_from_samples(dataset.samples)
    if {frame.sequence for frame in frames} != set(splits['validation']):
        raise ValueError('Source validation data does not cover the 12 reference sequences')
    if (len(frames) != audit['available_valid_person_frames']
            or audit['available_valid_person_frames'] != audit['expected_valid_person_frames']):
        raise ValueError('Missing validation images or source/dataset coverage mismatch')
    uids = sorted(frame.uid for frame in frames)
    if len(uids) != len(set(uids)):
        raise ValueError('Duplicate source validation person-frames')
    manifest = dict(format='3dpw-validation-manifest-v20', source=splits['source'],
        train=splits['train'], validation=splits['validation'], test=splits['test'],
        validation_frame_uids=uids, data_root=str(args.data_root.resolve()),
        split_list_sha256=hashlib.sha256(raw).hexdigest(), source_audit=audit)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps({'output': str(args.output), 'validation_person_frames': len(uids)}))


if __name__ == '__main__':
    main()
