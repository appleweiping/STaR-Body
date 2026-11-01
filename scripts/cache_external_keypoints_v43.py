#!/usr/bin/env python3
"""Cache external COCO-17 2D keypoints for every frame of an audited HMR sidecar.

C11 / v43 evidence source. The keypoints are *model inputs*, so only detector outputs
may be stored here; 3DPW ground-truth joints are never read.

Coordinate convention (identical to the HMR 2.0a landmark heads used by C10):
    u = x_crop / 256 - 0.5,  v = y_crop / 256 - 0.5
in the square 256x256 HMR2 crop produced by ``RawBBoxHMR3DPWDataset`` (poses2d box,
25% margin, single 192:256 aspect expansion). ``camera_project`` in
``models/body_projection_v28.py`` uses the same normalisation (focal 5000 / 256).

Sources
  vitpose     ViTPose (HuggingFace transformers implementation) run top-down on the exact
              HMR2 crop. The box passed to the processor is pre-shrunk by its standard 1.25
              padding, so ViTPose sees exactly the central 192x256 view of the HMR2 backbone.
  openpose3dpw  The OpenPose-18 detections shipped as ``poses2d`` in the 3DPW pickles,
              mapped to COCO-17. They are detector outputs (the same ones that define the
              crop box), not ground truth.

Output (torch.save, never overwritten):
    dict(format='star-body-external-keypoints-v43-r1', source, model, split,
         frame_uids=list[str], keypoints=float32[N,17,3] (u, v, confidence),
         schema='coco', coordinate=..., sidecar_sha256, ...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FORMAT = 'star-body-external-keypoints-v43-r1'
# OpenPose-18 (COCO-18 order used by 3DPW poses2d) -> COCO-17.
COCO17_FROM_OPENPOSE18 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
HMR_SIZE = 256.
# The HF ViTPose processor enlarges every box by padding_factor 1.25 (aspect 192:256 kept).
# (51.2, 25.6, 153.6, 204.8) x 1.25 = (32, 0, 192, 256): exactly the region the HMR2 ViT sees.
VITPOSE_BOX_XYWH = (51.2, 25.6, 153.6, 204.8)
SIDECAR_KEYS = ('anchor_revision', 'raw_bbox_margin', 'crop_protocol', 'vace_semantic_sha256', 'split')


class Crops:
    """Module-level (picklable) adapter returning the HMR2 256x256 RGB crops in sidecar order."""

    def __init__(self, dataset, order):
        self.dataset, self.order = dataset, order

    def __len__(self):
        return len(self.order)

    def __getitem__(self, row):
        return self.dataset[self.order[row]]['image']  # [3,256,256] RGB in [0,1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, required=True, help='3DPW root with imageFiles/ and sequenceFiles/')
    p.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    p.add_argument('--hmr-sidecar', type=Path, required=True, help='Audited HMR sidecar whose frame_uids define order')
    p.add_argument('--source', choices=('vitpose', 'openpose3dpw'), required=True)
    p.add_argument('--model', default='usyd-community/vitpose-plus-huge',
                   help='HF model id or local directory (vitpose only). vitpose-plus-* use the COCO expert (dataset_index 0).')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--crop-margin', type=float, default=.25, help='Must equal the audited HMR crop protocol (0.25)')
    p.add_argument('--max-frames', type=int, default=0, help='Development only: first N sidecar frames')
    p.add_argument('--fp16', action='store_true', help='Half precision ViTPose inference (CUDA only)')
    return p.parse_args(argv)


def openpose_to_crop(points, center_x, center_y, side):
    """3DPW poses2d [18,3] or [3,18] in image pixels -> COCO-17 crop-normalised (u, v, conf)."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    if points.shape == (3, 18):
        points = points.T
    if points.shape != (18, 3):
        raise ValueError(f'Unexpected poses2d shape {points.shape}')
    coco = points[list(COCO17_FROM_OPENPOSE18)]
    u = (coco[:, 0] - center_x) / side
    v = (coco[:, 1] - center_y) / side
    conf = coco[:, 2]
    good = np.isfinite(coco).all(1) & (conf > 0)
    out = np.stack((u, v, conf), 1)
    out[~good] = 0.
    return out.astype(np.float32)


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from divibody_training.audit_hmr2a_crop_protocol_v11 import RawBBoxHMR3DPWDataset
    from divibody_training.evaluate_unified_hmr2_divibody_3dpw import hmr_bbox_size, _load_annotation

    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; caches are never overwritten')
    if abs(args.crop_margin - .25) > 1e-12:
        raise ValueError('Only the audited raw-bbox 0.25 crop protocol is supported')
    sidecar = torch.load(args.hmr_sidecar, map_location='cpu', weights_only=False, mmap=True)
    signature = sidecar.get('signature', {})
    # The crops below are rebuilt with RawBBoxHMR3DPWDataset; refuse sidecars built otherwise.
    if signature.get('anchor_revision') != 'raw-bbox-v11' or abs(float(signature.get('raw_bbox_margin', -1)) - args.crop_margin) > 1e-9:
        raise ValueError('HMR sidecar was not built with the raw-bbox-v11 / 0.25 crop protocol: '
                         f'{signature.get("anchor_revision")!r}, margin {signature.get("raw_bbox_margin")!r}')
    if signature.get('split') != args.split:
        raise ValueError(f'HMR sidecar split {signature.get("split")!r} != {args.split!r}')
    frame_uids = list(sidecar['frame_uids'])
    if args.max_frames:
        frame_uids = frame_uids[:args.max_frames]
    if len(frame_uids) != len(set(frame_uids)):
        raise ValueError('Sidecar frame_uids are not unique')
    dataset = RawBBoxHMR3DPWDataset(root=args.data_root, split=args.split, image_size=256, frame_stride=1,
                                    max_samples=None, crop_to_person=True, crop_margin=args.crop_margin)
    uid_of = lambda s: f'{s.sequence}/person_{s.person_index}/frame_{s.frame_index:05d}'
    lookup = {uid_of(sample): index for index, sample in enumerate(dataset.samples)}
    missing = [uid for uid in frame_uids if uid not in lookup]
    if missing:
        raise ValueError(f'{len(missing)} sidecar frames are absent from the raw 3DPW split, e.g. {missing[:3]}')
    order = [lookup[uid] for uid in frame_uids]
    keypoints = np.zeros((len(frame_uids), 17, 3), dtype=np.float32)
    started = time.perf_counter()

    if args.source == 'openpose3dpw':
        for row, index in enumerate(order):
            sample = dataset.samples[index]
            annotation = _load_annotation(sample.annotation_path)
            from PIL import Image
            with Image.open(sample.image_path) as image:
                width, height = image.size
            bbox = dataset._person_bbox(annotation, sample.person_index, sample.frame_index, width, height)
            if bbox is None:
                bbox = (0., 0., float(width), float(height))
            cx, cy, side = hmr_bbox_size(bbox)
            points = np.asarray(annotation['poses2d'][sample.person_index][sample.frame_index])
            keypoints[row] = openpose_to_crop(points, cx, cy, side)
            if row % 5000 == 0:
                print(json.dumps(dict(event='progress', frames=row, total=len(order))), flush=True)
        model_name = '3DPW poses2d (OpenPose-18 detections shipped with the dataset)'
    else:
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        device = torch.device(args.device)
        processor = AutoProcessor.from_pretrained(args.model)
        model = VitPoseForPoseEstimation.from_pretrained(args.model).to(device).eval()
        if args.fp16:
            if device.type != 'cuda':
                raise ValueError('--fp16 requires CUDA')
            model = model.half()
        moe = getattr(model.config, 'backbone_config', None) is not None and \
            getattr(model.config.backbone_config, 'num_experts', 1) > 1

        loader = DataLoader(Crops(dataset, order), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        cursor = 0
        with torch.inference_mode():
            for images in loader:
                arrays = [(img.permute(1, 2, 0).numpy() * 255.).round().clip(0, 255).astype(np.uint8) for img in images]
                boxes = [[list(VITPOSE_BOX_XYWH)] for _ in arrays]
                inputs = processor(images=arrays, boxes=boxes, return_tensors='pt').to(device)
                if args.fp16:
                    inputs['pixel_values'] = inputs['pixel_values'].half()
                extra = dict(dataset_index=torch.zeros(len(arrays), dtype=torch.long, device=device)) if moe else {}
                outputs = model(**inputs, **extra)
                from types import SimpleNamespace  # float32 heatmaps: scipy filters reject float16
                results = processor.post_process_pose_estimation(SimpleNamespace(heatmaps=outputs.heatmaps.float()), boxes=boxes)
                for offset, per_image in enumerate(results):
                    person = per_image[0]
                    xy = person['keypoints'].float().cpu().numpy()
                    score = person['scores'].float().cpu().numpy()
                    if xy.shape != (17, 2):
                        raise ValueError(f'Expected COCO-17 ViTPose output, got {xy.shape}')
                    keypoints[cursor + offset, :, :2] = xy / HMR_SIZE - .5
                    keypoints[cursor + offset, :, 2] = score
                cursor += len(arrays)
                if (cursor // len(arrays)) % 50 == 0:
                    print(json.dumps(dict(event='progress', frames=cursor, total=len(order))), flush=True)
        if cursor != len(order):
            raise RuntimeError('Incomplete ViTPose pass')
        model_name = args.model

    if not np.isfinite(keypoints).all():
        raise FloatingPointError('Non-finite keypoints')
    payload = dict(format=FORMAT, source=args.source, model=model_name, split=args.split, schema='coco',
                   coordinate='HMR2 256-crop normalised: u=x/256-0.5, v=y/256-0.5; channel 2 = detector confidence',
                   frame_uids=frame_uids, keypoints=torch.from_numpy(keypoints),
                   sidecar=str(args.hmr_sidecar), sidecar_sha256=sha256(args.hmr_sidecar),
                   sidecar_signature=signature, sidecar_binding={k: signature.get(k) for k in SIDECAR_KEYS},
                   crop_margin=args.crop_margin, vitpose_box_xywh=VITPOSE_BOX_XYWH if args.source == 'vitpose' else None,
                   development_only=bool(args.max_frames), created_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.perf_counter() - started, ground_truth_used=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(args.output)
    conf = keypoints[..., 2]
    print(json.dumps(dict(event='done', output=str(args.output), frames=len(frame_uids),
                          mean_confidence=float(conf.mean()), frac_conf_gt_0_3=float((conf > .3).mean()),
                          elapsed_seconds=payload['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
