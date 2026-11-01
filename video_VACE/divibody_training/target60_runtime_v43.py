"""C11 (v43) runtime: v36 runtime plus external-evidence plumbing.

Only ``forward_batch`` changes: it forwards ``external_keypoints`` (detector outputs,
[B,T,17,3]) together with the v34 track context. Losses and datasets are re-exported
unchanged from target60_runtime_v36 so the training objective is identical to C10.
"""
import torch

from divibody_training.ram_parts_runtime_v27 import INPUT_KEYS
from divibody_training.target60_track_dataset_v34 import CONTEXT_KEYS
from divibody_training.target60_runtime_v36 import (  # noqa: F401  (re-exported, unchanged)
    V10Dataset, geometric_losses, pose_losses)

EVIDENCE_KEYS = ('external_keypoints',)
CONTEXT_KEYS_V43 = (*CONTEXT_KEYS, *EVIDENCE_KEYS)


def forward_batch(model, batch, device, amp_dtype, position_offset=0):
    values = [batch[k].to(device, dtype=torch.float32 if amp_dtype == 'none' else batch[k].dtype, non_blocking=True)
              for k in INPUT_KEYS]
    valid = batch.get('target_valid')
    valid = valid.to(device, dtype=torch.bool) if valid is not None else None
    schema = batch.get('landmark_schema', 'posetrack')
    if isinstance(schema, (list, tuple)):
        if len(set(schema)) != 1:
            raise ValueError('Each batch must have one landmark convention')
        schema = schema[0]
    source = batch.get('source_frame_ids')
    if source is not None:
        source = source.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and amp_dtype == 'bf16'):
        context = {k: batch[k].to(device, dtype=torch.float32 if batch[k].is_floating_point() else batch[k].dtype,
                                  non_blocking=True)
                   for k in CONTEXT_KEYS_V43 if k in batch}
        return model(*values, frame_valid=valid, landmark_schema=schema, source_frame_ids=source,
                     position_offset=position_offset, force_anchor=getattr(model, 'baseline_only', False), **context)
