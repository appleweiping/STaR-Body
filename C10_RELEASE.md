# C10 joint-indexed feedback source release

This source snapshot contains the implemented C10 model, its training and
evaluation entry points, a portable configuration example, and archived
validation measurements. Existing model and training files are copied unchanged
from the hash-bound source archive. Original project versions remain available.

## Model and entry points

C10 extends the common refinement parent with joint-indexed feedback. The
feedback module preserves 17 observation identities and uses 24 SMPL pose queries,
landmark/schema embeddings, and anatomy-distance attention bias. Its output feeds
the existing iterative GRU and masked part/time refinement. The observation head
is inherited; this branch does not use an external 2D detector or ground-truth
2D coordinates as forward inputs.

- `video_VACE/models/joint_feedback_v37.py`: landmark-to-pose feedback.
- `video_VACE/models/part_time_refiner_v36.py`: feedback integration and temporal masks.
- `video_VACE/models/human_decoder_kinematic_v36.py`: native C10 model construction.
- `video_VACE/divibody_training/train_target60_v36.py`: training phase.
- `scripts/train_from_config_v36.py`: portable configuration wrapper.
- `scripts/evaluate_target60_v36.py`: full-coverage evaluation.
- `scripts/test_joint_feedback_v37.py`: implementation checks; not performance experiments.

The historical CLI version is `v36`, while the required native checkpoint format
is `star-body-joint-feedback-v37-r1`. The configuration selects arm R2 and joint
feedback. Skeleton, inverse-kinematics, and shape-consensus branches in the source
are not enabled in the reported C10 run.

## Recorded result

The archived selection is step 1,000 EMA, evaluated on the full 3DPW validation
split with FP32 network, SMPL, and metrics: 10,413 person-frames and 10,226
acceleration triplets. Crops follow the annotation-derived protocol in the
audited caches. These values are validation results, not a held-out test result.

| Model | MPJPE mm | PA-MPJPE mm | PVE mm | Acceleration mm/frame² |
|---|---:|---:|---:|---:|
| Common refinement parent | 61.2628 | 37.6422 | 73.0249 | 21.4816 |
| C10 step 1,000 EMA | 61.1884 | 37.6134 | 72.8197 | 21.4425 |

`evidence/c10_validation.json` retains all three archived C10 evaluations, the
common-parent reproduction, loss-pilot comparisons, selection flags, and source
hash references. It does not infer completed runs from historical plans. The
archive contains no C10 held-out test or independent multi-seed evaluation.

## Reconstructing the runtime

Use a compatible Linux/Python/PyTorch CUDA environment and install the retained
dependency specifications. External HMR2/4D-Humans and Wan2.1 sources, licensed
SMPL assets, paired feature caches, UID coverage manifests, track prototypes,
and the native common refinement parent checkpoint must be supplied separately. Their expected
roles are visible in `configs/c10.example.json`; replace its relative `assets/`
paths with your audited assets. Do not use a different catalog, crop, temporal
mapping, or checkpoint while describing the result as the same protocol.

```sh
python -m pip install -r requirements-star-body-v31.txt
export HMR2_SOURCE_ROOT=/path/to/4D-Humans
export WAN_SOURCE_ROOT=/path/to/Wan2.1
bash scripts/run_star_body.sh scripts/train_from_config_v36.py --config configs/c10.example.json --dry-run
```

The dry run validates configuration and asset presence and does not train. It
will report missing assets in this source-only distribution. Actual training
uses the same command without `--dry-run` and must be initiated explicitly.
Evaluation uses the native-format checkpoint with `--model-version v36`, explicit
split/caches/SMPL/coverage arguments, and `--amp-dtype none`; run the evaluator's
`--help` for the complete interface. Test evaluation additionally requires an
immutable validation selection lock and a separate ground-truth cache.

This publication preparation performed source parsing, hash comparison, and
evidence transcription checks only. It did not run training or model inference.
Model weights, datasets, cached features, local-machine settings, and operational
logs are not included. Licenses and third-party notices are retained.

## Preparation and acceptance utilities

The release also includes cached-prediction and reference-report preparation
(`scripts/cache_target60_predictions_v34.py`,
`scripts/prepare_target60_reference_v34.py`), motion-cache preparation and
pretraining, V34 acceptance/resume/unit checks, V36 component/production checks,
and the portable versioned wrappers. Earlier versioned implementation files and
safe example configurations support dependency inspection and controlled
comparisons; only `configs/c10.example.json` describes the selected C10 setup.
These utilities are supplied as source and have not been run during publication
preparation. Server-local configurations, SSH launchers, historical provenance,
and operational monitoring notes are not part of this portable release.

`evidence/c10_transcribed_controls.json` identifies work-record transcriptions
separately from original run JSON. Full-precision figure inputs are supplied in
`evidence/c10_per_sequence_differences.json` and
`evidence/c10_continuation_records.json`; no unrecorded continuation points are
filled in.
