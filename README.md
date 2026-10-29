# STaR-Body

**Joint-indexed feedback for spatiotemporal human mesh refinement**

The current research snapshot is **C10**, with native checkpoint format
`star-body-joint-feedback-v37-r1`. It adds landmark-to-pose feedback to the
trained spatiotemporal body refiner. The repository preserves the original code
and earlier version files; the C10 release adds its required model, runtime,
training, evaluation, asset-preparation, acceptance-check, and configuration
files without rewriting those originals.

[C10 release guide](C10_RELEASE.md) · [Validation evidence](evidence/c10_validation.json) ·
[Recorded controls](evidence/c10_transcribed_controls.json) ·
[Source manifest](C10_SOURCE_MANIFEST.json)

## Manuscript

[Revised paper and editable Word](paper/) · [Supplementary material](paper/STaR-Body_Supplement.pdf)

## Method

STaR-Body combines cached HMR 2.0a image features and Wan/VACE video context with
geometric refinement of a camera-space SMPL body. The inherited decoder mixes
spatial-first and temporal-first attention, while a fixed track prototype
conditions a shared shape residual. The final refiner performs three bounded
rotation updates using local image samples and a grouped GRU.

C10 preserves the identities of 17 landmark observations and routes them to
24 pose joints through learned queries and an anatomical-distance attention
bias. The feedback enters before part pooling and masked temporal exchange.
The observation head is inherited; it is not an external 2D detector. The
reported configuration enables joint feedback and part/time refinement.
Optional skeleton, inverse-kinematics, and shape-consensus branches remain
available in the research source but are not enabled in C10.

## Recorded validation result

The selected C10 checkpoint is the EMA after 1,000 additional optimization
steps from the common refinement parent. Evaluation uses the full **3DPW validation**
split, FP32 network and SMPL computation, 10,413 person-frames, and 10,226
annotation-adjacent acceleration triplets. Person crops are annotation-derived
under the shared cached-input protocol.

| Model | MPJPE ↓ mm | PA-MPJPE ↓ mm | PVE ↓ mm | Acceleration ↓ mm/annotation-frame² |
|---|---:|---:|---:|---:|
| Common refinement parent | 61.2628 | 37.6422 | 73.0249 | 21.4816 |
| Mean feedback, +1,000-step EMA | 61.2619 | 37.6371 | 73.0492 | 21.4469 |
| C10 joint feedback, +1,000-step EMA | 61.1884 | 37.6134 | 72.8197 | 21.4425 |

Against the recorded equal-budget mean-feedback control, relative error
reductions are 0.12%, 0.06%, 0.31%, and 0.02%, respectively. These are observed
validation comparisons, not a held-out test benchmark or a multi-seed estimate.
The C10 and parent values come from the bundled machine-readable evaluations;
mean-feedback values are explicitly labeled transcriptions of contemporaneous
run records. The evidence files preserve full precision, source hashes, and
checkpoint-selection details.

## Source entry points

| Purpose | File |
|---|---|
| Joint-indexed feedback | [joint_feedback_v37.py](video_VACE/models/joint_feedback_v37.py) |
| Part/time GRU integration | [part_time_refiner_v36.py](video_VACE/models/part_time_refiner_v36.py) |
| Native C10 model | [human_decoder_kinematic_v36.py](video_VACE/models/human_decoder_kinematic_v36.py) |
| Training phase | [train_target60_v36.py](video_VACE/divibody_training/train_target60_v36.py) |
| Portable configuration wrapper | [train_from_config_v36.py](scripts/train_from_config_v36.py) |
| Full-coverage evaluation | [evaluate_target60_v36.py](scripts/evaluate_target60_v36.py) |
| Configuration example | [c10.example.json](configs/c10.example.json) |

The source release includes the portable preparation and acceptance utilities,
versioned predecessor implementations, and safe example configurations required
to inspect the complete C10 workflow. Private operational archives and machine
settings are excluded; this is a usable source release, not a mirror of every
archived server file. See the release guide for the tool inventory.

The historical CLI selector remains `v36`; checkpoint loading requires the
native C10 format stated above. Do not relabel a checkpoint from an earlier
format as C10. The release guide distinguishes the active track-shape residual
from the optional, inactive shape-consensus branch.

## Environment and external assets

Use a compatible Linux/Python/PyTorch CUDA environment. The original dependency
specifications are retained. HMR2/4D-Humans and Wan2.1 sources, licensed SMPL
assets and regressors, the audited common refinement parent checkpoint, paired feature caches,
coverage manifests, and matching track prototypes must be obtained separately.
The repository contains **no datasets, model weights, feature caches,
credentials, or machine-local configuration** in the new C10 release.

```bash
python -m pip install -r requirements-star-body-v31.txt
export HMR2_SOURCE_ROOT=/path/to/4D-Humans
export WAN_SOURCE_ROOT=/path/to/Wan2.1
# Fill the asset paths in a local copy of the example configuration.
cp configs/c10.example.json configs/c10.local.json
bash scripts/run_star_body.sh scripts/train_from_config_v36.py \
  --config configs/c10.local.json --dry-run
```

The dry run checks the configuration and asset presence without starting
training. A fresh source-only checkout will report missing external assets.
Actual training uses the same command without `--dry-run` and must be started
explicitly. Evaluation requires the matching checkpoint, caches, SMPL assets,
and independent frame-coverage manifest. Test evaluation additionally requires
an immutable validation selection lock and a separate ground-truth cache.

Publication preparation checked source hashes, Python syntax, configuration
transcription, and numerical evidence. It did not rerun training or model
inference, and the source bundle does not claim to be a complete GPU environment.
Older progress notes and server-specific controllers are historical records,
not the current C10 release procedure.

## Preservation and licensing

Original model and training files remain unchanged. The
[original source hashes](original_code_hashes.json) and the C10 manifest identify
the preserved code. Use separate versioned files for subsequent model changes.

The existing [license](LICENSE.txt), [third-party notices](THIRD_PARTY_NOTICES.md),
and [VACE license](video_VACE/LICENSE.txt) remain in force. External datasets,
body-model assets, source packages, and pretrained weights retain their own
license conditions. This project is a research implementation and is not an
official reproduction of the third-party methods it builds upon.
