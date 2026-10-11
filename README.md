# STaR-Body: Spatial-Temporal Attentive Refinement for 3D Human Body Reconstruction

Research code for camera-space SMPL body reconstruction from monocular video, developed on 3DPW. The only trained results are provisional, single-seed figures from an incomplete run on the 3DPW validation split; no STaR-Body test-split result exists.

**Contents:** [Overview](#overview) · [Project status](#project-status) · [Method](#method) · [Repository structure](#repository-structure) · [Requirements and installation](#requirements-and-installation) · [Data and external assets](#data-and-external-assets) · [Usage](#usage) · [Results](#results) · [Reproducibility and provenance](#reproducibility-and-provenance) · [Documentation index](#documentation-index) · [License and third-party notices](#license-and-third-party-notices) · [Citation](#citation) · [Acknowledgements](#acknowledgements)

## Overview

STaR-Body addresses video-based recovery of SMPL body pose, shape and camera in the camera coordinate frame. Accuracy is measured per frame by MPJPE, PA-MPJPE and PVE, and temporal quality by the acceleration error of continuous sequences; world-frame trajectory estimation is treated as a separate task and is out of scope ([docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)). The model is an iterative spatiotemporal refinement decoder placed on top of a frozen V28 estimator from the inherited DiviBody/VACE code base. For every frame, 24 SMPL joint queries and one shape and one camera query attend to HMR 2.0 spatial image tokens, to a compressed Wan2.1-VACE spatiotemporal feature memory and to learned 2D landmark features. They are then processed by blocks that mix a space-then-time and a time-then-space attention order. Separate heads predict root orientation, body rotations, shape and camera as residuals with respect to the frozen estimate, and each refinement iteration re-runs SMPL and re-samples image features around the projected joints. This repository provides the model and training source code, configuration templates, an evaluator, verification tools, provenance manifests and a redacted summary of the recorded validation evidence. Datasets, body models, pretrained weights, trained checkpoints and feature caches are not distributed.

## Project status

This is a research code release, not a finished benchmark release. The evidence available for each version differs:

| Version | Date | Role | Evidence (as reported; source in this repository) |
|---|---|---|---|
| V30 | 2026-10-08 (snapshot) | The only version with trained results | Single seed (2026), in-training checkpoints at 1,000-3,000 of a registered 6,000 steps, 3DPW **validation** only ([evidence/v30_validation_summary.json](evidence/v30_validation_summary.json)). No test result, no multi-seed result, and no evidence that either registered 6,000-step arm completed. The trained weights and full logs were not recovered. |
| V31 | Implemented 2026-10-08; delivery manifest 2026-10-09T01:11:42Z | Current development code. Same architecture, shape/radius behaviour and loss defaults as V30, except for a re-implemented `rotation_6d` (always on; different numerics and degenerate-input fallback) and a V31-only checkpoint format, so default V31 is not numerically identical to V30 | Functional verification only: synthetic numerical checks, a real 8-frame forward/backward/restore check, and an untrained consistency check on a 16-frame development subset ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)). No trained metrics. The matched V30/V31 experiment is registered but not started ([configs/v31_experiment_plan.json](configs/v31_experiment_plan.json)). |
| V30 rebuild | Undated (after V31) | Infrastructure to retrain the unchanged V30 model under V31-style provenance binding; not yet run beyond dry-run checks | CPU, AST, CLI-binding and dry-run checks only; no training ([provenance/v30_rebuild_copy.json](provenance/v30_rebuild_copy.json)). |

All STaR-Body scores recorded here are provisional. This repository makes no claim that STaR-Body outperforms published methods. Such a claim would require a frozen model, multiple seeds, an independent same-protocol evaluation of the competing methods, and a test evaluation that is disclosed as such. The test split has already been observed by earlier models of this project ([configs/v31_experiment_plan.json](configs/v31_experiment_plan.json)).

## Method

### Task and evaluation scope

The target task is 3DPW camera-space SMPL reconstruction. The primary metrics are MPJPE, PA-MPJPE and PVE in millimetres, and the acceleration error of continuous sequences in mm/frame² ([docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)). The evaluation protocol, `matched_project_3dpw_owned_sequence_v1` ([scripts/evaluate_body_v31.py](scripts/evaluate_body_v31.py)), is defined as follows:

- **Joints and metrics.** Joints are H36M-regressed and mapped to the SPIN/LSP J14 set. MPJPE is pelvis-aligned; PA-MPJPE applies a per-frame Procrustes similarity alignment on J14; PVE is the unaligned SMPL vertex error.
- **Body models.** Predictions use the neutral SMPL model and ground truth uses the gendered SMPL model.
- **Frame counting.** Each person-frame is counted once, by its owning window.
- **Acceleration.** The acceleration error is the second difference of the hip-centred J14 prediction-minus-target error across adjacent annotation UIDs, including UIDs that span window boundaries. Because annotation and source frame identities differ, the unit mm/frame² is not directly comparable with m/s² values reported in the literature ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)).
- **Crops and post-processing.** Person crops are oracle crops derived from the 3DPW `poses2d` annotations, not detector boxes, which must be disclosed in any comparison ([cache_continuous_3dpw_vace_features_v31_resume.py](video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py), [cache_hmr2a_rgb_anchor_v31_resume.py](video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v31_resume.py)). No flip test and no smoothing or other post-processing are applied.

### Architecture

The description below follows the V30 implementation ([video_VACE/models/human_decoder_spatiotemporal_v30.py](video_VACE/models/human_decoder_spatiotemporal_v30.py)). V31 differs only as listed under [Changes in V31](#changes-in-v31).

**Inputs.** Each training sample is a window of T = 8 consecutive target frames ([video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py](video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py)). Each window provides:
- HMR 2.0a spatial image tokens: a 16×12 grid of 192 tokens of width 1280 per frame, cached in BF16.
- Wan2.1-VACE features: 5 latent slices, each a 6×6 grid of width 1536. They are encoded from a 17-frame RGB context at 96 px, in which the 8 target frames occupy context positions 4-11.
- The HMR 2.0a pose, shape and crop-camera estimates.
- A frame-validity mask.
- Mandatory integer source-frame identifiers.

**Frozen reference.** An audited V28 (or V29) checkpoint is loaded through the V29 model class ([video_VACE/models/human_decoder_parts_v29.py](video_VACE/models/human_decoder_parts_v29.py)). It is frozen (no gradients, always in evaluation mode) and produces the initial estimate (R₀, β₀, c₀). V28 is the inherited "anchor-safe iterative geometric feedback" model ([video_VACE/models/human_decoder_parts_v28.py](video_VACE/models/human_decoder_parts_v28.py)).

**Queries.** Each frame carries 26 query tokens: 24 SMPL joint tokens (token 0 is the root), one shape token and one camera token. The shape and camera tokens are initialised as the mean of the joint tokens. A joint token is the sum of two linear encodings, plus two embeddings that are added to every token:
- the current state (rotation, betas and camera);
- the local image feedback described below;
- a learned per-token embedding;
- a sinusoidal embedding of the source-frame identifier, relative to the first valid frame of the window.

Duplicated source frames are masked, so they are not used as independent temporal evidence.

**Refinement iteration.** All iterations share the same weights. One iteration performs the following steps in order:
1. **SMPL feedback.** Run the neutral SMPL model on the current state and project the 24 joints with the crop camera (focal length 5000 px for a 256 px crop). Then bilinearly sample a 3×3 neighbourhood (radius one token cell) of the linearly projected HMR 2.0 token map around each joint ([video_VACE/models/human_decoder_parts_v27.py](video_VACE/models/human_decoder_parts_v27.py), [video_VACE/models/body_projection_v28.py](video_VACE/models/body_projection_v28.py)).
2. **Image attention.** Per-frame cross-attention of the 26 queries to the 192 image tokens, with 2D sinusoidal positions.
3. **VACE attention.** Window-level cross-attention of all T×26 queries to the VACE memory. The memory is adaptively average-pooled to 2×3 cells per latent slice (30 tokens) and carries spatial and latent-index embeddings. The latent time index is handled separately from the source-frame identity.
4. **Landmark attention.** Cross-attention to 17 landmark features produced by an observation head that attends to the image tokens.
5. **Dual-order blocks.** `depth` blocks, each computing a spatial-then-temporal branch and a temporal-then-spatial branch. Every axis attention is a pre-norm multi-head self-attention followed by an MLP. The two branches are mixed by a per-token learned softmax gate. Temporal attention masks invalid and duplicated frames.
6. **Output heads.** Separate heads predict root 6D rotation (token 0), body 6D rotations (tokens 1-23), shape (token 24) and camera (token 25):
   - The shape correction is Δβ = 3·tanh(s + sₜ), where s is a 10-dimensional term averaged over the independent frames of the window and sₜ is a 10-dimensional per-frame term.
   - The camera correction is a tanh-bounded log-scale (±0.5) and translation (±0.5).
   - All four heads are zero-initialised.
7. **Residual update.** R = rot6d(I₆ + ΔR) · R₀, s = s₀·exp(Δs), t = t₀ + Δt and β = β₀ + Δβ. Invalid frames keep the reference values. Each iteration re-predicts the full residual with respect to the frozen estimate, and the resulting state drives the SMPL feedback of the next iteration. Unlike the inherited models, the rotation residual is not restricted by a small-angle gate.

Because the output heads are zero-initialised, the initial output equals the frozen V28 estimate. This was confirmed on a real training window with a maximum difference of 0 ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)). The attention, MLP and projection layers are randomly initialised.

**Defaults and size.** The defaults are hidden width 256, 8 attention heads, depth 4 and 3 refinement iterations, which is the value set in every shipped configuration; the trainer's command-line default is 2. The model has 28,329,887 parameters in total, including the frozen reference; the V28 model alone has 16,723,786 ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)). Windows of 16 or 32 frames are not implemented.

**Not part of the current method.** The current method does not include:
- motion pretraining;
- conditional flow or generative completion;
- end-to-end fine-tuning of the visual backbones;
- gravity or world-trajectory estimation;
- distillation from a geometry teacher (see [Auxiliary research tracks](#auxiliary-research-tracks)).

### Training objective and checkpoint selection

The example V30 configuration ([configs/star_body_v30.example.json](configs/star_body_v30.example.json)) trains with pure 3D geometry supervision on the 3DPW training split. 3D ground truth enters only the losses, never the model inputs ([docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)). The person crops given to HMR 2.0a and VACE, however, are oracle crops computed per frame from the 3DPW `poses2d` annotations with a 25 % margin (VACE cache view protocol `oracle_person_tube_crop_before_vace_v1`), not detector boxes.

**Disabled auxiliary losses.** The PoseTrack and COCO 2D losses exist but are disabled: `pose-2d-weight` is 0, `coco-weight` is 0, and `geometry-only-steps` is 6001, which exceeds `max-steps` of 6000.

**Geometry loss and weights.** The geometry loss is the inherited V27 loss with its anchor term set to zero ([video_VACE/divibody_training/ram_parts_runtime_v27.py](video_VACE/divibody_training/ram_parts_runtime_v27.py)). The runtime overrides the configured `anchor-weight` of 0.1 to 0 for the 3D loss; the configured value is used only by the 2D pose loss, which is inactive because `pose-2d-weight` is 0. In the V30 step engine, the geometry loss that is executed is the code-identical copy in [ram_parts_runtime_v29.py](video_VACE/divibody_training/ram_parts_runtime_v29.py), called through the V29 boundary-partition wrapper, because the engine's import from [v29_boundary_partition.py](video_VACE/divibody_training/v29_boundary_partition.py) shadows the one from [ram_parts_runtime_v30.py](video_VACE/divibody_training/ram_parts_runtime_v30.py) ([v30_training_engine.py](video_VACE/divibody_training/v30_training_engine.py)). Its terms and weights are:

| Term | Weight |
|---|---|
| Pelvis-aligned J14 joint error | 1.0 |
| SMPL vertex error | 1.0 |
| Rotation error | 0.05 |
| Betas error | 0.001 |
| Velocity error | 0.5 |
| Within-window acceleration (canonical frame-owner triplets) | 3.0128553000311373 (base weight; see note below) |
| Cross-window boundary acceleration (2 triplets per step, deterministic cyclic sampling of the audited training pool) | 0.9871446999688626 |
| Update regularisation | 0.001 |
| Loss on intermediate iterations | 0.25 |
| Within-window shape-consistency (variance of betas) | 0.01 |

The two acceleration weights sum to a total motion weight of 4.0, which is split by class fraction ([video_VACE/divibody_training/v29_boundary_partition.py](video_VACE/divibody_training/v29_boundary_partition.py), [video_VACE/divibody_training/v29_boundary_training.py](video_VACE/divibody_training/v29_boundary_training.py), [scripts/prepare_rebuilt_experiments_v31.py](scripts/prepare_rebuilt_experiments_v31.py)). The within-window acceleration weight is a base weight: at each step the boundary-partition wrapper multiplies it by the uniform-window factor n·W/(B·N), where n is the number of canonical within-window triplets in the minibatch, W the number of train windows, B the number of windows in the minibatch and N the size of the canonical within-window pool ([v29_boundary_partition.py](video_VACE/divibody_training/v29_boundary_partition.py); likewise in [v31_boundary_partition.py](video_VACE/divibody_training/v31_boundary_partition.py)).

**Optimisation.** Only the new modules are optimised ([video_VACE/divibody_training/parts_training_utils_v30.py](video_VACE/divibody_training/parts_training_utils_v30.py)), with these settings:
- AdamW, with learning rate 1e-4 for the geometry and shape parameters, 1e-4 for the observation modules and 1e-5 for the camera head;
- weight decay 1e-4, except for biases and other one-dimensional parameters;
- 200 linear warm-up steps followed by a cosine decay to 0.1 of the base rate;
- gradient-norm clipping at 1.0 and an EMA decay of 0.995;
- a batch of 4 windows, 6,000 steps and seed 2026;
- BF16 autocast by default, with SMPL, projection and rotation composition kept in FP32.

**Checkpoint selection.** The initial model is evaluated at step 0, and both the student and the EMA weights every 1,000 steps, on the full validation split with 10,413 person-frames and 10,226 acceleration triplets ([video_VACE/divibody_training/v30_training_engine.py](video_VACE/divibody_training/v30_training_engine.py), [video_VACE/divibody_training/v29_geometry_policy_stable.py](video_VACE/divibody_training/v29_geometry_policy_stable.py); counts from [evidence/v30_validation_summary.json](evidence/v30_validation_summary.json) and the coverage gate in [scripts/evaluate_body_v31.py](scripts/evaluate_body_v31.py)). A checkpoint is accepted only if two conditions hold:
- MPJPE, PA-MPJPE and PVE are all strictly below the frozen V28 values measured at step 0;
- the acceleration error does not exceed a ceiling. For V30 this ceiling is the historical V28 validation value of 22.329070147964064 mm/frame², which is specific to this project's cache protocol.

Among the accepted checkpoints, the one that maximises the negative sum of the baseline-normalised MPJPE, PA-MPJPE and PVE is kept. The test split is never used for selection.

**Ablation switches.** The flags `--disable-temporal`, `--disable-local-feedback` and `--disable-vace` remove only the corresponding new modules. Because the frozen reference retains its own temporal and VACE modelling, `--disable-temporal` is not a model without temporal modelling ([docs/REPRODUCTION_ZH.md](docs/REPRODUCTION_ZH.md)).

### Changes in V31

V31 was produced by copying the V30 sources before editing; [provenance/v31_model_copy.json](provenance/v31_model_copy.json) and [provenance/v31_training_copy_manifest_v1.json](provenance/v31_training_copy_manifest_v1.json) record each copy. The model's docstring states that its options "have no demonstrated reconstruction-metric benefit" ([video_VACE/models/human_decoder_spatiotemporal_v31.py](video_VACE/models/human_decoder_spatiotemporal_v31.py)). The changes are:

1. **Robust 6D rotation.** The V30 `rotation_6d` can return non-orthogonal matrices for collinear or nearly collinear FP32 inputs. V31 rescales each 3-vector by its largest component and rebuilds the orthonormal basis from normalised cross products. In CPU synthetic tests the maximum orthogonality error falls from about 1 to 4.77e-7, with finite gradients, as reported in [docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md); the result file `model_math_v31.json` is not included, and the comparison can be regenerated with [scripts/verify_model_math_v31.py](scripts/verify_model_math_v31.py). This corrects numerical validity; it is not a demonstrated accuracy improvement.
2. **Opt-in ablations.** V31 adds two settings whose defaults reproduce the V30 functions exactly:
   - `frame-shape-delta-scale` in [0, 1] (default 1) scales the per-frame shape term. Setting it to 0 does not make the total shape constant, because the V28 initial betas vary per frame.
   - `local-radius-schedule` sets one sampling radius per iteration. The default is empty, meaning radius 1; the registered candidate is [2.0, 1.0, 0.5].
3. **Loss routing.** In the V30 step engine, the geometry loss is supplied by the V29 boundary-partition wrapper. V31 routes the full chain to its own runtime ([video_VACE/divibody_training/v31_boundary_partition.py](video_VACE/divibody_training/v31_boundary_partition.py)). The V30 losses are kept by default, and the old routing is documented as mathematically equivalent ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)). A deduplicated shape-consistency loss, `--deduplicate-shape-consistency`, is available as an opt-in.
4. **Assets and resume provenance.** PoseTrack caches are no longer required when pose training and pose validation are disabled. Source, asset and environment fingerprints are bound into the resume state. Full-cache SHA-256 verification (`--verify-cache-sha256`) and a torch-free `--dry-run` are also available ([video_VACE/divibody_training/v31_training_contract.py](video_VACE/divibody_training/v31_training_contract.py), [scripts/train_from_config_v31.py](scripts/train_from_config_v31.py)).
5. **Launcher and dependency fixes.** [scripts/run_star_body_v31.sh](scripts/run_star_body_v31.sh) no longer places `video_VACE/vace/models` on `PYTHONPATH`. That directory can shadow the external Wan2.1 `wan` package; the first real V31 smoke check failed with the resulting circular import. [requirements-star-body-v31.txt](requirements-star-body-v31.txt) adds `chumpy==0.70`, which is needed to read legacy SMPL pickles.
6. **Checkpoint format.** The V31 format is `spatial-dual-order-smpl-feedback-v31`. V31 loads only native V31 checkpoints or V28/V29 initialisations; native V30 checkpoints cannot be migrated to V31.

### Component map

| Component | V30 (evaluated) | V31 (development) |
|---|---|---|
| Decoder, dual-order blocks, heads, `rotation_6d` | [human_decoder_spatiotemporal_v30.py](video_VACE/models/human_decoder_spatiotemporal_v30.py) | [human_decoder_spatiotemporal_v31.py](video_VACE/models/human_decoder_spatiotemporal_v31.py) |
| Frozen-reference loader, observation head, projection and sampling helpers | [human_decoder_parts_v29.py](video_VACE/models/human_decoder_parts_v29.py), with the inherited [human_decoder_parts_v27.py](video_VACE/models/human_decoder_parts_v27.py) and [body_projection_v28.py](video_VACE/models/body_projection_v28.py) | same |
| Dataset wrapper, forward pass, geometry losses | [ram_parts_runtime_v30.py](video_VACE/divibody_training/ram_parts_runtime_v30.py) (dataset wrapper, forward pass); geometry loss as executed by the V30 engine: [ram_parts_runtime_v29.py](video_VACE/divibody_training/ram_parts_runtime_v29.py) via `v29_boundary_partition.py` (code-identical to the V30 runtime's loss) | [ram_parts_runtime_v31.py](video_VACE/divibody_training/ram_parts_runtime_v31.py) |
| Within-window and boundary motion supervision | [v29_boundary_partition.py](video_VACE/divibody_training/v29_boundary_partition.py), [v29_boundary_training.py](video_VACE/divibody_training/v29_boundary_training.py) | [v31_boundary_partition.py](video_VACE/divibody_training/v31_boundary_partition.py), [v29_boundary_training.py](video_VACE/divibody_training/v29_boundary_training.py) |
| Optimiser, schedule, EMA | [parts_training_utils_v30.py](video_VACE/divibody_training/parts_training_utils_v30.py) | [parts_training_utils_v31.py](video_VACE/divibody_training/parts_training_utils_v31.py) |
| Step engine and checkpoint selection | [v30_training_engine.py](video_VACE/divibody_training/v30_training_engine.py), [v29_geometry_policy_stable.py](video_VACE/divibody_training/v29_geometry_policy_stable.py) | [v31_training_engine.py](video_VACE/divibody_training/v31_training_engine.py), [v29_geometry_policy_stable.py](video_VACE/divibody_training/v29_geometry_policy_stable.py) |
| Training entry point | [train_spatiotemporal_v30_v3.py](video_VACE/divibody_training/train_spatiotemporal_v30_v3.py) | [train_spatiotemporal_v31.py](video_VACE/divibody_training/train_spatiotemporal_v31.py), [v31_training_contract.py](video_VACE/divibody_training/v31_training_contract.py) |
| In-training validation metrics | [v29_sequence_evaluation_report.py](video_VACE/divibody_training/v29_sequence_evaluation_report.py) | same |
| Standalone evaluator | None at V30. [scripts/evaluate_body_v31.py](scripts/evaluate_body_v31.py), added with V31, can evaluate V30 checkpoints but produced none of the recorded V30 figures. | [scripts/evaluate_body_v31.py](scripts/evaluate_body_v31.py), [body_sequence_evaluation_v31.py](video_VACE/divibody_training/body_sequence_evaluation_v31.py) (covers V28, V30 and V31; no full-split result recorded) |
| JSON configuration launcher | [scripts/train_from_config.py](scripts/train_from_config.py) (publication wrapper; no recorded training run) | [scripts/train_from_config_v31.py](scripts/train_from_config_v31.py) |

`train_spatiotemporal_v30_v3.py` is the only V30 trainer used for recorded training; [train_spatiotemporal_v30_rebuild.py](video_VACE/divibody_training/train_spatiotemporal_v30_rebuild.py) is the (untrained) rebuild entry described under [Usage](#usage). Two other V30-named files are kept unchanged and are not entry points:
- [train_spatiotemporal_v30.py](video_VACE/divibody_training/train_spatiotemporal_v30.py) is a byte-identical copy of `train_parts_v29_stitched.py`, left over from a failed first builder, and does not train V30.
- [train_spatiotemporal_v30_v2.py](video_VACE/divibody_training/train_spatiotemporal_v30_v2.py) rejects a zero pose weight.

## Repository structure

```text
.
├── video_VACE/
│   ├── models/                  Model definitions: inherited originals, V29, V30 and V31
│   ├── divibody_training/       Runtimes, trainers, step engines, cache builders, evaluation helpers
│   ├── vace/                    Upstream VACE source code (retained; no upstream commit or diff recorded),
│   │                            plus two research copies for feature capture
│   └── README.md, UserGuide.md, LICENSE.txt, requirements*   Upstream VACE documentation and licence
├── scripts/
│   ├── run_star_body.sh                  V30 release launcher (sets PYTHONPATH)
│   ├── run_star_body_v31.sh              Corrected launcher (recommended)
│   ├── train_from_config*.py             JSON-configuration launchers (V30, V31, V30 rebuild)
│   ├── evaluate_body_v31.py              Owned-frame evaluator for V28/V30/V31
│   ├── verify_*_v31.py                   Verification tools
│   ├── prepare_rebuilt_experiments_*.py  Preregistration of matched experiments
│   ├── v29/, v30/                        Historical research tools and server controllers
│   └── ...                               Pipeline and environment tools
├── configs/                     Example configurations and the registered V31 experiment plan
├── docs/                        Design, reproduction and status documents (in Chinese)
├── evidence/                    Redacted summary of the V30 validation evidence
├── provenance/                  Copy-before-edit and hash manifests
├── runs/                        Auxiliary engineering tools (28 scripts; no results)
├── original_code_hashes.json    SHA-256 of the 230 inherited Python files
├── requirements-star-body.txt, requirements-star-body-v31.txt
└── AGENTS.md, THIRD_PARTY_NOTICES.md, LICENSE.txt
```

The [runs/](runs/) directory contains engineering tools from the cache-rebuild and reproducibility work. None of them produce or claim model metrics, and none of their outcomes are recorded in this repository:
- [runs/phase4_singlefile_tools_001/](runs/phase4_singlefile_tools_001/) and [runs/phase4_singlefile_tools_002/](runs/phase4_singlefile_tools_002/) recover the rebuilt HMR sidecar into a single file on the CPU; the second set adds a reversible CRC-only ZIP repair of a completely filled partial, and no tensor bytes are rewritten.
- [runs/phase4_v30_smoke_tools_001/](runs/phase4_v30_smoke_tools_001/) holds a non-executable plan and a review draft for one finite V30 smoke pair.
- [runs/phase5_determinism_tools_001/](runs/phase5_determinism_tools_001/) and [runs/phase5_determinism_tools_002/](runs/phase5_determinism_tools_002/) hold bounded determinism probes. They include a CPU-FP32 V30 variant that is explicitly a new numerical protocol, not a bitwise reproduction of the original CUDA run.
- [runs/phase5_guard_tools_001/](runs/phase5_guard_tools_001/) holds a Linux memory and disk guard wrapper.

Several of these controllers depend on server paths and files that are not in the repository.

### Code lineage and version numbering

The file version numbers come from several sources and do not form a single sequence:

- **Inherited DiviBody code.** These are the 230 Python files pinned in [original_code_hashes.json](original_code_hashes.json): 202 under `video_VACE/divibody_training/` and 28 under `video_VACE/models/`. They are byte-identical to their recorded hashes and cover the inherited chain up to V28. They also include files from an older, independent numbering series whose numbers overlap with later STaR-Body version numbers. Examples are `autoresearch_body_v29.py` and `train_parts_v29_research.py` (inherited files that are distinct from the V29 stage below), `train_parts_v31_routed.py` (a routed-gradient study for the V28 refiner whose docstring refers to that older series' own v29/v30), and files numbered v33, v35, v36, v39, v39b, v41, v41b and v42. Those numbers do not refer to STaR-Body versions.
- **Upstream VACE.** `video_VACE/vace/` and the VACE documentation, licence and requirement files, except two research copies that add feature capture, [video_VACE/vace/models/wan/wan_vace_feature.py](video_VACE/vace/models/wan/wan_vace_feature.py) ("an isolated copy of the original pipeline") and [video_VACE/vace/vace_wan_feature_inference.py](video_VACE/vace/vace_wan_feature_inference.py), which imports it. These two files are retained from the original code base; they are hashed in [provenance/publication_source_hashes.json](provenance/publication_source_hashes.json) but are not covered by [original_code_hashes.json](original_code_hashes.json).
- **V29 stage.** These are predecessor files added before V30, including the V29X/V29Y strong-RGB variants and [scripts/v29/](scripts/v29/). They are not in the original manifest (the two v29-numbered files listed above belong to the inherited series, not to this stage), and the V30 snapshot policy lists them as new V29/V30 filenames ([provenance/publication_source_hashes.json](provenance/publication_source_hashes.json)). The V29 model class is used to load the frozen V28 reference.
- **V30.** The STaR-Body architecture: the model, runtime, optimiser utilities, trainers, step engine, [scripts/v30/](scripts/v30/) and the V30 release launchers. [provenance/copy_manifest_v2.json](provenance/copy_manifest_v2.json) records the derivation of five V30 files from V29 counterparts: the model, runtime, optimiser utilities, step engine and the superseded trainer `train_spatiotemporal_v30_v2.py`. The derivation of the effective trainer `train_spatiotemporal_v30_v3.py` is not recorded.
- **V31 and later infrastructure.** These are the 79 tracked files that are not listed in the V30 snapshot inventory (the inventory file itself excluded):
  - the V31 model and training chain;
  - the evaluator and verification tools;
  - the resumable cache builders;
  - the V30 rebuild chain;
  - [runs/](runs/);
  - the V31 configurations and experiment plan, [docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md), the V31-era provenance records, [requirements-star-body-v31.txt](requirements-star-body-v31.txt), `scripts/.gitignore`, the server pipeline and paused-stage tools, and the `v32` storage variants.

  The `v32`-named files ([cache_hmr2a_rgb_anchor_v32_scratch.py](video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v32_scratch.py), [prepare_rebuilt_experiments_v32_storage.py](scripts/prepare_rebuilt_experiments_v32_storage.py)) are storage-only variants of V31 tooling with unchanged encoding and tensor layout; there is no V32 model.

## Requirements and installation

The original experiments used Linux, Python 3.10, PyTorch with CUDA, and real SMPL assets ([docs/REPRODUCTION_ZH.md](docs/REPRODUCTION_ZH.md)). The V31 work used a server with Python 3.10.20, PyTorch 2.5.1+cu124, NumPy 1.26.4 and an NVIDIA RTX A6000. That configuration is an observed inventory, not an environment lock verified from scratch ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)).

```bash
python -m venv env
source env/bin/activate
pip install -r requirements-star-body.txt        # PyTorch/CUDA and flash-attn builds must match the machine
pip install --no-build-isolation chumpy==0.70    # reads legacy SMPL pickles; see requirements-star-body-v31.txt
```

About the requirement files:
- [requirements-star-body-v31.txt](requirements-star-body-v31.txt) is [requirements-star-body.txt](requirements-star-body.txt) plus `chumpy==0.70`. Its comment states that recent pip versions need `--no-build-isolation` for chumpy, so chumpy is installed as a separate step above.
- Both files include the unchanged upstream VACE requirements ([video_VACE/requirements.txt](video_VACE/requirements.txt), [video_VACE/requirements/framework.txt](video_VACE/requirements/framework.txt)), which pin `numpy<2` and list `flash_attn` and `onnxruntime-gpu`.
- The CUDA and `flash-attn` builds must match the target machine.
- These files are dependency entry points, not lock files.
- The installation commands were not executed for publication.

Two external source trees (Wan2.1 and 4D-Humans) must be installed separately and made importable. The launchers read the following environment variables:

| Variable | Points to |
|---|---|
| `WAN_SOURCE_ROOT` | Root of a Wan2.1 checkout that contains the `wan` package |
| `HMR2_SOURCE_ROOT` | Root of a 4D-Humans checkout that contains the `hmr2` package |
| `DIVIBODY_SOURCE_ROOT` | Original source directory, required by some legacy cache checks |
| `STAR_BODY_PYTHON` | Optional interpreter for the launchers (default: `python`) |
| `V28D_MEMORY_LIMIT_GIB` | Optional per-process CUDA allocator cap used by the training GPU entry (default: 10) |

The launchers prepend `video_VACE`, `video_VACE/divibody_training` and the repository root to `PYTHONPATH` (the V30 release launcher `scripts/run_star_body.sh` also adds `video_VACE/vace/models`), then append `WAN_SOURCE_ROOT` and `HMR2_SOURCE_ROOT` when they are set. They also set `PYTHONDONTWRITEBYTECODE=1` and run from the repository root.

## Data and external assets

No dataset, body model, pretrained or trained weight, feature cache, credential or machine-local configuration is distributed ([AGENTS.md](AGENTS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). There are two reasons:
- The datasets, body models and third-party weights are subject to their own licences.
- The caches are large derived artefacts: the four core train and validation caches hold about 18 GiB of tensors, and 30-40 GiB of free disk should be reserved.

The trained V30 weights were not recovered and cannot be provided.

| Configuration key | Required content |
|---|---|
| `train-vace-cache`, `val-vace-cache` | Complete VACE cache for the split: format `divibody-contiguous-vace-video-features-v5` (train, stride 8) or `-v6-stride4` (validation, stride 4). Contains BF16 features `[windows, 5, 6, 6, 1536]`, real metadata, the Boolean masks `target_valid_mask`/`frame_owner_mask` `[·, 8]`, `velocity_owner_mask` `[·, 7]` and `acceleration_owner_mask` `[·, 6]`, and labels. |
| `train-hmr-sidecar`, `val-hmr-sidecar` | HMR 2.0a sidecar `divibody-hmr2a-anchor-sidecar-v8`, matching the VACE UIDs and windows exactly. Contains BF16 `rgb_tokens` `[frames, 192, 1280]` and the initial `base_rotmat`, `base_betas` and `base_camera`. Built with the HMR 2.0a checkpoint of SHA-256 `2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e`. |
| `pose-*-vace-dir`, `pose-*-hmr-dir` | PoseTrack shard caches. The V30 v3 trainer requires them even when the pose weight is 0; they are optional in V31 and in the V30 rebuild. |
| `init-checkpoint` | Audited V28/V29 initialisation. Formal runs start from the original V28 checkpoint; the registered plan fixes its SHA-256 as `4254e31c11d6c44f362fe26934cd04ed8f2d37f2b513ef9656a54ccfddb750bd` (see the note below the table). |
| `smpl-model-root` | `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`, with `SMPL_to_J19.pkl` in the parent directory. SMPL-H, MHR and SMPL-X are not interchangeable with SMPL. |
| `joint-regressor` | The H36M joint regressor `.npy`, of shape `[17, 6890]`. |
| `validation-manifest` | Audited full validation identities in format `v28d-validation-reference-v1`. It is built by the inherited [v28d_validation_audit.py](video_VACE/divibody_training/v28d_validation_audit.py) from the official 3DPW annotation archive (https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip). |

The frozen V28 initialisation (SHA-256 `4254e31c…750bd`) is a checkpoint of the inherited code base ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)). It is not distributed, and this repository documents no recipe that regenerates it: the inherited V28 trainers and launchers are present (e.g. [train_parts_v28.py](video_VACE/divibody_training/train_parts_v28.py)), but the run that produced this checkpoint, and hence its training budget, is not recorded. Without this checkpoint, V30 and V31 cannot be trained or evaluated as registered, and a substitute V28 makes results non-comparable with those reported here.

The person-frame and acceleration-triplet counts below are formal gates enforced by [scripts/evaluate_body_v31.py](scripts/evaluate_body_v31.py) (validation, test) and [scripts/prepare_rebuilt_experiments_v31.py](scripts/prepare_rebuilt_experiments_v31.py) (train, which also checks stride 8 for train and stride 4 for validation). The window counts are the expected rebuild sizes from [docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md); these two scripts do not check them, but [scripts/verify_rebuild_stage_v31.py](scripts/verify_rebuild_stage_v31.py) does when given `--expected-windows`, as in the stages written by [scripts/generate_server_pipeline_v31.py](scripts/generate_server_pipeline_v31.py). A formal run refuses development-only caches.

| Split | Person-frames | Acceleration triplets | Windows |
|---|---|---|---|
| Train | 22,735 | 22,481 | 2,909 (stride 8) |
| Validation | 10,413 | 10,226 | 2,578 (stride 4) |
| Test | 35,515 | 34,752 | — |

The caches can be rebuilt with two resumable builders ([video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py](video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py), [video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v31_resume.py](video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v31_resume.py)). They require:
- the 3DPW data;
- the Wan2.1-VACE-1.3B checkpoint directory;
- the 4D-Humans HMR 2.0a checkpoint (`logs/train/multiruns/hmr2a/0/checkpoints/epoch=10-step=25000.ckpt` with its `model_config.yaml`).

The following invocations follow the stages written by [scripts/generate_server_pipeline_v31.py](scripts/generate_server_pipeline_v31.py) for the validation split; their outcomes are not recorded in this repository:

```text
bash scripts/run_star_body_v31.sh video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py \
  --data-root 3DPW_DIR --split validation --target-stride 4 \
  --vace-checkpoint-dir WAN2.1_VACE_1.3B_DIR --output CACHE_DIR/vace_validation_s4.pt
bash scripts/run_star_body_v31.sh video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v31_resume.py \
  --vace-cache CACHE_DIR/vace_validation_s4.pt --data-root 3DPW_DIR --hmr-source-root 4D_HUMANS_DIR \
  --hmr2a-checkpoint 4D_HUMANS_DIR/logs/train/multiruns/hmr2a/0/checkpoints/epoch=10-step=25000.ckpt \
  --hmr-model-config 4D_HUMANS_DIR/logs/train/multiruns/hmr2a/0/model_config.yaml \
  --hmr-data-root 4D_HUMANS_DIR/data --output CACHE_DIR/hmr_validation_s4_raw25.pt
```

The train caches are built in the same way with `--split train --target-stride 8` and the names `vace_train_s8.pt` and `hmr_train_s8_raw25.pt`. These four file names are required by the preregistration script (see [Usage](#usage)). The complete stage order (validation caches, V28 baseline, train caches, preregistration, pause/resume acceptance, formal training and evaluation) is the one generated by [scripts/generate_server_pipeline_v31.py](scripts/generate_server_pipeline_v31.py).

The hashes of the historical caches were not recovered. Rebuilt caches therefore cannot be claimed to reproduce the historical experiments byte for byte ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)).

## Usage

All commands are run from the repository root. Copy an example configuration to a `*.local.json` file, which Git ignores, and replace the placeholder paths with local assets. Each training run needs a new, empty `output-dir` unless it resumes from a checkpoint; existing runs are never overwritten. To check an installation without any assets, run `python scripts/verify_model_math_v31.py --output NEW_REPORT.json` (CPU only, requires only PyTorch).

### Training V30 (publication wrapper around the trainer used for the recorded run)

Original release command:

```bash
cp configs/star_body_v30.example.json configs/star_body_v30.local.json
# edit cache, SMPL, regressor and initialisation paths
export WAN_SOURCE_ROOT=/path/to/Wan2.1
export HMR2_SOURCE_ROOT=/path/to/4D-Humans
export DIVIBODY_SOURCE_ROOT=/path/to/original/divibody_vace

CUDA_VISIBLE_DEVICES=0 bash scripts/run_star_body.sh \
  scripts/train_from_config.py --config configs/star_body_v30.local.json
```

[scripts/train_from_config.py](scripts/train_from_config.py) is a wrapper added for publication. It translates the JSON configuration into the CLI of `divibody_training.train_spatiotemporal_v30_v3` and runs it through `divibody_training.v28d_gpu_entry`. The recorded run did not use this wrapper: it was started by a server controller, the latest of which in this repository is [scripts/v30/launch_v30_architecture_v3.py](scripts/v30/launch_v30_architecture_v3.py); it calls the same `v28d_gpu_entry` → `train_spatiotemporal_v30_v3` entry directly, with arguments built from a server plan file that is not in this repository. The example configuration is derived from the registered `spatial_motion` arm ([docs/REPRODUCTION_ZH.md](docs/REPRODUCTION_ZH.md)); its equivalence with the registered arguments cannot be checked here. No training run with this wrapper is recorded, and [docs/REPRODUCTION_ZH.md](docs/REPRODUCTION_ZH.md) states that it has not been verified by training on a fresh checkout.

The release launcher `scripts/run_star_body.sh` adds `video_VACE/vace/models` to `PYTHONPATH`, which can shadow the external `wan` package (see [Changes in V31](#changes-in-v31)). The corrected launcher differs functionally only in omitting that path ([scripts/run_star_body.sh](scripts/run_star_body.sh) line 5, [scripts/run_star_body_v31.sh](scripts/run_star_body_v31.sh) line 7). The same wrapper can be started with `CUDA_VISIBLE_DEVICES=0 bash scripts/run_star_body_v31.sh scripts/train_from_config.py --config configs/star_body_v30.local.json`; this variant has not been executed either.

### Training V31

```bash
cp configs/star_body_v31.compat.example.json configs/star_body_v31.compat.local.json
# validate arguments and asset paths without importing torch
bash scripts/run_star_body_v31.sh scripts/train_from_config_v31.py \
  --config configs/star_body_v31.compat.local.json --dry-run
# train
CUDA_VISIBLE_DEVICES=0 bash scripts/run_star_body_v31.sh scripts/train_from_config_v31.py \
  --config configs/star_body_v31.compat.local.json
```

`--dry-run` exits with status 0 when the asset preflight passes and 2 otherwise. Three example configurations select the registered V31 arms:
- [configs/star_body_v31.compat.example.json](configs/star_body_v31.compat.example.json): V30-compatible defaults;
- [configs/star_body_v31.shared_shape.example.json](configs/star_body_v31.shared_shape.example.json): `frame-shape-delta-scale` 0;
- [configs/star_body_v31.local_schedule.example.json](configs/star_body_v31.local_schedule.example.json): radius schedule [2.0, 1.0, 0.5].

These examples still carry the historical acceleration ceiling 22.329070147964064. The registered plan requires that formal runs use a ceiling re-measured from a new full V28 validation ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)) at the same precision as the experiment ([scripts/prepare_rebuilt_experiments_v31.py](scripts/prepare_rebuilt_experiments_v31.py)). The preregistration script below sets this ceiling automatically.

### Rebuilding V30 under the V31 provenance infrastructure

```bash
cp configs/star_body_v30_rebuild.example.json configs/star_body_v30_rebuild.local.json
# edit paths; set baseline-report to an existing V28 report
bash scripts/run_star_body_v31.sh scripts/train_from_config_v30_rebuild.py \
  --config configs/star_body_v30_rebuild.local.json --dry-run
CUDA_VISIBLE_DEVICES=0 bash scripts/run_star_body_v31.sh scripts/train_from_config_v30_rebuild.py \
  --config configs/star_body_v30_rebuild.local.json
```

This command runs the unchanged V30 model, runtime and optimiser. The V31-only switches are rejected, and PoseTrack is optional. A formal run requires two settings:
- `baseline-report`: a complete V28 validation report produced by `evaluate_body_v31.py` with `--amp-dtype bf16 --hash-large-caches` (see [Evaluation](#evaluation)), from which the acceleration ceiling is derived. The example path, `evidence/v28_full_validation_rebuilt.json`, does not yet exist; the report must first be produced.
- `verify-cache-sha256`.

### Preregistering the matched experiments

```bash
bash scripts/run_star_body_v31.sh scripts/prepare_rebuilt_experiments_v31.py \
  --cache-root CACHE_DIR --baseline-report V28_VALIDATION.json \
  --config-dir NEW_CONFIG_DIR --runs-root RUNS_DIR \
  --init-checkpoint V28.pt --smpl-model-root SMPL_DIR \
  --joint-regressor J_regressor_h36m.npy --validation-manifest validation_reference.json \
  --amp-dtype bf16   # default; must equal the AMP mode of the baseline report
```

`CACHE_DIR` must contain exactly `vace_train_s8.pt`, `hmr_train_s8_raw25.pt`, `vace_validation_s4.pt` and `hmr_validation_s4_raw25.pt` (train stride 8, validation stride 4). `--amp-dtype` accepts `bf16` (default) or `none` and must equal the precision recorded in the V28 baseline report. Because `evaluate_body_v31.py` defaults to `none`, either evaluate the baseline with `--amp-dtype bf16` or pass `--amp-dtype none` here. The baseline must also have been evaluated with `--hash-large-caches`, because the baseline binding requires full cache SHA-256 ([video_VACE/divibody_training/v30_rebuild_training_contract.py](video_VACE/divibody_training/v30_rebuild_training_contract.py)).

This is a CPU-only step that does not start training. From complete rebuilt caches and a complete V28 validation report, it writes the following:
- configurations for the four formal arms: `v30_rebuild`, `v31_robust`, `v31_shared_shape` and `v31_local_schedule`;
- six four-step pause/resume acceptance configurations;
- an experiment plan.

It derives the motion weights from the class-fraction split of 4.0. The arm names correspond to `v30_matched`, `v31_robust_rotation`, `v31_no_frame_shape_delta` and `v31_local_radius_schedule` in [configs/v31_experiment_plan.json](configs/v31_experiment_plan.json). Pause/resume acceptance uses [scripts/run_paused_training_stage_v31.py](scripts/run_paused_training_stage_v31.py) and [scripts/verify_training_resume_v31.py](scripts/verify_training_resume_v31.py).

### Evaluation

Synopsis (braces list alternatives, brackets mark optional flags):

```text
bash scripts/run_star_body_v31.sh scripts/evaluate_body_v31.py \
  --model-version {v28,v30,v31} --checkpoint CHECKPOINT.pt --checkpoint-weights {model,ema_model} \
  --split validation --vace-cache VACE_VALIDATION.pt --hmr-sidecar HMR_VALIDATION.pt \
  --coverage-manifest validation_reference.json \
  --smpl-model-root SMPL_DIR --joint-regressor J_regressor_h36m.npy \
  --output NEW_RESULT.json [--amp-dtype {bf16,none}] [--hash-large-caches]
```

For the V28 baseline that is passed to `prepare_rebuilt_experiments_v31.py` or to the V30 rebuild (`baseline-report`), run the evaluator with `--model-version v28 --checkpoint-weights model --split validation --amp-dtype bf16 --hash-large-caches`. Both consumers default to BF16 and reject a baseline in another AMP mode, and both reject a baseline without full cache SHA-256 ([scripts/generate_server_pipeline_v31.py](scripts/generate_server_pipeline_v31.py) uses these flags).

The evaluator performs no optimisation and no label-based selection. Its main rules are:
- A formal run requires `--coverage-manifest` and full coverage.
- `--development-only` accepts partial caches, but the output is then labelled `diagnostic_subset`.
- The test split additionally requires a separate UID-indexed `--gt-cache`.
- `--output` must not already exist.
- The default precision is `--amp-dtype none` (FP32), while the trainer defaults to BF16 autocast. A V28 baseline and the runs compared against it must use the same mode.

### Verification

| Tool | Purpose |
|---|---|
| [scripts/verify_model_math_v31.py](scripts/verify_model_math_v31.py) | CPU-only synthetic checks of the V31 rotation, shape, radius, configuration round-trip and masked-attention logic against V30. It requires only PyTorch and makes no metric claim. |
| [scripts/verify_real_v31.py](scripts/verify_real_v31.py) | Bounded check on one real training window: forward and backward pass, one optimiser step, exact disk restore and rejection of a corrupted checkpoint. It is not a benchmark and reads no validation or test data. [scripts/verify_real_precision_v31.py](scripts/verify_real_precision_v31.py) adds `--amp-dtype`. |
| [scripts/verify_cache_rebuild_v31.py](scripts/verify_cache_rebuild_v31.py), [scripts/verify_rebuild_stage_v31.py](scripts/verify_rebuild_stage_v31.py) | Read-only UID-wise parity and acceptance checks for rebuilt caches and evaluation outputs. |
| [scripts/verify_experiment_stage_v31.py](scripts/verify_experiment_stage_v31.py) | Read-only receipts for prepared, training, resume and evaluation stages. It does not recompute metrics. |
| [scripts/inspect_environment_v31.py](scripts/inspect_environment_v31.py) | Records the runtime, the package versions and the commits of the external sources. This is an inventory, not a lock file. |

### Historical and server-specific controllers

The following scripts are retained as research records. They depend on the original server directory layout, absolute paths, plan files or Linux-specific facilities, and they are not general entry points:
- [scripts/v29/](scripts/v29/) and [scripts/v30/](scripts/v30/), including `launch_v30_architecture_v1-3.py` and `launch_geometry_teacher_v30_v1-5.py`;
- [scripts/generate_server_pipeline_v31.py](scripts/generate_server_pipeline_v31.py), which writes a 22-stage server plan;
- [scripts/rebuild_pipeline_v31_v1.py](scripts/rebuild_pipeline_v31_v1.py), which requires `fcntl`, `/proc` and `nvidia-smi`;
- the `run_*` controllers under [runs/](runs/).

### Auxiliary research tracks

A separate geometry-teacher route is implemented in `scripts/v30/` ([docs/GEOMETRY_TEACHER_ZH.md](docs/GEOMETRY_TEACHER_ZH.md)):
- It runs SAM 3D Body on the same annotation-derived (oracle) person boxes and maps the MHR mesh to SMPL with the official surface mapping.
- An independent 400-step SMPL fit then receives only the teacher mesh and the HMR initial values. No filtering by ground-truth error is applied, and the fitted translation is stored separately from the crop camera.

The entry points are [scripts/v30/prepare_geometry_teacher_v30_v1.py](scripts/v30/prepare_geometry_teacher_v30_v1.py) (teacher extraction) and [scripts/v30/fit_geometry_teacher_v30_v1.py](scripts/v30/fit_geometry_teacher_v30_v1.py) (SMPL fit). The 64-frame teacher-prediction reproduction passed. The 400-step refit failed its element-wise rotation reproducibility check, and the controller stopped; `launch_geometry_teacher_v30_v5.py` is the last controller that ran, and it is retained in its failed state. `diagnose_geometry_repro_v30_v1.py` is prepared but has not been executed. As a result, no full teacher cache, no teacher validation score and no distillation result exist. Distillation is disabled in the registered plan.

## Results

### V30 on 3DPW validation (provisional)

Source: [evidence/v30_validation_summary.json](evidence/v30_validation_summary.json), a redacted summary of downloaded raw metrics with SHA-256 digests of the source archive and metrics file. Values are rounded to four decimals; full precision is in the JSON file.

| Step | Weights | MPJPE (mm) | PA-MPJPE (mm) | PVE (mm) | Accel. (mm/frame²) | Evidence |
|---:|---|---:|---:|---:|---:|---|
| 0 | Frozen V28 (initialisation) | 62.8595 | 38.8244 | 77.4927 | 22.3291 | Downloaded metrics |
| 1,000 | Student | 62.2863 | 38.3659 | 75.7207 | 22.2236 | Downloaded metrics |
| 1,000 | EMA | 62.5788 | 38.3228 | 74.9971 | 22.1498 | Downloaded metrics |
| 2,000 | Student | 62.2959 | 38.2825 | 74.3991 | 21.9998 | Downloaded metrics |
| 2,000 | EMA | 62.0758 | 38.1532 | 74.3238 | 22.0177 | Downloaded metrics |
| 3,000 | EMA | 61.7578 | 37.9109 | 73.7713 | 21.8073 | SSH log observation only (`metrics.jsonl`, 2026-10-07); artefact not downloaded |

Conditions and limitations:
- **Data and protocol.** 3DPW validation, 10,413 person-frames and 10,226 acceleration triplets, evaluated with the in-training validation of the V30 engine under the protocol described above. The same split is used to accept and select checkpoints in this run and has been reused for model and design decisions across earlier project versions ([v29_geometry_policy_stable.py](video_VACE/divibody_training/v29_geometry_policy_stable.py), [autoresearch_body_v29.py](video_VACE/divibody_training/autoresearch_body_v29.py), [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)). These figures are development-set estimates, not held-out or generalisation results.
- **Run.** A single seed (2026) of the registered schedule (6,000 steps, warm-up 200, cosine decay). The summary records `training_completed: false` and `test_evaluated: false`. No checkpoint after step 3,000 is recorded, and there is no evidence that either registered arm (`spatial_motion`, `no_new_temporal`) completed ([docs/EXPERIMENT_STATUS_ZH.md](docs/EXPERIMENT_STATUS_ZH.md)).
- **Uncertainty.** Only point estimates exist. No confidence interval, sequence-paired bootstrap or seed variance is available, so the differences from the step-0 row (e.g. 0.78 / 0.67 / 3.17 mm MPJPE / PA-MPJPE / PVE and 0.31 mm/frame² acceleration at the 2,000-step EMA checkpoint) are not established as reliable. The registered plan requires multiple seeds and sequence-paired uncertainty before any contribution claim ([configs/v31_experiment_plan.json](configs/v31_experiment_plan.json)).
- **Arm.** The evidence file does not name the arm. Attributing these rows to `spatial_motion` is an inference: that arm was scheduled first, and the arms ran sequentially ([scripts/v30/launch_v30_architecture_v3.py](scripts/v30/launch_v30_architecture_v3.py)).
- **Precision.** The precision mode of these evaluations is not recorded; the trainer default is BF16 autocast.
- **Reproduction.** The weights and full logs of this run were not recovered. The V31 work did not re-confirm these values ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)).

### Historical reference results (inherited V28 and V29-stage models)

The following results are recorded in [docs/EXPERIMENT_STATUS_ZH.md](docs/EXPERIMENT_STATUS_ZH.md) and [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md). The training budgets of these checkpoints are not documented in this repository.

| Model | Split | MPJPE (mm) | PA-MPJPE (mm) | PVE (mm) | Accel. (mm/frame²) |
|---|---|---:|---:|---:|---:|
| Original V28 | Validation (10,413 person-frames) | 62.8595 | 38.8244 | 77.4927 | 22.3291 |
| Original V28 | Test (35,515 person-frames, 24 sequences) | 68.3920 | 39.4807 | 80.3045 | 19.4941 |
| V29Y (global SAM features) | Test (same protocol) | 69.0898 | 39.7860 | 81.8590 | 19.5345 |

These results bear on V30 in two ways:
- V29Y improved on validation but regressed on test on every metric. Validation gains in this project have therefore not reliably transferred to test.
- [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md) also records a single-seed, 6,000-step matched validation control from the V29 stage. It reports 61.7031 / 37.9098 / 73.6943 mm without a strong global visual branch and 61.5375 / 37.9033 / 73.5444 mm with it (MPJPE / PA-MPJPE / PVE). The model variant, checkpoint type and frame count are not stated, so these figures are not directly comparable with the V30 rows. However, this control, whose two arms shared initialisation, data, learning rate, motion coefficients and a 6,000-step budget, reports validation figures of the same magnitude as the provisional V30 rows, and the registered V30 ablation arm (`no_new_temporal`) has no recorded result. This repository therefore contains no evidence that attributes V30's change relative to the frozen V28 initialisation to the new V30 modules.

### V31

No trained V31 metrics exist. The following checks were recorded on 2026-10-08 ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)):
- **Synthetic rotation checks.** The rotation-orthogonality checks summarised above passed.
- **Real 8-frame acceptance check.** It was run for both V30 and V31 on the first valid training window: seed 2026, FP32, 3 iterations, 10 GiB allocator cap. The check confirmed zero initial deviation from V28, valid SO(3) outputs and gradients, optimiser coverage, a gradient-free frozen reference, exact save/restore and rejection of a corrupted checkpoint. The single optimiser step increased the loss (V30: 0.091068 to 0.097114), so the check demonstrates function, not optimisation benefit.
- **Development-subset consistency check.** On a 16-frame validation subset, V28 and the untrained, zero-initialised V30 and V31 gave identical metrics. This confirms the consistency of initialisation and evaluation. It is not a 3DPW result and must not be compared with any other figure.

### Relation to published results

For orientation only, [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md) lists 3DPW figures reported in two papers: SAM 3D Body with DINOv3 (https://arxiv.org/html/2602.15989v1), 54.8 / 33.8 / 63.6 mm, and GVHMR (https://arxiv.org/html/2409.06662v1), 55.6 / 36.2 / 67.2 mm (MPJPE / PA-MPJPE / PVE). The 3DPW split behind these figures is not recorded in this repository, whereas every V30 figure above is on the validation split. These figures were not reproduced on the same machine or under the input protocol used here. Crops, body representation and acceleration units would have to be matched before any comparison. No comparison or ranking is claimed.

## Reproducibility and provenance

- **Immutable originals.** Model and training code is never edited in place. A new version is created by copying a file to a new name and editing only the copy ([AGENTS.md](AGENTS.md)). The copy records under [provenance/](provenance/) differ in detail:
  - Most records made from the V31 work onward give the source SHA-256 before copying, the hash of the copy before editing, the edited destination and confirmation that the source was unchanged afterwards (e.g. [provenance/v31_model_copy.json](provenance/v31_model_copy.json), [provenance/v30_rebuild_copy.json](provenance/v30_rebuild_copy.json)).
  - The V30-era [provenance/copy_manifest_v2.json](provenance/copy_manifest_v2.json) and the geometry-teacher manifests record only source and final copy hashes; so do [provenance/v31_rebuild_controller_copy.json](provenance/v31_rebuild_controller_copy.json) and the additional copy records in [provenance/v31_delivery_manifest.json](provenance/v31_delivery_manifest.json). [provenance/v31_server_pipeline_generator.json](provenance/v31_server_pipeline_generator.json) records a byte-identical import, not an edited copy.
  - copy_manifest_v2.json covers the superseded `train_spatiotemporal_v30_v2.py` rather than the effective `train_spatiotemporal_v30_v3.py`.
  - Copies made for the `runs/` tools (for example the determinism probes, whose docstrings state that they were copied before editing) and the `v32` storage builder have no copy record.
- **Hash manifests.** The repository includes the following manifests:
  - [original_code_hashes.json](original_code_hashes.json): the 230 inherited Python files.
  - [provenance/publication_source_hashes.json](provenance/publication_source_hashes.json): 431 copied files in the V30 snapshot.
  - [provenance/publication_inventory_20261008.json](provenance/publication_inventory_20261008.json): all 450 files of the V30 snapshot (excluding the inventory itself), with `asset_files_included: false` and `model_tests_executed_for_publication: false`.
  - [provenance/v31_delivery_manifest.json](provenance/v31_delivery_manifest.json): 452 intake files verified unchanged at V31 delivery, plus the SHA-256 of 25 new V31 files (`new_files`).

  At the time of this release, every file listed in these manifests still matches its recorded SHA-256, except `README.md`, which has been rewritten since the V30 snapshot inventory. The manifests do not cover every tracked file. Of the 79 files added after the V30 snapshot, 25 are pinned in `new_files` of the V31 delivery manifest and 11 others in the per-copy records under [provenance/](provenance/) (for example the V30 rebuild chain in [provenance/v30_rebuild_copy.json](provenance/v30_rebuild_copy.json)). 35 code files are not pinned by any manifest or copy record under [provenance/](provenance/): the 28 scripts under `runs/`, `scripts/prepare_rebuilt_experiments_v31.py`, `scripts/prepare_rebuilt_experiments_v32_storage.py`, `scripts/verify_cache_rebuild_v31.py`, `scripts/verify_experiment_stage_v31.py`, `scripts/verify_rebuild_stage_v31.py`, `scripts/verify_training_resume_v31.py` and `video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v32_scratch.py`. Six of these have their SHA-256 recorded only inside other `runs/` tools, which are themselves unpinned: the `v32` scratch cache builder and its byte-identical copy `runs/phase4_singlefile_tools_001/cache_hmr2a_rgb_anchor_v32_scratch.copy.py` (as `COPY_SHA` in the phase-4 recovery tools `recover_hmr_singlefile_v1.py`, `run_guarded_recovery_v1.py` and their `v2` successors), `runs/phase5_guard_tools_001/run_memory_disk_guard_v3.py` (as `GUARD_SHA` in `run_determinism_probe_v1.py` and `run_determinism_probe_v2.py`), `scripts/prepare_rebuilt_experiments_v31.py` (in `runs/phase4_v30_smoke_tools_001/test_v30_storage_plan_v1.py` and `test_v30_storage_plan_v2.py`), and `runs/phase4_singlefile_tools_001/test_hmr_singlefile_v1.py` and `test_hmr_singlefile_v2.py` (as `copy_source_sha256` in `test_hmr_singlefile_v2.py` and `runs/phase4_singlefile_tools_002/test_hmr_singlefile_v3.py`, respectively). The remaining eight are post-snapshot provenance records themselves. [.gitattributes](.gitattributes) disables line-ending conversion (`* -text`) so that the hashes remain valid on every platform.
- **Publication scope.** No model was retrained and no model tests were run for the publication of the V30 snapshot. The installation commands were not executed for publication, and no training run with the V30 release wrapper is recorded ([docs/REPRODUCTION_ZH.md](docs/REPRODUCTION_ZH.md)). The executed V31 checks are those listed under [Results](#results), plus: a formal-entry run that correctly refused the development cache (exit 1; `formal_subset_rejection.log`, not included); two first real smoke checks that failed (Wan import cycle, dynamic-module file name) and passed after fixes ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)); and the CPU, AST and CLI checks recorded in [provenance/](provenance/).
- **Determinism.** Bitwise reproduction of the historical CUDA run has not been established. The deterministic V30 variant in `runs/phase5_determinism_tools_*` defines a new numerical protocol. Native checkpoints are restored strictly, and the pause/resume tooling checks that the restored training state is exactly equal to an uninterrupted run. However, interrupted resumption of the full V31 engine had not been tested when the V31 documentation was written, and no pause/resume result is included.
- **Selection hygiene.** Checkpoints and arms are selected on validation only. The test split has been observed by earlier project models, and any future test report must disclose this. The registered plan defers test, and external validation on datasets not involved in development, until the model choice is frozen ([configs/v31_experiment_plan.json](configs/v31_experiment_plan.json), [docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md), [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)).
- **Evidence not included.** The documentation refers to several evidence files that are not part of this repository and can be cited only as reported:
  - `model_math_v31.json`, `real_v30_003.json`, `real_v31_001.json`, `environment_001.json` and `formal_subset_rejection.log`, cited in [docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md);
  - `runs/v30_real_forward_v2_20261007/result.json`, cited in [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md).

  `docs/DESIGN_ZH.md` also refers to `STATUS_ZH.md`; the corresponding document in this repository is [docs/EXPERIMENT_STATUS_ZH.md](docs/EXPERIMENT_STATUS_ZH.md).
- **Initialisation record.** The internal validation field of the V28 checkpoint refers to an older 3,162-frame set, not to the 10,413-frame protocol ([docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md)).

## Documentation index

| Document | Content |
|---|---|
| [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md) (in Chinese) | V30 design rationale, related methods with links, success criteria and research roadmap |
| [docs/REPRODUCTION_ZH.md](docs/REPRODUCTION_ZH.md) (in Chinese) | Release scope, environment, asset table, training and selection rules, teacher route |
| [docs/EXPERIMENT_STATUS_ZH.md](docs/EXPERIMENT_STATUS_ZH.md) (in Chinese) | Last confirmed results as of the 2026-10-08 snapshot |
| [docs/V31_IMPLEMENTATION_ZH.md](docs/V31_IMPLEMENTATION_ZH.md) (in Chinese) | V31 changes, functional acceptance, development-subset check, outstanding work |
| [docs/GEOMETRY_TEACHER_ZH.md](docs/GEOMETRY_TEACHER_ZH.md) (in Chinese) | Geometry-teacher implementation, failure record and diagnosis plan |
| [configs/v31_experiment_plan.json](configs/v31_experiment_plan.json) | Registered matched V30/V31 experiment (not started) |
| [evidence/v30_validation_summary.json](evidence/v30_validation_summary.json) | Redacted V30 validation evidence |
| [AGENTS.md](AGENTS.md) (in Chinese) | Project rules: copy-before-edit, no assets in Git, reporting requirements for scores |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (in Chinese) | Third-party sources and licences |
| [video_VACE/README.md](video_VACE/README.md), [video_VACE/UserGuide.md](video_VACE/UserGuide.md) | Upstream VACE documentation |

## License and third-party notices

[LICENSE.txt](LICENSE.txt) contains the Apache License 2.0 text and is byte-identical to the upstream VACE licence retained at [video_VACE/LICENSE.txt](video_VACE/LICENSE.txt). The repository contains no separate copyright or licence statement specific to the STaR-Body additions or to the inherited DiviBody code beyond these files and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which describes LICENSE.txt as the retained VACE licence. No file in the repository states whether this licence also covers files outside the upstream VACE code; please open an issue at https://github.com/appleweiping/STaR-Body/issues before reusing those files. The following external components are used but not distributed, each under its own terms:
- Wan2.1 source and weights;
- 4D-Humans/HMR 2.0 source, weights and body-model assets;
- SAM 3D Body;
- the MHR conversion assets, where the fitting code uses the official surface mapping but its independent optimiser is not the official parameter solver;
- DINOv3;
- SMPL/SMPL-X;
- 3DPW, AMASS, COCO and PoseTrack.

The project name and the combination of mechanisms do not imply any affiliation with the authors of the works listed below, nor do they imply that their reported results are reproduced.

## Citation

To cite this repository:

```bibtex
@software{yan2026starbody,
  author = {Yan, Weiping},
  title  = {{STaR-Body}: Spatial-Temporal Attentive Refinement for {3D} Human Body Reconstruction},
  year   = {2026},
  url    = {https://github.com/appleweiping/STaR-Body},
  note   = {Research code. V30 evaluated on 3DPW validation (single seed, provisional); V31 development version}
}
```

Please also cite the upstream projects and datasets used (VACE, Wan2.1, 4D-Humans/HMR 2.0, SMPL and 3DPW) as their maintainers request.

## Acknowledgements

This work builds on the following projects:
- the inherited DiviBody code base;
- VACE: All-in-One Video Creation and Editing, arXiv:2503.07598, https://github.com/ali-vilab/VACE;
- Wan2.1, https://github.com/Wan-Video/Wan2.1;
- 4D-Humans (HMR 2.0); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the official repository.

The geometry-teacher track uses:
- SAM 3D Body, https://github.com/facebookresearch/sam-3d-body;
- MHR, https://github.com/facebookresearch/MHR;
- DINOv3, https://github.com/facebookresearch/dinov3.

Two mechanisms are adapted from published designs:
- the spatial and temporal dual-stream modelling of MotionBERT (arXiv:2210.06551, https://arxiv.org/abs/2210.06551);
- the projection-based local feedback of PyMAF (arXiv:2103.16507, https://arxiv.org/abs/2103.16507).

Further methods consulted in the design are SAM 3D Body (https://arxiv.org/html/2602.15989v1), WHAM (https://wham.is.tue.mpg.de/index.html), GVHMR (https://arxiv.org/html/2409.06662v1) and FactorizedHMR (https://arxiv.org/html/2605.14854v1). STaR-Body is not an official reproduction of any of these works and does not inherit their reported results ([docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)).
