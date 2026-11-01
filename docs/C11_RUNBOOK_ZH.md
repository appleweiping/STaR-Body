# C11 运行手册（给在服务器上跑实验的合作者）

目标：在 **2026-11-07** 前拿到 C11 全部 validation/test 结果，生成论文数字；11/10 注册，11/16 提交。计划与理由见 `docs/C11_CVPR_PLAN_ZH.md`。

总原则：

- 所有新代码都是 `*_v43.py` / `*c11*.py` 新文件，原文件一行未改。
- 结果文件只新建、不覆盖。
- test 只能通过 validation lock 评测，且每个 checkpoint 只评一次。
- **实验运行期间不要 `git pull`，也不要改任何 `.py`。** 训练 resume 会校验所有 `.py` 的哈希，改了就无法续跑。建议为实验单独开一个 checkout：`git worktree add ../star-body-c11 origin/claude/c11-cvpr2027`（detached HEAD，即使当前已检出该分支也能用）。

---

## 0. 环境

```bash
git fetch origin claude/c11-cvpr2027
```

```bash
git worktree add ../star-body-c11 origin/claude/c11-cvpr2027 && cd ../star-body-c11
```

在原来跑 C10 的 Python 环境中（PyTorch + CUDA，`requirements-star-body-v31.txt`），只需额外安装 ViTPose 依赖：

```bash
python -m pip install "transformers>=4.49" accelerate matplotlib scipy
```

```bash
export HMR2_SOURCE_ROOT=/path/to/4D-Humans WAN_SOURCE_ROOT=/path/to/Wan2.1
```

国内服务器下载 HuggingFace 权重需要镜像。也可以先在其他机器上用 `huggingface-cli download usyd-community/vitpose-plus-huge` 下载，再把目录拷到服务器，并把 `--model` 指向本地目录：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**启动器**：C11 的命令都用 `scripts/run_star_body_v43.sh`。旧的 `scripts/run_star_body.sh` 会把 `video_VACE/vace/models` 加进 PYTHONPATH，遮蔽外部 Wan2.1 的 `wan` 包，从而导致循环导入。新的启动器与 C10 当时使用的 `run_target60_v34.sh` 相同。

## 1. CPU 自检（2 分钟，不需要数据）

```bash
bash scripts/run_star_body_v43.sh scripts/test_evidence_router_v43.py
```

```bash
bash scripts/run_star_body_v43.sh scripts/test_c11_model_v43.py
```

两者都应显示 `OK`。测试内容：默认 router 与 C10 的 `joint_feedback_v37` 逐位一致；所有变体在初始化时精确复现父模型；C10 checkpoint 能无损转为 v43；外部证据路径有梯度。

## 2. 配置

```bash
cp configs/c11/base.example.json configs/c11/base.local.json
```

```bash
cp configs/c11/matrix.example.json configs/c11/matrix.local.json
```

- `base.local.json`：把 `assets/...` 换成 C10 实际使用的文件，要求与 C10 run 的 `run_config.json` 完全一致。涉及的字段：train/val VACE、HMR sidecar、track prototypes、`init-checkpoint`（A5 父模型）、`parent-validation-report`、`initial-reference-report`、`validation-manifest`、SMPL、H36M regressor。
- `matrix.local.json`：
  - `base_config` 改为 `configs/c11/base.local.json`；
  - `keypoints` 填入要生成的缓存路径；
  - `references` 填入父模型、C10（步 1000 EMA，即 C10 的 `best.pt`）和 C00 均值对照的 checkpoint；
  - `eval.validation` 与 `eval.test` 填入评测资产（test 的资产见 §6）。
- `*.local.json` 已被 git 忽略，不会被提交。

## 3. Day-0 审计（10 分钟，CPU）

**3a. 观测头是否从未训练。** 这决定论文叙述（计划 §1 第 1 条）：

```bash
bash scripts/run_star_body_v43.sh scripts/inspect_checkpoint_v43.py --checkpoint <A5父模型.pt> <C10 best.pt> <C00 best.pt> --output results/c11/checkpoint_audit.json
```

如果 verdict 是 `UNTRAINED`，说明 C10 的“landmark 证据”是参考漂移，论文按此叙述。同时请把 C00 实际使用的 seed 和 A5 的 adaptation_config 发给论文作者。

**3b. 资产一致性：**

```bash
bash scripts/run_star_body_v43.sh scripts/preflight_c11.py --matrix configs/c11/matrix.local.json --output results/c11/preflight.json --splits train validation
```

必须通过的检查：

- 每个 window 都有 `target_source_frame_ids`，否则时间模块完全不起作用；
- validation 为 10413/10226；
- train 为 22735 帧。

keypoints 显示 missing 是正常的，第 4 步才生成。

## 4. 生成外部关键点缓存（GPU，约 1 小时内）

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json keypoints --data-root /path/to/3DPW --source vitpose --gpus 0 --splits train validation test
```

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json keypoints --data-root /path/to/3DPW --source openpose3dpw --gpus 0 --splits train validation test
```

- 输入是 3DPW 根目录，下面有 `imageFiles/` 和 `sequenceFiles/`。
- ViTPose 运行在 HMR2 完全相同的 256×256 crop 上，坐标直接与 C10 landmark 对齐。
- 完成后输出里的 `mean_confidence` 通常在 0.6–0.8；如果远低于这个范围，说明 crop 或权重有问题，请停下来检查。
- test 的 sidecar 若还没有，先完成 §6，再单独跑 `--splits test`。

再次运行 preflight，这时 keypoints 应显示 `covers_sidecar: true`。

## 5. 复现检查与 smoke run（约 1–2 GPU 小时）

**5a. 用 v43 评测器重评归档 checkpoint（validation）：**

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json evaluate --split validation --references-only --gpus 0
```

`results/c11/validation/R-c10-archived_s0.json` 里的 `metrics.mpjpe_mm` 必须等于 **61.1884**（PA 37.6134，PVE 72.8197，accel 21.4425），误差在 0.001 mm 以内。父模型应为 61.2628 / 37.6422 / 73.0249 / 21.4816。v43 评测器另外输出 `mpjpe_hip_mm`、`pve_pelvis_mm`、`pve_hip_mm`、`acceleration_error_m_per_s2` 以及逐帧 `.npz`。

**5b. smoke run**（每个 core 变体 2 步，16 个 validation window，结果是 development-only，只用来检查管线）：

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json train --smoke --tier core --seeds 2026 --gpus 0,1,2
```

日志在 `runs/c11/_logs/`。全部正常退出后，删掉 `runs/c11_smoke/`。

## 6. 若没有 test 缓存：按 C10 同一协议构建（建议 11/2 前完成）

test 需要以下文件，与 validation 同一格式、同一协议：

| 文件 | 命令（参数用 `--help` 查看） |
|---|---|
| test 覆盖清单（35515/34752） | `bash scripts/run_star_body_v43.sh scripts/prepare_target60_reference_v34.py --archive <3DPW.zip> --split test --output assets/test_manifest.json` |
| feature-only VACE 缓存 | 先查 validation 缓存的 stride 和编码参数：`python -c "import torch;s=torch.load('<validation_vace.pt>',map_location='cpu',weights_only=False,mmap=True)['signature'];print(s.get('format'),s.get('target_stride'),{k:v for k,v in s.items() if k!='dataset_preflight'})"`。然后用 v31 可续跑的构建脚本，参数**与 validation 完全相同**（尤其是 `--target-stride`），加上 `--split test --feature-only`：`bash scripts/run_star_body_v43.sh video_VACE/divibody_training/cache_continuous_3dpw_vace_features_v31_resume.py --data-root /path/to/3DPW --split test --feature-only --target-stride <同 validation> --vace-checkpoint-dir <Wan2.1-VACE-1.3B> --output assets/cache/test_vace.pt`。中断后加 `--resume` 继续；`--plan-only` 可以只打印 window 数。评测器只接受 `...-v5`（stride 8）或 `...-v6-stride4` 格式。注意：论文写的是 validation 1,357 个 window（对应 stride 8），而 V31 管线默认 stride 4（2,578 个 window）。请以 C10 实际使用的缓存为准，并把结果回报给论文作者 |
| HMR2a sidecar | v31/v32 构建脚本拒绝 test，请使用新增的 test 版本（v32 的版本化拷贝，可续跑，含 chumpy 兼容补丁），参数与 v32 一致：`bash scripts/run_star_body_v43.sh video_VACE/divibody_training/cache_hmr2a_rgb_anchor_v43_test.py --vace-cache assets/cache/test_vace.pt --data-root /path/to/3DPW --hmr-source-root $HMR2_SOURCE_ROOT --hmr2a-checkpoint <HMR2.0a ckpt，sha 2dcf7963…> --hmr-model-config ... --hmr-data-root ... --output assets/cache/test_hmr.pt`（其余必填参数用 `--help` 查看）。约 16 GiB；可用 `--max-new-frames N` 分段，再用 `--resume` 继续 |
| GT 缓存（新脚本，只读 3DPW pickle） | `bash scripts/run_star_body_v43.sh scripts/make_gt_cache_v43.py --data-root /path/to/3DPW --split test --output assets/cache/test_gt.pt --hmr-sidecar assets/cache/test_hmr.pt` |
| test track prototypes | `bash scripts/run_star_body_v43.sh scripts/cache_target60_predictions_v34.py --checkpoint <V30 祖父 checkpoint，其 sha 必须等于 C10 的 prototype_source_v30_sha256> --vace-cache ... --hmr-sidecar ... --smpl-model-root ... --split test --coverage-manifest assets/test_manifest.json --output assets/test_track_prototypes.pt` |

上表中的脚本都会导入 3DPW 数据集类，而这些类会连带导入 Wan/VACE 代码，所以即使是纯 CPU 的 GT 和关键点脚本，也需要先设置 `WAN_SOURCE_ROOT`。

如果服务器上已有 V28/V29 时代的 test 缓存，先用 preflight 检查格式（它会比对 validation 与 test 的 stride）：

```bash
bash scripts/run_star_body_v43.sh scripts/preflight_c11.py --matrix configs/c11/matrix.local.json --output results/c11/preflight_test.json --splits validation test
```

格式不符就按上表重建，不要混用不同协议。

## 7. 正式训练矩阵

每个 run 训练 1000 步（stop-after = 1000），产出 `runs/c11/<变体>/s<seed>/step_001000.pt`（含 student 和 EMA）。先跑 core：

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json train --tier core --gpus 0,1,2
```

再跑 ablation，有余力再跑 optional：

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json train --tier ablation --gpus 0,1,2
```

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json train --tier optional --gpus 0,1,2
```

查看进度：

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json status --tier core ablation optional
```

- **在 tmux/screen 里运行 runner**（或者 `nohup ... > runs/c11/runner.log 2>&1 &`），这样 SSH 断开不会杀掉任务。
- 每个 run 每 100 步写一次可续跑的 `latest.pt`。中断后重跑同一命令即可：有 `latest.pt` 的 run 会自动 resume，最多损失 100 步；没有 `latest.pt` 的 run 会被移到 `*.incomplete_<时间>` 后从头重跑。
- 正在运行的 run 会留下 `runs/c11/<变体>/s<seed>.running` 锁文件，`status` 显示为 `running`，再次执行 `train` 时会被跳过，不会被移走。如果确认进程已经不在（用 `nvidia-smi`/`ps` 检查），锁会被自动判定为失效。
- 请记录第一个 run 的耗时，写进 `results/c11/notes.md`，用于预算。
- **A2-head-joint s2026 应复现 C10。** 它的配置与 C10 完全一致，第 1000 步 EMA 的 validation MPJPE 应约为 61.19，允许 GPU 非确定性带来的约 0.01–0.05 mm 差异。差异更大时先停下来排查。

## 8. 评测、锁定、test

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json evaluate --split validation --gpus 0,1,2
```

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json lock
```

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json evaluate --split test --gpus 0,1,2
```

- `lock` 会为每个 checkpoint 写 `results/c11/locks/*.json`。锁绑定 checkpoint 的 SHA、EMA 权重、步数、配置以及完整的 validation 结果。test 评测没有锁会被拒绝。
- **不要因为 test 数字去改步数、种子或变体。** 所有变体统一使用第 1000 步 EMA（预注册规则）。

## 9. 统计、图、论文

```bash
python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json analyze
```

```bash
python scripts/make_c11_figures.py --matrix configs/c11/matrix.local.json
```

```bash
cd paper/latex && bash build.sh
```

`analyze` 生成三个文件：

- `evidence/c11/summary.json` 和 `evidence/c11/summary.md`：可提交；均值 ± std；配对差的 95% CI 合并了种子方差（t 分布，自由度 S−1）和测试序列抽样方差（按序列做 cluster bootstrap）；序列胜负数；`*` 标记可靠差异（CI 不含 0 且所有种子同号）；
- `paper/latex/generated/results.tex`：论文中所有 `\res{...}` 的数字来源。

重新编译后，论文里的红色 TBD 会变成真实数字。随后按 `docs/C11_CVPR_PLAN_ZH.md` §5 的决策树，改写论文中所有红色 `\TODO{...}`。

## 10. 需要回传给论文作者的文件（都很小，可以提交到分支）

- `results/c11/checkpoint_audit.json` 和 `results/c11/preflight*.json`。`results/` 被 git 忽略，请复制到 `evidence/c11/` 再提交。
- `evidence/c11/summary.json` 和 `evidence/c11/summary.md`。
- 每个 run 的 `run_config.json`、`metrics.jsonl`，以及 `results/c11/{validation,test}/*.json`。评测 JSON 很小，复制到 `evidence/c11/raw/`。逐帧 `.npz` 只在服务器上保留。
- 单个 run 的 GPU 型号、耗时和显存。

不要提交数据、权重、缓存，也不要提交 `*.local.json`。
