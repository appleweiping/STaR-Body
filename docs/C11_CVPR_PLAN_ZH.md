# C11：CVPR 2027 差距分析与实验计划

日期：2026-11-01。分支：`claude/c11-cvpr2027`。本文件是计划和判断，不是实验结果；所有数字都标明来源和 split。

## 0. 截止日期（CVPR 2027 官方 Call for Papers / Author Guidelines）

| 事项 | 时间（AoE） |
|---|---|
| 论文注册（OpenReview enrollment，标题+摘要） | **2026-11-10 23:59** |
| 论文提交 | **2026-11-16 23:59**，不延期 |
| 补充材料 | 2026-11-23 23:59 |

- 正文 8 页（含图表），参考文献和 **LLM Usage Statement**（必须有，否则 desk reject）不计页数。
- 官方只提供 **LaTeX** 模板（`author-kit-CVPR2027-v1-latex.zip`）。原 Word 稿件不能直接提交，已转为 `paper/latex/`。
- 双盲：不能链接本 GitHub 仓库，不能写作者信息；可写“代码将公开”。

## 1. 现状结论（基于逐文件代码审读，2026-11-01）

C10 = 在训练好的 A5 父模型（V30 双序解码器 + track-shape + GRU 精修器）上继续训练 1,000 步，把 17 个 landmark 的残差 token 通过带解剖距离偏置的注意力路由到 24 个 SMPL 关节。归档结果（单种子、3DPW **validation**、在同一 split 上选模）：

| 模型 | MPJPE | PA-MPJPE | PVE | Accel |
|---|---:|---:|---:|---:|
| 父模型 | 61.26 | 37.64 | 73.02 | 21.48 |
| 均值反馈对照（转录值） | 61.26 | 37.64 | 73.05 | 21.45 |
| C10 | 61.19 | 37.61 | 72.82 | 21.44 |

单位：mm，加速度 mm/frame²。来源：`evidence/c10_validation.json`、`evidence/c10_transcribed_controls.json`。

**按严重度排列的问题**（详细位置见各 reader 报告，已在 C11 代码中处理的标 ✅）：

1. **C10 的 “landmark 证据” 不是图像证据（高）。** V30 的 ObservationHead 以零权重初始化（坐标头为 0，log-variance 的 bias 为 −4，visibility 为 0）。之后所有阶段的 2D loss 权重都是 0，V34/V36 的 contract 还明确禁止 2D loss，所以这个头从未收到梯度。结果是 `observed_2d == 冻结参考模型的投影`、logvar 恒为 −4、visibility 恒为 0.5。C10 实际路由的是“当前估计相对冻结参考的投影漂移”，论文里 “routes predicted landmark evidence” 的叙述不成立。用随机权重全链 CPU 前向已复现；**需在服务器上用真实 checkpoint 确认**：运行 `scripts/inspect_checkpoint_v43.py`。
2. **对照不匹配（高）。** 均值反馈只有 512 个参数，输入是 4 维均值；joint 有 105,472 个参数，输入是 7 维 × 17。容量和输入都不同，增益无法归因到“身份保持”。✅ C11 加入了容量匹配的均值池化、WHAM 式有序拼接、无身份集合注意力、无偏置、错位解剖、无反馈六种对照。
3. **单种子、在同一 validation 上选模、没有 test（高）。** 效应为 MPJPE −0.07 mm；序列级 bootstrap 95% CI 为 [−0.21, +0.03]，包含 0。C10 第 1000 步的 student（未做 EMA）反而比父模型差，增益主要来自 EMA。✅ C11 改为 3 个种子、预注册固定步数、test 通过 validation lock 只评一次。
4. **Table 1 把我们的 validation 与他人的 test 混排（高）。** 项目历史上 V28 从 validation 到 test 退了 +5.5 mm（62.86 → 68.39）。C10 若有类似落差，会比 WHAM/GVHMR 差约 9–11 mm。
5. **指标口径与文献不一致（高）。**
   - MPJPE 以 H36M 第 0 关节为根（SPIN/CameraHMR 口径），WHAM/GVHMR 用 J14 髋中点。
   - PVE 未对齐，不对应任何近期协议。
   - 加速度单位是 mm/frame²，而 WHAM/GVHMR 用 m/s²（换算 ×0.9）。
   - ✅ v43 评测器同时输出 mpjpe/mpjpe_hip、pve/pve_pelvis/pve_hip、accel 的 mm/frame² 和 m/s²。
6. **时间平滑性差（中）。** 21.44 mm/frame² ≈ 19.3 m/s²，对比 WHAM 6.6、GVHMR 5.0 m/s²，属 VIBE 量级。论文不能宣称时间上的优势，要如实写成局限。
7. **协议依赖标注（中，需披露）。**
   - 裁剪框来自 3DPW poses2d（+25% margin，而 margin 本身是在 validation 上选出来的）；
   - track 用 GT person id；prototype 是离线非因果的整段统计；`frame_valid` 和 VACE 分段来自 campose_valid。
   - **test 集历史上被多次看过。** 用 test 评测过的版本有 geometry-v3、V10–V13、V15–V17、V26–V28、V29（三次）、V29Y，比文档里写的多得多。V29Y 的 validation 改善反而在 test 上退步。论文中必须披露这一点。
   - VACE 特征的贡献从未被消融。审稿人会问“为什么用视频生成模型做特征、代价是否值得”，而父模型已固化 VACE，无法在本轮补做，只能列为局限，并准备 rebuttal 方案。
8. **其他（低–中）。**
   - v39–v42 属于更早的 “parts” 支线，不是 C10 的后续，也没有任何结果记录。
   - C10 的 PoseTrack 头部映射不是官方映射（neck / 2·nose−neck）。
   - v36 里有约 15 个 CLI 参数被静默忽略，用它们做消融其实是在重跑 C10。
   - 旋转上界写松了：每步上界应为 (√3/2)·α，三步合计 5.2°/13.0°。
   - A5 父模型无法从仓库复现，缺少配置。
   - resume 会对所有 `.py` 文件做哈希，运行中改动或新增文件会导致 resume 失败。
   - 旧启动器 `scripts/run_star_body.sh` 把 `video_VACE/vace/models` 加进 PYTHONPATH，会遮蔽 Wan2.1 的 `wan` 包。✅ C11 改用 `scripts/run_star_body_v43.sh`。
   - v31/v32 的 HMR 构建脚本拒绝 test，旧的 v11 test 包装脚本不可续跑。✅ 新增 `cache_hmr2a_rgb_anchor_v43_test.py`。
   - 论文写 validation 有 1,357 个 window（stride 8），而 V31 管线默认 stride 4（2,578 个）。需要确认 C10 实际用的是哪一种；test 必须用相同的 stride（preflight 会检查）。
9. **新颖性风险（高）。** 关键点身份并不新：WHAM 用有序向量，ReFit 做逐关键点反馈，HybrIK 用 IK 从关键点解析旋转；树距离偏置是 Graphormer 的 spatial encoding；零初始化是 ReZero/ControlNet 的做法。均值池化对照本身是“稻草人”。✅ C11 把贡献定位为“在匹配对照下隔离接口因素”的分析加方法，并加入 WHAM 式有序向量对照（B7）；related work 已如实写出这些先例。

**判断**：按现有结果投稿基本会被拒。拒稿理由会是：效应在噪声内、只有 val 没有 test、对照有混淆、“证据”不是证据、口径与文献不一致。要达到 CVPR 质量，至少需要四样东西：真实证据、严格对照、test 数字、多种子统计。

## 2. C11 的科学问题与方法

**问题**：当 2D 证据是真实图像证据（ViTPose COCO-17 检测）时，保留每个关键点的身份并按解剖结构把它路由到对应关节，是否优于以下做法？

- (a) 置换不变的均值池化（DeepSets 式，容量匹配）；
- (b) WHAM/TRAM 式“按固定顺序拼接成一个向量”的全局条件（保留了位置身份，但不做路由）；
- (c) 无反馈、只训练同样步数的对照。

还要回答证据质量的影响：自参考漂移（C10）< 3DPW 自带 OpenPose < ViTPose。

**方法（v43，全部是新文件，原文件未改动）**：

- `video_VACE/models/evidence_router_v43.py`：
  - 每个检测关键点是一个 token：残差 u−p(x)、当前投影 p(x)、−2 log c、c、有效位；
  - token 加上关键点/来源身份嵌入；
  - 24 个关节 query 通过解剖距离偏置的 4 头注意力读取这些 token；
  - 输出投影零初始化；
  - 默认配置与 C10 逐位一致（有测试）。
- 消融开关：
  - `routing ∈ {anatomy, none, shuffled}`；
  - `identity ∈ {on, off}`；
  - `pooling ∈ {joint, mean, concat}`；
  - `sources ∈ {head, external, head+external}`；
  - 独立的 `init_seed`。
- 外部证据在每次精修迭代都按当前身体重新计算残差，在 part pooling 和时间交换之前进入关节特征。
- `scripts/cache_external_keypoints_v43.py`：
  - 在与 HMR2 **完全相同的 256×256 crop** 上运行 ViTPose（HuggingFace transformers 实现）；
  - 坐标与 C10 landmark 同一归一化（u = x/256 − 0.5）；
  - 也可直接读取 3DPW 自带的 OpenPose poses2d。

## 3. 实验矩阵（`configs/c11/matrix.example.json`）

所有 run 的设置相同：

- 从同一 A5 父模型出发，同一调度（6000 步 cosine，200 步 warmup，分阶段解冻）；
- 每个 run 1,000 步，取 EMA；种子 2026/2027/2028；
- 预注册的固定步数是 1000，不在 test 上做任何选择。

| ID | tier | 证据 | 汇聚方式 | 回答什么 |
|---|---|---|---|---|
| A0-none | ablation | 无 | — | 只多训练 1000 步的增益 |
| A1-mean | core | 头部漂移 | C00 均值（512 参数） | 复现 C10 的对照 |
| A2-head-joint | core | 头部漂移 | 路由（=C10；s2026 应复现 C10） | C10 是否可复现、是否显著 |
| B1-ext-pooled | core | ViTPose | 容量匹配均值池化 | 置换不变的全局证据 |
| B7-ext-concat | core | ViTPose | 有序拼接 → MLP（WHAM 式） | 位置身份但不路由 |
| **B4-ext-joint** | core | ViTPose | 身份 + 解剖路由 | **主方法** |
| B2-ext-set | ablation | ViTPose | 集合注意力，无身份、无偏置 | 身份是否必要 |
| B3-ext-nobias | ablation | ViTPose | 有身份，无解剖偏置 | 解剖偏置是否必要 |
| B5-ext-shuffled | ablation | ViTPose | 错位解剖偏置（固定错排） | 偏置是否必须正确 |
| B6-both-joint | ablation | 头部 + ViTPose | 路由 | 两种证据融合 |
| C4-op-joint / C1-op-pooled | optional | 3DPW OpenPose | 路由 / 池化 | 证据质量轴 |

主要比较（按种子配对；95% CI 合并种子方差（t，自由度 2）和序列 cluster bootstrap 方差，见 `scripts/analyze_results_c11.py`。3 个种子时区间刻意偏宽，以保证覆盖率）：

- B4 vs B1、B4 vs B7、B4 vs B2/B3/B5、B4 vs A2；
- B1 vs A1、A1 vs A0、A2 vs A0、B4 vs A0；
- B6 vs B4；C4 vs C1；B4 vs C4。

规模：core 5×3 = 15 个 run，ablation 5×3 = 15 个，optional 2×3 = 6 个。

## 4. 时间表（必须严格执行）

| 日期 | 工作 | 产出 |
|---|---|---|
| 11/1–11/2 | 拉分支；CPU 测试；`inspect_checkpoint_v43`（确认问题 1）；`preflight_c11`；生成 ViTPose 与 OpenPose 关键点缓存（train/val/test）；用 v43 评测器重评父模型/C10/C00 的 validation（应复现 61.1884 等）；smoke run | 资产与数值一致性确认 |
| 11/2–11/4 | core 层 15 个 run（记录单个 run 的 GPU 小时数） | core 的 validation 结果 |
| 11/3–11/6 | ablation 层 15 个 run；有余力再跑 optional | 消融结果 |
| 11/6–11/7 | 全部 validation 评测 → lock → test 评测（每个 checkpoint 只评一次）；如缺 test 缓存，在 11/2 前补齐（见 RUNBOOK §6） | test 数字 |
| 11/7–11/8 | `run_c11_matrix.py analyze`；`scripts/make_c11_figures.py`；按 §5 决策树改写论文中的 `\TODO` | 论文初稿（含真实数字） |
| **11/10** | 在 OpenReview 注册（标题+摘要，摘要可之后修改） | 注册完成 |
| 11/10–11/15 | 改写、内审、补充材料；页数检查 | 定稿 |
| **11/16** | 提交 | — |

GPU 预算：

- 先跑 1 个 run 量出时间 T（GPU 小时）。总量 ≈ 30·T / GPU 数，再加评测时间（每次评测约为前向 10k–35k 帧）。
- 如果 T > 2 h 且 GPU 少于 3 张：先保 core（15 run），ablation 只跑 2 个种子，并在论文中如实写明。

## 5. 结果驱动的论文决策树

记 Δ(X, Y) 为 X − Y 在 test 上的配对差，“显著”指 CI 不含 0 且 3 个种子同号。

1. **B4 显著优于 B1 且优于 B7**：主论点成立，即在真实证据下，身份保持 + 解剖路由优于两种全局条件方式。再用 B2/B3/B5 说明是身份和正确解剖共同起作用。题目、摘要、贡献按 router 写。
2. **B4 ≈ B7，但两者都优于 B1**：身份（位置）重要，路由不是必要条件。论点改为“在保留身份的前提下，路由的参数量更少、更可解释、可融合多源”，并且要量化参数量和速度。这种情况的投稿把握较低。
3. **外部证据 ≈ 无证据（B4 ≈ B1 ≈ A0）**：说明在冻结特征上做精修，额外的 2D 证据没有用。这不是 CVPR 正会论文，建议转投 workshop 或做分析类论文，或者把重点放在“自参考证据”诊断上。
4. 不论哪种情况：
   - Table 1 必须使用 test 数字，并按每篇文献各自的口径对齐（我们的数字同时报两种根节点和两种 PVE 对齐）；
   - 加速度要如实写为局限；
   - 要披露标注派生裁剪框、GT track id、离线 prototype、历史 test 曝光。

## 6. 风险与备选

- **ViTPose 权重下载**：服务器在国内时设 `HF_ENDPOINT=https://hf-mirror.com`，或者先在本地下载再拷贝目录，`--model` 参数可以指向本地目录。
- **ViTPose-H 在遮挡帧上不可靠**：`-2 log c` 和 c 会作为特征输入；可加 `external_min_confidence` 消融（例如 0.3）。
- **A2-s2026 未能复现 C10**：先检查 GPU 非确定性带来的差异是否只有约 0.01 mm 量级。差异大时停下来，排查父模型和配置差异（C10 的 stop-after 是 2000，但这不影响前 1000 步）。
- **EMDB/RICH 泛化**：现有管线是 3DPW 专用，11/16 前不现实。在论文中列为局限，并作为 rebuttal 阶段的补充计划。
- **只训练 1000 步可能不够**：主结论不变；如果有余力，对 B4/B1/B7 续训到 3000 步（resume 时允许修改 stop-after），作为补充材料中的预算曲线。

## 7. 代码与文档索引

- 运行手册：`docs/C11_RUNBOOK_ZH.md`
- 代码：`video_VACE/models/*_v43.py`、`video_VACE/divibody_training/*_v43.py`、`scripts/*_v43.py`、`scripts/*c11*.py`
- 测试（CPU）：`scripts/test_evidence_router_v43.py`、`scripts/test_c11_model_v43.py`
- 论文：`paper/latex/`（CVPR 2027 官方模板；数字来自 `paper/latex/generated/results.tex`，由分析脚本生成）
