# C10 文献更新与最近机制核查（2026-11-04）

本报告是给后续 Astra 设计/查新审阅的证据输入，不是完整查新 PASS、SOTA 排名或实验结果。沿用 C10 当前 camera-space SMPL、3DPW cached-input 协议；没有启动训练、推断或修改论文/模型。依据固定版本 ARIS `f76ec8d6b77792e2242cca049e08d0cbb7d54e19` 的 Codex research-lit 与 novelty-check 收集资料，独立新颖性裁决尚待单独的新上下文 reviewer。

| 验收要求 | 本轮状态与证据 |
|---|---|
| 8–15篇与现方法相关文献，优先2024–2026 | 15个冻结条目；14篇读取原始摘要/正文，1篇 JFG-HMR 仅取得出版社索引内容 |
| 阅读原始内容，不据标题编造 | 每条的 read status、位置、短摘录见 ../provenance/aris_literature_sources_20261104.json；ReFit、PyMAF-X、TRAM、PromptHMR、SAM 3D Body 阅读了相关正文 |
| 覆盖最近6个月 | MoPO (2026-05)、FactorizedHMR v4 (2026-10)、DETRAM (2026-07) 已读摘要；非穷尽检索 |
| 分开 reported 与同输入复现 | 下文列协议门槛；本轮没有同输入 rerun，也未加入新数字到论文 |
| 可审计技能/来源/覆盖回执 | literature_read_receipt.json；13个 arXiv 条目的存在性由 verify_papers.py 机械检查，范围限元数据存在，不证明语义新颖性 |
| 完整 novelty-check 独立裁决 | 未完成；本报告仅输出邻近机制和待核查项，不输出 ABANDON 或新颖性分数 |

未发现项目 `papers/`、`literature/` 或 AGENTS 配置的 Paper Library；本地起点是既有 manuscript/baseline 元数据，并非重新扫描完整本地论文库。Zotero/Obsidian 工具未配置，未伪称使用。全部新增本地文件限定本工作目录。

## 最近机制：需要收紧的是主张边界

**ReFit 是最重要的已读近邻。** 它不仅“采用 recurrent fitting”：正文已经用右手对齐误差应影响右臂/肩而不影响左侧来说明解耦更新，并实现每个 keypoint 一个 feature channel、按身份顺序拼接反馈和每个参数组的 GRU。因而不能把“首次保留解剖身份”“首次关节专属 recurrent correction”归给 C10，也不能把所有旧方法说成平均掉 landmark identity。C00 的均值压缩是本项目特定接口的控制条件。C10 可核查差异应落在“把继承观测头的连续残差/投影/有效性按 schema 保留为17个 token，通过解剖距离偏置投递给24个 pose query，并在 part/time refinement 前注入”。这是否形成足够贡献，还需要独立裁决及分离输入/参数量/路由效应的实验。[ReFit 原文](https://yufu-wang.github.io/refit_humans/files/ICCV_2023_ReFit.pdf)

**第二个应补进检索与讨论的近邻是 JFG-HMR。** 出版社索引文字明确涉及 joint-image cross-attention、global/local fusion 与 ReFit feedback loop；直接页面返回403，未取得完整摘要/正文。它构成需要取得全文的具体风险，不构成“已经与 C10 相同”的证据。[出版社条目](https://www.sciencedirect.com/science/article/pii/S0097849325001803) 另有 PostoMETRO 已把2D pose token与image token交由joint/vertex queries读取；不能笼统宣称“以关节查询读取姿态证据”新颖。[PostoMETRO 摘要](https://arxiv.org/abs/2403.12473)

**空间对齐、视频融合和冻结视觉编码器都有直接先例。** PyMAF-X 有局部 mesh-aligned features 与全局 grid features 的 attention；STAF 结合 projection-based local alignment 与 temporal coherence；TRAM 的 VIMO 在冻结 HMR2 ViT 上分别加入 image-domain 和 motion-domain temporal transformer。这些可作为 C10 的公共基础与需对比的设计，而非三个组件存在就意味着三个独立新贡献。[PyMAF-X](https://arxiv.org/html/2207.06400v4)、[STAF](https://arxiv.org/abs/2401.01730)、[TRAM/VIMO](https://arxiv.org/html/2403.17346v2)

**当前检索没有证明 track prototype 的具体构造被既有工作完整包含，也没有证明其新颖。** 本轮 focused reading 足以识别最近 feedback/temporal neighbors，但未完成所有 shape-consistency、track aggregation 与 robust prototype 文献核查。不能把没找到等同于首次。FactorizedHMR 的最新v4强调稳定 torso/root 与不确定 limb completion，属于可能邻近的结构分解，不应据此恢复已暂停/未授权的 flow 或 detector/IK 路线。[FactorizedHMR v4](https://arxiv.org/abs/2605.14854)

## 冻结文献条目

“结果范围”仅指读到的论文所研究/报告的任务，不搬用其自称 SOTA 作为本项目结论。具体数值仍留在既有逐行文献证据，不在本轮重新转录。

| 文献（作者、年份、venue） | 实际读取与核心内容 | 与C10关系/结果范围 |
|---|---|---|
| [ReFit](https://yufu-wang.github.io/refit_humans/files/ICCV_2023_ReFit.pdf)，Wang & Daniilidis，ICCV2023 | 摘要、相关工作、方法；keypoint独立通道、投影取样、ordered concat、参数组GRU | 最接近 recurrent geometric feedback；single-image SMPL；不可当成均值反馈 |
| [PyMAF-X](https://arxiv.org/html/2207.06400v4)，Zhang等，TPAMI2023，v4更新2026 | 摘要与SAA方法；grid+mesh-aligned attention、part-specific full-body integration | 局部/全局alignment近邻；body/full-body设置应分开 |
| [WHAM](https://arxiv.org/abs/2312.07531)，Shin等，CVPR2024 | 摘要、官方overview；2D序列motion prior+video features、contact-aware trajectory | camera/world video基线；detector及motion预训练不同 |
| [TRAM/VIMO](https://arxiv.org/html/2403.17346v2)，Wang等，ECCV2024 | 摘要、related work、§3.4；冻结HMR2+两域temporal transformer | 最接近frozen-HMR视频增强；VIMO是该论文组件，不重复计数 |
| [PromptHMR](https://arxiv.org/html/2504.06397v2)，Wang等，CVPR2025 | 摘要、§3.4；full-image prompt，video decoder token temporal attention | SMPL-X、prompt/whole-frame条件不同；不可偷换为同输入 |
| [SAM 3D Body](https://arxiv.org/html/2602.15989v1)，Yang等，CVPR2026 | 摘要、related work、evaluation；MHR、keypoint/mask prompt、MHR→SMPL mapping | 强数据/表示参照；映射和不同backbone需声明 |
| [GVHMR](https://arxiv.org/abs/2409.06662)，Shen等，SIGGRAPH Asia2024 | 摘要；gravity-view坐标、camera rotation变换 | camera/world基线；未读最新TPAMI扩展正文 |
| [CameraHMR](https://arxiv.org/abs/2411.08128)，Patel & Black，3DV2025 | 摘要；HumanFoV、dense keypoints、perspective fitting及pGT改进 | camera和数据质量参照；不能仅按网络结构公平性解释数字 |
| [OnlineHMR](https://arxiv.org/abs/2603.17355)，Zhao等，CVPR2026 | 摘要、官方repo概述；causal KV、sliding window、incremental SLAM | 严格online条件不同于C10 offline；不搬用latency数字 |
| [PostoMETRO](https://arxiv.org/abs/2403.12473)，Yang等，WACV2025 | 原摘要；pose tokens+image tokens供joint/vertex查询 | 2D证据token化近邻；occlusion与标准benchmark |
| [MoPO](https://arxiv.org/abs/2605.09856)，Tang等，2026预印本，venue未核 | 原摘要；visibility、history pose completion、image fusion、IK | 遮挡与运动先验参照；不是当前C10已执行组件 |
| [DETRAM](https://arxiv.org/abs/2607.09089)，Lee等，2026预印本，venue未核 | 原摘要；persistent person queries、检测/跟踪/HMR统一 | identity指人/track，不能混同landmark identity；tracking结果与mesh结果分开 |
| [FactorizedHMR](https://arxiv.org/abs/2605.14854)，Kwon & Chen，arXiv称NeurIPS2026接受 | 最新v4摘要；torso-root确定性anchor+limb flow completion | 部位不确定性分解近邻；不据此称重复或恢复旧路线 |
| [STAF](https://arxiv.org/abs/2401.01730)，Yao等，TCSVT2024 | 原摘要、官方metadata；temporal attention+projection alignment+sequence pooling | 空间/时间alignment直接近邻；需要补充related work |
| [JFG-HMR](https://www.sciencedirect.com/science/article/pii/S0097849325001803)，Yao等，Computers & Graphics2025 | **仅出版社索引摘要片段/高亮，直接页面403** | joint-image cross-attention+ReFit loop可能很近；全文待获取，未完成阅读 |

## 比较协议必须分层

1. **Literature-reported**：保留原论文的split、训练数据、body model、backbone、输入框/提示、postprocessing及指标定义。当前20行文献test与本方法validation只提供背景参照，不是同协议胜负；本轮不删除更强或不利行，不追加未核实数值。
2. **Same-input checkpoint rerun**：应冻结原RGB帧、person identity、annotation/detector bbox、crop affine、帧cadence与覆盖集合。不能把只能消费自己特征的模型强塞入HMR/VACE缓存后称官方复现；允许各方法正常编码相同RGB，并把编码算力纳入效率。
3. **Representation/metric**：SMPL/SMPL-X/MHR映射及版本、14-joint regressor、root centering、PA scale/rotation/translation、PVE vertex correspondence、性别body model全部显式匹配。SAM3DB的MHR→SMPL转换并非零成本等价；指标名相同也不自动同口径。
4. **Temporal**：8-frame target/17-frame context与不同方法窗口、future access、padding、source stride需报告。C10 offline不能宣称满足OnlineHMR的causality；acceleration需统一annotation-frame²或seconds²及有效triplet/边界owner计数。
5. **Selection**：先锁validation选模规则、所有seed和hyperparameter，再运行独立评价。当前3DPW test曾被项目历史模型观察；不能重新命名为完全未见test。外部新数据若用于调参，须保留选择记录。
6. **机制因果**：ReFit-style通道身份、PostoMETRO-style pose evidence、C10 identity embedding、anatomical bias和新输入字段可能共同改变结果。至少需保留参数与输入完全一致、仅平均broadcast路由的完整对照；随后才判断“逐关节投递”的独立贡献。这里是需要Astra审定的实验建议，未自动扩预算或改模型。

## 下一项可执行交付

给Astra提供本目录三份证据及现有C00/C10原始结果，先裁定准确的一句delta、是否补入JFG-HMR全文核查，以及最小但完整的机制对照矩阵。独立审阅应读原文与C10代码路径，而非只读本报告结论。源代码检查重点是 `joint_feedback_v37.py`、`part_time_refiner_v36.py` 和 `human_decoder_kinematic_v36.py`。本轮不宣布完整查新通过；已发现的邻近工作也不足以作ABANDON判断。
