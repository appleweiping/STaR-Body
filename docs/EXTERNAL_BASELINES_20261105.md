# 外部主比较规模与执行约束（2026-11-05）

本次根据用户新指令将主线改为外部顶会方法的标准benchmark。内部C00/C10仅作历史、组件和机制对照，不计外部baseline数量；不再创建或保留定时任务。科研executor保持Sol，重要方法设计由Astra负责。

## 实际主表审计

只计外部独立方法，排除ours。不同backbone、训练数据、flip或同方法重训为配置行，不新增方法；EMDB-1/2是一个数据集的不同协议，不算两个数据集。缺失单元格不计作已完成比较。

| 论文 | Venue | 主camera-space表/页 | 外部方法 | 配置行 | 表中数据集/协议 | 来源 |
|---|---|---|---:|---:|---|---|
| WHAM | CVPR 2024 | Table 1 / PDF p7 | 14 | 14 | 3DPW, RICH, EMDB-1 | [原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.pdf) |
| PromptHMR | CVPR 2025 | Table 1 / PDF p7 | 9 | 9 | 3DPW, EMDB, RICH | [原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_PromptHMR_Promptable_Human_Mesh_Recovery_CVPR_2025_paper.pdf) |
| OnlineHMR | CVPR 2026 | Table 1 / PDF p6 | 12 | 12 | 3DPW, EMDB-1 | [原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_OnlineHMR_Video-based_Online_World-Grounded_Human_Mesh_Recovery_CVPR_2026_paper.pdf) |
| TRAM | ECCV 2024 | Table 4 / PDF p12 | 13 | 14 | 3DPW, EMDB-1 | [原文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01796.pdf) |
| GVHMR | SIGGRAPH Asia 2024 | Table 2 / PDF p6 | 15 | 15 | 3DPW, RICH test, EMDB-1 | [原文](https://arxiv.org/pdf/2409.06662v1) |

这5篇的相近camera-space主表为9–15个外部方法、2–3个核心数据集家族。这是实际论文规模，不是会议硬性最低数。WHAM/GVHMR各数据集有空单元格；PromptHMR另评HI4D/CHI3D交互，HBW为文本消融，不能混成六个完整主比较数据集。OnlineHMR将online/offline分组，不能拿一个条件下领先当全表SOTA。

## 本项目明确采用的规模

- 核心矩阵：3DPW、RICH、EMDB三个benchmark家族，15个外部方法候选加最终新方法。候选名单见`configs/external_benchmark_registry.json`；实际复现数仍是0。
- 首先复现近期最强、合法可运行的公开checkpoint及最近机制近邻；完整矩阵保留经典锚点，但不通过堆弱基线达到数量。
- SAM 3D Body等异构表示保留为强参照；只有转换、关节/网格对应和输入条件通过核验后，才进入相应同口径指标列。不得因强方法难赢而删除。
- 3DPW为历史已观察数据；EMDB不同子集不重复计数据集。不宣称所有数据均为未见test。
- 本次用户将规模确定方式改为参照近年同任务顶会实践。旧“6数据集×7框架”要求保留在历史合同中，状态改为被本次明确范围依据取代，而非标为已完成；当前三数据集矩阵也只有计划，尚未跑完。

## 复现协议与数据选择

每行锁定论文版本、官方代码commit、checkpoint SHA、输入条件、表示、训练数据、测试增强、后处理、frame/person IDs及官方指标实现。官方发布权重优先复评，不把所有外部方法重新训练作为必需前置。原RGB和人员对应保持一致，允许各方法按原生流程编码；不得强塞C10缓存并称官方复现。

三项指标细节必须先通过：合法正确的SMPL/SMPL-X/MHR资产及映射；各dataset规定的J14/J24及root/PA/PVE对齐；实际时间戳与加速度单位。旧annotation-frame²和新方法m/s²不能直接混表。用同一预测交叉核对官方与统一评测器。当前历史性别资产同SHA问题未解决，旧数字继续只标为继承协议结果。

数据集按任务和合法数据可用性入选。开发仅用预先登记train/validation，EMDB若无合法development split则作为冻结后的外部评估，不拿test挑方向。验证阶段按预算判断明显无前景的组合，及时止损换候选数据/设计；写清退出原因、已耗算力及全部正负结果。test已开启后不通过换集、换seed、换指标抹掉失败。

## 科研路线与组件验收

主路线：选择G0中最强且可复现的公开视频起点，加入一个轻量、统一的反馈校正机制；竞争路线为SAM 3D Body强单帧起点加同一机制。最多两个实质候选，选胜者后才跑完整消融。不得为了容易显示提升而退回弱起点。

候选机制仅是假设：A可靠性估计→B同身份跨帧证据聚合→C几何约束更新。A产生同一可靠性权重，B产生聚合残差，C用它们和投影局部敏感度约束更新。三者必须围绕同一错误反馈传播问题。先在真实train/validation预测上证明现象与改进空间，再实现新版本；此处不声称新颖性或有效性已经通过。

每项真正w/o消融必须能独立删除目标组件并保持其余协议一致。删除后更好，判为有害；删除后经等效检验无实际差异，判为无作用，重新设计。区间宽/跨零不冒称有效或等效，只能按预注册seed上限补证据，仍不清楚则不通过。新设计最终须满足三个组件各有作用；不是保留一个无效模块凑数。

先定最小实际效应delta：以G0开发集外部起点误差的1%及数值重复误差为候选依据，在看组件结果前冻结；同时报告原始差值和配对seed。Full+三个w/o为第一组，最终2^3组合和预注册A-B/B-C交互检验验证耦合。新版本训练目标最多两个真实项；不把十多个loss重新分组包装为两项。

详细claim、预算gate和执行表见`refine-logs/EXPERIMENT_PLAN.md`及`EXPERIMENT_TRACKER.md`。历史C10原6000步文件与证据保留，恢复只作独立取证支线；它不再阻塞外部benchmark准备，也不自动启动C00。


## 已固定官方源码的候选名单

HMR2、ReFit、TokenHMR、WHAM、TRAM、GVHMR、PromptHMR、OnlineHMR、Human3R、CameraHMR、CLIFF、HybrIK、TCMR、GLoT、SAM 3D Body。15个官方仓库commit与README已核对，另检查27份相关源码/文档；权重SHA和实际评测仍待完成。CameraHMR为3DV2025，GVHMR为SIGGRAPH Asia2024，Human3R为ICLR2026，会议来源如实记录。

已确认的执行缺口：OnlineHMR文档中的train.py会进入训练，不能当eval-only运行；ReFit官方脚本硬编码3dpw_test，G0必须改为冻结开发集；PromptHMR当前核实入口只评图像模型，视频入口待查；CLIFF只有已核实demo入口；SAM 3D Body原生MHR的PVE映射未建立。注册表中所有命令都是源码证据文本，所有45候选格均未运行，不据此自动启动。


## 原文训练规模（不是本项目已投入算力）

| 方法 | 原文训练设置 | 执行含义 |
|---|---|---|
| WHAM | AMASS预训80 epochs，混合真实数据微调30 epochs；batch64、81帧 | 优先评官方权重，不重复整套训练 |
| TRAM | 100K iterations、batch24、16帧；冻结HMR2 ViT-H | 原文版与当前仓库更新版分开标记 |
| GVHMR | 论文500 epochs、batch256、120帧；2张RTX4090约13小时 | 作者网页420 epochs与论文不同，保留版本差异 |
| PromptHMR | 896²输入、batch96、约350K步；消融448²、150K步 | 有权重和评测入口，训练代码因许可未发布 |
| OnlineHMR | 约52K iterations、1张80GB H100 | 公开配置MAX_STEP250K与论文收敛步数不同，不能混称同一次训练 |

以上来自同一表格审计的论文方法/实验段与作者仓库，具体来源列在`evidence/peer_benchmark_scale_20261105.json`。新方法的完整训练量必须由数据量与实际吞吐另行确定；旧C10的6000步不是上述外部方法的公平训练规模估计。
