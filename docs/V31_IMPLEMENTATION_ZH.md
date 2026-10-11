# STaR-Body V31 接手与真实验证报告

日期：2026-10-08（America/Chicago）。**本轮完成源码接手、环境复用、V31实现、真实功能验收和开发子集评估；完整训练、精度消融和独立test尚未完成。**

## 已交付与保留原则

主项目：`E:\STaR-Body`。服务器隔离副本：`/media/lenovo/data2/mjx/STaR-Body_V31_20261008`。没有覆盖原项目；接手时的全部452个文件再次SHA256核验不变，包括230个原Python文件。没有提交Git、数据、模型权重、特征缓存或凭据。

新增V31模型、runtime、optimizer、boundary loss、训练引擎、配置包装器与独立消融配置。训练改动均先复制旧文件，来源记录位于provenance。新增统一评估器、真实验证器、环境记录器和修正后的启动脚本；完整源码见本目录ZIP。

## 已核实环境与资产

本机只有CPU版Torch；服务器实际使用Python3.10.20、Torch2.5.1+cu124、NumPy1.26.4、RTX A6000。既有环境与runtime_packages只读复用，未升级共享环境；观察到的154项包版本和外部源码提交号在environment_001.json，**不是已从零验证的环境锁文件**。

找到真实V28 native checkpoint，其SHA256为 `4254e31c11d6c44f362fe26934cd04ed8f2d37f2b513ef9656a54ccfddb750bd`。有SMPL三种性别、H36M regressor、J19、HMR2/Wan官方源码及视觉权重。找到了完整train/validation原始RGB与标注，CPU实际枚举train22,735人帧、validation10,413人帧，缺失RGB为0。

未找回历史V30权重/完整训练日志，也未找到完整HMR/VACE缓存。当前真实缓存每个split只有2窗16人帧，明确development_only。V28 checkpoint内部validation为旧3162帧，不能当作文档10413帧完整结果；历史README数值未被本轮重新确认。

## V31改动与证据

1. **稳健6D旋转。** 原V30在近共线/共线浮点输入下可能产生非正交矩阵；V31用尺度归一和cross重建正交基。CPU合成测试中最大正交误差从约1降至4.77e-7，梯度有限。见model_math_v31.json。此为数值合法性修复，不等于重建精度提升。
2. **可消融结构。** 新增逐帧shape delta系数（默认1，候选0）、逐轮局部采样半径（默认原值，候选[2,1,0.5]）。默认shape/采样子函数与V30精确一致。禁用逐帧delta不代表总shape逐帧一致，因V28初值仍会变化。真实精度消融尚未执行。
3. **实际loss路由。** V30训练engine的几何loss被V29 boundary wrapper覆盖；V31复制整条调用链导向runtime_v31。默认保留V30损失，shape去重监督仅显式开启。这个旧路由目前数学等价，不能据此否定历史V30实验。
4. **纯几何资产依赖。** 禁用PoseTrack训练/验证时不再强制加载对应缓存。恢复配置增加模型/损失/optimizer源码、SMPL/regressor/初始化/环境/数据指纹；正式配置启用缓存全SHA。完整engine中断续训仍未实测。
5. **真实启动修复。** 新launcher移除会遮蔽外部Wan包的vace/models路径；补列chumpy依赖。两个首次检查失败（Wan导入环、动态模块假文件名）均保留日志，后续重跑通过。

native V31严格恢复已验证；**native V30直接迁移到V31暂不支持**，当前支持从同一V28初始化，避免静默转换实验身份。

## 真实8帧验收

使用train首个合法8帧窗口，seed2026、FP32、3轮refinement、4 CPU线程，物理GPU1映射为cuda:0，单进程CUDA分配器限额10GiB。

| 检查 | V30 | V31 |
|---|---|---|
| 初始rotmat/betas/camera与冻结V28最大差 | 均0 | 均0 |
| 真实SMPL前向、SO(3)、反向梯度 | 通过 | 通过 |
| 优化器覆盖与一次真实更新 | 通过 | 通过 |
| 冻结参考无梯度、状态不变 | 通过 | 通过 |
| 模型/optimizer落盘加载逐元素一致 | 通过 | 通过 |
| 重载预测精确一致、损坏checkpoint拒绝 | 通过 | 通过 |
| 峰值Torch allocated显存 | 534.4MiB | 535.3MiB |

证据为real_v30_003.json和real_v31_001.json。非零头只用于暴露内部梯度，已恢复后才保存checkpoint；诊断扰动没有成为训练初始化。一次step后的loss没有要求下降，V30实测由0.091068到0.097114，因此不能把此验收称为优化收益。这里的smoke checkpoint不是正式训练成果，也未进入Git或源码ZIP。

## 本轮真实指标：仅16帧开发子集

样本为validation的courtyard_basketball_01/person_0，16个owner人帧、14个跨窗口加速度三元组。V30/V31均从同一V28**零初始化、没有训练**。固定输入checkpoint，无选模；seed20261008。GT只进入指标计算，optimizer_steps=0。

| 模型 | MPJPE mm | PA-MPJPE mm | PVE mm | accel mm/frame² |
|---|---:|---:|---:|---:|
| V28 | 30.5439 | 14.4515 | 42.7899 | 7.8820 |
| V30零初始化 | 30.5439 | 14.4515 | 42.7899 | 7.8820 |
| V31零初始化 | 30.5439 | 14.4515 | 42.7899 | 7.8820 |

三者一致验证了初始化与评估一致性。**这不是完整3DPW成绩，不代表V30/V31优于V28，不可与README历史全量结果或论文SOTA作排名。**

指标保持原项目协议：H36M回归J14、pelvis MPJPE、逐帧Procrustes PA、未对齐PVE、预测neutral SMPL/GT gendered SMPL，按annotation UID拼接的hip-centered加速度。使用标注派生person crop，无FlipEval，无平滑后处理。annotation/source帧身份不同，不能将mm/frame²直接与论文m/s²比较。

| 模型 | 总参数 | 评估峰值allocated MiB | 16帧forward秒 |
|---|---:|---:|---:|
| v28 | 16,723,786 | 176.0 | 1.386 |
| v30 | 28,329,887 | 220.8 | 1.769 |
| v31 | 28,329,887 | 220.8 | 2.079 |

这是共享繁忙GPU、batch1、无warmup的功能运行测量，包含设备传输和SMPL反馈，不是稳定独占性能benchmark；不能据此判定V31相对V30精确时延差。每份JSON记录峰值reserved、完整配置与同步计时范围。

正式入口实际拒绝development缓存，exit1且没有写出正式结果；见formal_subset_rejection.log。没有改低10,413/10,226的validation覆盖门槛或35,515/34,752的test覆盖门槛。

## 尚未完成与下一步

完整缓存需重新编码train stride8的2,909窗和validation stride4的2,578窗，四缓存核心tensor约18GiB，应预留30–40GiB；完整命令、权重/源码路径、UID摘要和协议在《完整缓存重建计划》。历史缓存哈希未恢复，重建所得不能声称逐字节复现历史实验。

目前两GPU持续有其他任务，host RAM约26GiB可用且swap已满；本轮仅运行有限功能检查，没有启动持续全量编码/训练，也没有停止任何已有任务。旧缓存构建器不支持分片恢复，HMR旧脚本还会覆盖output并无条件标记非development；计划要求输出路径不存在并先验收完整VACE覆盖。

正式实验登记在v31_experiment_plan.json：同一V28、seed2026、每臂6000步、每1000步student/EMA验证；对照V30、V31稳健旋转、单项shape收缩、单项radius计划。数据哈希与新全量V28加速度门槛必须在开跑前锁定，不能照搬历史22.32907。多种子2027/2028预登记，test不能用于挑臂/种子/检查点。冻结方案后才进行test与未参与设计的外部泛化。

《相关方法与改进排序》核查了SAM 3D Body、MotionBERT、PyMAF、WHAM、GVHMR原始来源，并按收益假设、成本与风险排序。教师缓存与SMPL拟合复现没有新增成功证据，未启用蒸馏。

本轮结论只到“真实运行链路通过、数值问题修复”；**完整评估、消融贡献、独立泛化和SOTA差距仍待后续正式实验。**
