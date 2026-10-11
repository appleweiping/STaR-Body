# STaR-Body

**Spatial–Temporal attentive Refinement for 3D Human Body Reconstruction**  
时空注意力驱动的三维人体迭代重建模型。

当前发布版本为 **STaR-Body V30**：保留原 V28/V29 文件，在独立新文件中实现新的空间与时间解码结构。本仓库包含项目源码、训练入口、教师几何准备与实验说明；数据集、预训练权重、SMPL 资产及特征缓存需要自行准备。

## 模型结构

- 对真实 HMR2 **16×12 空间 token**执行关节查询跨注意力。
- 空间→时间、时间→空间两个注意力分支学习融合。
- 使用压缩后的 VACE 时空特征作为附加记忆。
- 每轮执行 SMPL，并从当前投影附近重新采样图像局部证据。
- 分开预测根朝向、身体旋转、形状与相机；当前窗口为 8 帧。
- 冻结 V28 提供初始化，新模块零初始化时保持原预测。

默认隐层 256、8 个 attention heads、4 个双流块、3 轮迭代。当前实现没有加入动作预训练、条件 flow 或端到端视觉骨干微调。SAM 教师几何准备是独立研究流程，尚未完成完整缓存或用于 V30 蒸馏。

核心文件：

| 用途 | 文件 |
|---|---|
| V30 模型 | [human_decoder_spatiotemporal_v30.py](video_VACE/models/human_decoder_spatiotemporal_v30.py) |
| 数据与前向 | [ram_parts_runtime_v30.py](video_VACE/divibody_training/ram_parts_runtime_v30.py) |
| 有效训练入口 | [train_spatiotemporal_v30_v3.py](video_VACE/divibody_training/train_spatiotemporal_v30_v3.py) |
| 步骤训练引擎 | [v30_training_engine.py](video_VACE/divibody_training/v30_training_engine.py) |
| 教师几何提取 | [prepare_geometry_teacher_v30_v1.py](scripts/v30/prepare_geometry_teacher_v30_v1.py) |
| 教师 SMPL 拟合 | [fit_geometry_teacher_v30_v1.py](scripts/v30/fit_geometry_teacher_v30_v1.py) |

## 当前实验状态

以下为 2026-10-07 最后确认的完整 **3DPW validation** 结果，共 10,413 个有效人帧、10,226 个加速度三元组；V30 是单种子、训练中的 EMA 检查点。

| 模型 | MPJPE ↓ mm | PA-MPJPE ↓ mm | PVE ↓ mm | 加速度误差 ↓ mm/frame² |
|---|---:|---:|---:|---:|
| 原 V28 | 62.8595 | 38.8244 | 77.4927 | 22.3291 |
| V30，2,000 步 EMA（已下载证据） | 62.0758 | 38.1532 | 74.3238 | 22.0177 |
| V30，3,000 步 EMA（SSH 日志观察） | 61.7578 | 37.9109 | 73.7713 | 21.8073 |

**尚未证明超过 SOTA。** 两臂各 6,000 步的正式实验在最后检查时尚未完成；V30 的完整 test 尚未评估。历史 V29Y 在 validation 上改善、完整 test 上退步，说明不能把 validation 增益当作泛化结论。详见 [实验状态](docs/EXPERIMENT_STATUS_ZH.md)。

## 安装与训练

主要执行环境为 Linux、Python 3.10、PyTorch/CUDA。安装项目依赖，并准备外部 Wan2.1、4D-Humans/HMR2 源码：

```bash
python -m venv env
source env/bin/activate
pip install -r requirements-star-body.txt
```

以上安装命令是发布说明，未在本次 GitHub 发布中重新执行。CUDA 与 `flash-attn` 的版本需要匹配实际机器。外部资产及目录说明见 [复现指南](docs/REPRODUCTION_ZH.md)。

复制配置后，填写真实缓存与权重路径：

```bash
cp configs/star_body_v30.example.json configs/star_body_v30.local.json
# 编辑 configs/star_body_v30.local.json 中的缓存、SMPL 与初始化路径
export WAN_SOURCE_ROOT=/path/to/Wan2.1
export HMR2_SOURCE_ROOT=/path/to/4D-Humans
export DIVIBODY_SOURCE_ROOT=/path/to/original/divibody_vace

CUDA_VISIBLE_DEVICES=0 bash scripts/run_star_body.sh \
  scripts/train_from_config.py --config configs/star_body_v30.local.json
```

`scripts/run_star_body.sh` 使用当前虚拟环境的 Python；也可以通过 `STAR_BODY_PYTHON` 指定解释器。训练直接调用已运行过的 V30 v3 入口；发布用包装器只负责路径和参数。**本次发布没有重新训练或运行模型测试。**

原服务器的控制器也以原文件保留在 `scripts/v30/`，依赖特定研究目录和历史计划，不能直接作为新机器的通用启动命令。

## 数据、权重与协议

- 训练与 validation 缓存必须包含匹配的 RGB/VACE/HMR 身份、有效帧、唯一 owner 与运动 mask。
- 需要原 V28 初始化 checkpoint、SMPL 模型、H36M joint regressor 和 `SMPL_to_J19.pkl`；训练器当前仍要求 PoseTrack 缓存路径，即使把辅助权重设为 0。
- 使用标注派生的人框，需在比较结果时披露；加速度单位为 **mm/frame²**。
- 3DPW/COCO/PoseTrack/AMASS、人体模型和第三方权重遵循各自授权条件，本仓库不包含这些资产。
- 教师拟合只接收预测网格及 HMR 初值；拟合平移与 crop camera 分开保存。MHR 和 SMPL 不能直接当作相同表示。

## 文档与来源

- [结构设计与原始研究来源](docs/DESIGN_ZH.md)
- [复现指南](docs/REPRODUCTION_ZH.md)
- [教师几何准备状态](docs/GEOMETRY_TEACHER_ZH.md)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)
- [原代码 SHA256](original_code_hashes.json)
- [发布代码来源 SHA256](provenance/publication_source_hashes.json)

本项目借鉴 MotionBERT 的空间/时间双流、PyMAF 的投影局部反馈等机制，但不是这些模型的官方复现，也不继承其论文成绩。原 VACE 说明与许可保留在 `video_VACE/`。
