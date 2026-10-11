# 第三方来源与许可

本项目保留来自原 DiviBody/VACE 项目的代码及其原文件，新增模型使用独立版本文件名。

- **VACE**：[ali-vilab/VACE](https://github.com/ali-vilab/VACE)。原说明保留于 `video_VACE/README.md`，Apache-2.0 许可保留于 `LICENSE.txt` 和 `video_VACE/LICENSE.txt`。
- **Wan2.1**：[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)。作为外部依赖，不包含其源码 checkout 或权重。
- **HMR2 / 4D-Humans**：[shubham-goel/4D-Humans](https://github.com/shubham-goel/4D-Humans)。作为外部依赖；源码、权重和人体模型资产分别遵循其许可。
- **SAM 3D Body**：[facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)。用于独立教师准备；本仓库不包含其权重或源码 checkout。
- **MHR 转换资产**：[facebookresearch/MHR](https://github.com/facebookresearch/MHR)。拟合代码使用官方表面映射，但独立优化器不是官方参数求解器；转换资产不随仓库分发。
- **DINOv3**：[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)。由外部教师环境调用，未分发其权重。
- **SMPL/SMPL-X、3DPW、AMASS、COCO、PoseTrack**：数据与模型资产需要单独获取并遵守各自条件，未随代码分发。

方法借鉴与原论文链接列于 `docs/DESIGN_ZH.md`。本项目名称及组合结构不表示与上述作者存在合作关系，也不表示本代码复现其论文成绩。
