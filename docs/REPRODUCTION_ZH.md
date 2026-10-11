# 复现与资产准备

## 发布范围

仓库保留 230 个原 Python 文件的字节与 SHA256，并包含 V29/V30 新文件、VACE 源码和文本配置。`original_code_hashes.json` 可用于追溯原代码；`provenance/publication_source_hashes.json` 记录本次复制来源哈希。它们不表示所有历史入口都适用于当前模型。

原 V28/V29、新 V30 使用不同模型格式。V30 能从已审计的 V28/V29 初始化或严格加载自身 native checkpoint。不能随意用同名但格式不同的 checkpoint 代替。

## 环境

原实验主要使用 Linux、Python 3.10、PyTorch/CUDA 和真实 SMPL。`requirements-star-body.txt` 给出依赖入口，不是完整环境锁文件。外部源码应根据实际机器安装：

```text
STaR-Body/
  video_VACE/                 项目源码
  scripts/                   发布包装器与历史研究入口
  configs/                   示例配置
  data/                      本地缓存，不提交
  checkpoints/               本地初始化与资产，不提交
  dependencies/              外部源码，不提交
```

`WAN_SOURCE_ROOT` 指向含 `wan` 包的 Wan2.1 根目录；`HMR2_SOURCE_ROOT` 指向含 `hmr2` 包的 4D-Humans 根目录。部分旧缓存校验会引用原 source directory，需通过 `DIVIBODY_SOURCE_ROOT` 显式设置。

使用 `scripts/run_star_body.sh` 设置 PYTHONPATH，再调用 `scripts/train_from_config.py`。这是新增发布包装器，不更改模型、训练器或损失；尚未在新 checkout 上执行训练验证。

## 配置中的资产

| 配置项 | 必需内容 |
|---|---|
| train/val-vace-cache | 对应 split 的完整 VACE payload，真实 metadata、target_valid/frame_owner/velocity_owner/acceleration_owner 与训练/验证标签 |
| train/val-hmr-sidecar | 与 VACE UID/窗口严格一致的 HMR2 空间 token、rotmat/betas/camera 初值 |
| pose-*-vace-dir / pose-*-hmr-dir | PoseTrack 分片缓存；当前训练器仍初始化这些对象，即使辅助 pose 权重为 0 |
| init-checkpoint | 已审计 V28/V29 初始化；正式实验从原 V28 开始 |
| smpl-model-root | SMPL male/female/neutral 资产；不得把 SMPLH、MHR 或 SMPL-X 直接当作等价参数 |
| joint-regressor | 当前协议的 H36M joint regressor `.npy` |
| validation-manifest | 已审计的完整 validation 身份与覆盖说明 |

HMR2 SMPL 包装器还需要 `smpl-model-root` 父目录内的 `SMPL_to_J19.pkl`。数据和模型资产不包含在 Git 中；示例路径必须替换成真实本地资产。

## 训练与选模

示例配置来自真实注册的 V30 `spatial_motion` 第一臂：种子 2026、6,000 步、每 1,000 步完整验证，纯几何训练。运动损失按唯一帧 owner 划分 within/boundary 三元组，避免重复计数。

选模要求 validation 的 MPJPE、PA-MPJPE、PVE 都低于冻结原 V28，并且加速度误差不超过 22.329070147964064 mm/frame²。该上限属于本项目缓存协议；换数据或口径时不能机械沿用。

`disable-temporal` 关闭新增 V30 时间模块，冻结参考模型仍保留自身旧时序部分，所以该消融不是“全模型无时序”。训练中不根据 test 挑选模型；此前 test 已被项目历史模型观察过，正式报告需披露。

## 教师路线

`scripts/v30/prepare_geometry_teacher_v30_v1.py` 和拟合入口依赖另外准备的官方 SAM3D、DINOv3、MHR 及真实 RGB。当前脚本的完整控制器仍假定原研究目录布局；README 的通用训练包装器不负责配置教师环境。

64 帧教师完整预测复现已通过，400 步拟合复现未通过参数逐元素阈值，完整教师缓存流程停止等待诊断。不能用失败的控制器状态宣称完整缓存或蒸馏已经完成。详见 GEOMETRY_TEACHER_ZH.md。
