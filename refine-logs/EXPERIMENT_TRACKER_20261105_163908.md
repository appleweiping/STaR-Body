# Experiment Tracker

状态仅表示实际执行，不以计划代替实验。未发现真实latest/合法新benchmark资产前，不启动GPU训练。历史C10/C00单独归档。

| Run ID | Gate | 目的 | 系统/数据 | 状态 | 下一项证据 |
|---|---|---|---|---|---|
| G0-001 | G0 | 性别资产和评测口径 | 官方SMPL/SMPL-X/MHR及J14/J24 | BLOCKED_ASSETS | 合法来源+identity+官方输出对齐 |
| G0-002 | G0 | 最强视频起点复评 | WHAM/TRAM/GVHMR/PromptHMR/OnlineHMR/Human3R | TODO | 官方commit/checkpoint+完整开发输出 |
| G0-003 | G0 | 单帧强起点及转换成本 | SAM3DB、CameraHMR、TokenHMR | TODO | 映射误差/资源profile |
| G0-004 | G0 | 冻结完整外部矩阵 | 15候选×3数据家族 | TODO | 每cell输入/训练数据/metric/coverage |
| G1-001 | G1 | 错误反馈observation | 固定train/val困难片段 | TODO | 分层前后误差和失败机制 |
| G2-A | G2 | 推荐主线4臂筛选 | 强视频起点+Full/3w/o | WAIT_G0_G1 | 每臂<=0.15B |
| G2-B | G2 | 最多一个竞争方向 | 强单帧起点+Full/3w/o | WAIT_G0_G1 | 映射通过后才运行 |
| G3-001 | G3 | 删除贡献 | 胜者4配置×3seed | WAIT_G2 | delta/CI/副指标保护 |
| G4-001 | G4 | 耦合机制 | 补齐8组合×3seed | WAIT_G3 | 预注册关键交互 |
| G5-001 | G5 | 完整外部test | 冻结最终模型+全部可比强baseline | WAIT_FREEZE | SOTA、coverage、效率 |
| LEGACY-RECOVERY | 历史 | 找回原latest与资产 | C10/C00 | BLOCKED_RUNTIME | 不自动启动C00，不重置旧状态 |
