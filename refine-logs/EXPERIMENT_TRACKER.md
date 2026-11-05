# Experiment Tracker

状态仅表示实际执行，不以计划代替实验。旧latest不是新路线前提；未冻结新路线合法资产、开发ID、评测协议和绝对预算前，不启动新训练。历史C10/C00单独归档。

| Run ID | Gate | 目的 | 系统/数据 | 状态 | 下一项证据 |
|---|---|---|---|---|---|
| G0-001 | G0 | 性别资产和评测口径 | 官方SMPL/SMPL-X/MHR及J14/J24 | BLOCKED_ASSETS | 合法来源+identity+官方输出对齐 |
| G0-002 | G0b | 最强视频起点复评 | WHAM/TRAM/GVHMR/PromptHMR/OnlineHMR/Human3R | WHAM_ADAPTER_REVIEWED_WEIGHT_DOWNLOADED | 527307587 B权重SHA已记；真实资产、开发ID及完整开发输出待核 |
| G0-003 | G0 | 单帧强起点及转换成本 | SAM3DB、CameraHMR、TokenHMR | TODO | 映射误差/资源profile |
| G0-004 | G0a | 冻结外部矩阵元数据（不跑test） | 15候选×3数据家族 | METADATA_CREATED_PENDING_ELIGIBILITY | 每方法每指标资格、输入/训练数据/coverage |
| G1-001 | G1 | 错误反馈observation | 固定train/val困难片段 | TODO | 分层前后误差和失败机制 |
| G2-A | G2 | 推荐主线4臂筛选 | 强视频起点+Full/3w/o | WAIT_G0_G1 | 每臂<=0.15B |
| G2-B | G2 | 最多一个竞争方向 | 强单帧起点+Full/3w/o | WAIT_G0_G1 | 映射通过后才运行 |
| G3-001 | G3 | 删除贡献 | 胜者4配置×5seed | WAIT_G2 | delta/CI/副指标保护 |
| G4-001 | G4 | 耦合机制 | 补齐8组合×5seed | WAIT_G3 | 预注册关键交互 |
| G5-001 | G5 | 完整外部test | 冻结最终模型+全部可比强baseline | WAIT_FREEZE | SOTA、coverage、效率 |
| LEGACY-RECOVERY | 历史 | 找回原latest与资产 | C10/C00 | BLOCKED_RUNTIME | 不自动启动C00，不重置旧状态 |

G2硬门：开发ID/seed、绝对B和B_eval、三指标保护线、checkpoint SHA及评测资格均须冻结；当前均未完成。原定时任务删除，不自动排队。
