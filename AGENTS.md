# 项目协作规则

- 修改模型或训练代码之前，先复制原文件，使用新的文件名修改；所有原文件代码保持不变。
- 尽量使用中文交流，代码除外。
- 不把数据集、模型权重、特征缓存、凭据和机器本地配置提交到 Git。
- 模型成绩必须明确 split、协议、单位、训练步数和证据来源；没有同协议独立评估时，不宣称达到 SOTA。

## ARIS 续研入口

- 本项目从现有 C10/V37r1 继续；先读 `docs/ARIS_CONTINUATION.md`、`docs/ARIS_ACCEPTANCE.md` 和 `.aris/research_contract.json`。
- ARIS 固定版本见 `.aris/aris.lock.json`，本地完整技能位于 `.agents/skills/`，不自动跟随 upstream。
- 常规执行用 Sol；重要方向、方法与实验设计用 Astra。同家族审查记录为 provisional，不伪称跨家族独立通过。
- 本机所有新增数据、缓存与中间产物只放 E 盘；后台进程无窗口。原模型和训练源文件不改写。
- STaR-Body 与 TempoRAC 均为服务器高优先级项目；本聊天只负责 STaR-Body。按精确项目路径、启动命令和运行记录确认归属，不能只按账号、Python 进程名或 GPU 编号识别。不得把 TempoRAC 当作普通低优先级任务抢占或停止，也不修改其代码、数据、队列及配置。每个实质阶段完成后同步 STaR-Body 已审查代码与去敏状态到 GitHub，并核对服务器版本。
