# C10 创新性复核状态

2026-11-04 的 fresh Astra 审阅结论为 **PROCEED WITH CAUTION，5/10，same-family/provisional**。不是完整新颖性通过，也不是否定当前续训的依据。该分数只描述本次有限范围审阅。

需要在最终稿中处理的直接近邻：

- [MotionBERT（ICCV 2023）](https://arxiv.org/abs/2210.06551) 已给出空间→时间、时间→空间双支及 softmax 融合。当前双顺序模块应补充归属与引用，不能作为独立原创原理。
- [ReFit（ICCV 2023）](https://arxiv.org/abs/2308.11184) 已有身份相关的局部反馈和分参数 GRU；[PostoMETRO（WACV 2025）](https://arxiv.org/abs/2403.12473) 已有 pose token 与 joint/vertex query。C10 不能泛称首次身份保持或首次关节查询反馈。
- [JFG-HMR（2025）](https://www.sciencedirect.com/science/article/pii/S0097849325001803) 的出版社索引摘要和部分正文包含 joint-image cross-attention、局部投影特征与 GRU 更新。直接全文页面仍受限，核心方法公式及代码未取得，不能声称已排除重复。

当前可供检验的具体差异是：继承观测头产生的 17 个具名连续 residual/projection token，经解剖距离偏置投递给 24 个 pose query，再进入既有 part/time refinement。是否构成有效贡献，需要在原配对完成后进行输入、参数不变的广播路由对照和去解剖偏置对照，并补足配对 seed。当前 C00/C10 同时改变输入字段、参数量与投递方式，不能将整体增益全部归因于身份路由。

原始全文审阅记录保留在本地 `runs/session_20261104/work/astra_novelty_review.md`。现有论文文件尚未修改；最终写作时再连同新结果、引用和全部页面排版一并复核。
