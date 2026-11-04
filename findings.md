# 研究发现与当前主张边界

## C10 checkpoint 3000：中期结果到主张审阅（2026-11-04）

范围为已完成的3000步student/EMA验证，不是原6000步配对实验完成。固定ARIS result-to-claim流程使用fresh原生Astra槽位（请求gpt-6-astra/ultra）；模型实际后端身份未另获遥测，保留reported_model=null。结论为 **claim_supported=partial、same-family/provisional**。高置信度仅针对本次证据边界判断；完整实验诚信审计尚未运行，不能称科研或投稿质量已通过。

| 可以记录的事实 | 仍不能提出的主张 |
|---|---|
| student和EMA均完成10,413帧、10,226三元组、12序列的3000步validation | 6000步训练已经完成、达到60/37.3目标或SOTA |
| 同一C10 student从2000继续到3000，四项全局误差均降低 | 3000步同预算胜过C00；C00当前仍为2000步 |
| 原选择器先接纳student；EMA未超过同轮更新后的best score | EMA因保护指标失败被拒、EMA普遍无效，或两种权重全部指标都改善 |
| progress与文件stat已记录3000保存阶段，原训练继续推进 | checkpoint payload或完整可恢复状态已独立核验 |

student3000的MPJPE/PA-MPJPE/PVE为60.90945183/37.60175182/72.72183195 mm，acceleration为21.25095700 mm/annotation-frame²。EMA3000为61.12309274/37.64490795/72.84600688及21.27345757。同权重EMA的PA从2000步37.62280025升至37.64490795；这条反例与所有序列回退均保留。EMA拒绝原因只有`target60_score_not_improved`，不能改写为保护指标失败。

当前工作主张收窄为：在冻结3DPW validation协议下，一条C10续训轨迹的student全局指标改善；尚无新增同预算方法对照、机制因果、多seed稳定性、显著性或跨域证据。接口同时改变输入字段、容量和投递形式，不能把整体差异全部归因于joint routing。

下一步继续已授权的C10→C00原6000步严格续训，保持数据、loss、selector和horizon，保留全部student/EMA与负结果；在完整配对结果之后再由设计审查确定必要的等输入/等容量路由对照、配对seed和组件删除消融。本次没有授权或启动新增预算，也没有修改最终论文。一次限时CPU checkpoint读取超时，独立payload身份核验仍开放；不因此重启健康训练或重跑已完成validation。

完整数值与逐序列证据见`evidence/c10_step3000_complete_20261104.json`和`docs/C10_STEP3000_COMPLETE_20261104.md`。私有完整请求/响应与身份元数据保存在`.aris/traces/result-to-claim/2026-11-04_run01/`，不提交Git。后续只有新证据出现才重新评估相应主张，不按定时器重复刷评审。
