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

## 运行状态更新（2026-11-05）

恢复连接后的只读检查发现原记录PID不存在、原source/experiment/controller路径不可用；当前不能执行或承诺连续训练。此前3000步数值与审阅结论仍作为历史证据保留，不能据此声称checkpoint当前可恢复。最近保存进度报告为C10=3000、C00=2000；真实latest及原绑定资产定位/核验完成前，下一步是恢复资产可用性，而非重新初始化或启动其他实验。服务器侧本次状态快照尚未同步；详细范围和限制见`evidence/runtime_unavailable_20261105.json`。本轮无新validation，未重复语义科研审查。

## 标准评测口径待验收（2026-11-05总体进度核对）

既有本地baseline审计确认：外部方法在C10共享validation上的完整同协议复现数为0；文献表的14种方法、20个3DPW test设置不能与C10 validation直接排名。历史实际male/female模型文件曾同SHA，C10 run_config又将male/female/neutral记为同SHA；当前服务器资产尚不可复验，标准gendered-GT等价未建立。保留原数值为继承协议内部结果，既不删除也不升级为SOTA结论。此处是读取既有证据，不是新实验；原严格续训合同不作中途修改。具体去敏字段与来源摘要见`evidence/protocol_status_20261105.json`。

## 外部主比较与新机制方向（2026-11-05用户更新）

研究主线为STaR-Body的问题与方法创新；近期强外部方法用于比较，内部C10/C00用于历史与消融证据；原定时任务已删除。五篇主表审计支持9–15外部方法、2–3核心数据家族的参考规模，当前计划15候选×3核心，完成复现仍0。Astra建议优先强可复现视频起点加可靠性→跨帧证据→几何约束轻量校正，SAM3DB为最多一个竞争方向。所有新颖性、每组件贡献和SOTA均待G0/G1及实际实验；不因计划完备声称通过。验证止损与数据切换必须记录，test之后不隐藏失败。

## JFG-HMR增量覆盖（2026-11-05）

取得出版社搜索索引预览，确认通用关节图像注意力、全局局部融合与ReFit式反馈已有先行描述；不能作为STaR首创。完整3.2/3.3及实验未读，全文查新仍开放。Astra增量审阅保留现有三组件候选，优先既定遮挡恢复observation与同输入容量控制；无新预测或性能结果。见`docs/JFG_HMR_INCREMENTAL_REVIEW_20261105.md`。
