# G0权重与指标协议核验（2026-11-05）

此项服务于STaR-Body方法验证。只做CPU权重元数据检查与合成几何核验，没有训练、模型forward或真实benchmark成绩。

## 已完成

- 重新计算官方WHAM权重SHA，与既有回执一致。保持weights_only/CPU/mmap，仅对静态检出的标准NumPy scalar/dtype和trace要求的Float64DType作窄兼容。读取到epoch/model/performance/optimizer四个顶层字段、97个model tensor；未将performance作为本项目结果，未把含buffers的元素数当可训练参数量。首轮超时与默认受限读取失败均保留。
- 现有4项CPU解析测试通过。对固定[WHAM指标源码](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/eval/eval_utils.py)的三个纯数学函数作交叉检查：髋部对齐差0，FP32 PA最大坐标差约9.54e-7米，连续帧加速度原始差0。这仅支持所测合成输入的一致性。

## 必须保留的协议差异

WHAM加速度helper返回N-2项，其评测caller随后额外取[1:-1]；N=9时由7项变为5项。本项目工具还会排除轨迹边界与断帧。真实数据必须核对相同person/frame IDs与triplet选择，不能仅凭公式相同就合表。

WHAM米输入的加速度乘30²后为m/s²；旧报告为mm/annotation-frame²。在真实30fps且完全相同triplet下，后者乘0.9才与前者单位相同；不能把该常数用于未知帧率、断帧或不同裁剪的结果。此次保留两边原公式，不改旧实验，也不据合成测试认定真实协议已统一。

## 尚未通过

已查明确本地路径中没有独立SMPL及真实3DPW val；checkpoint内SMPL buffers不等于官方构造路径及资产资格已满足。真实数据schema、性别模型、关节/顶点对齐、覆盖率和官方/统一评测器真实输出仍待核对。服务器入口仍在认证前被拒绝，未取得新的目录、PID或训练进度。

机器证据见`evidence/g0_cpu_protocol_20261105.json`；完整失败/成功stdout与复现脚本只保存在E盘私有work。无GPU或TempoRAC操作，无定时任务。
