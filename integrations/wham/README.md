# WHAM 3DPW 开发评测适配

对应官方仓库 `yohanshin/WHAM`，固定commit `2b54f7797391c94876848b905ed875b154c4a295`。这是外部baseline的开发入口补丁，不是STaR-Body模型或最终test评测结果。

## 已实现

- 仅这个3DPW开发入口默认val，并拒绝test或其他数据集。
- 指定checkpoint或`dataset/parsed_data/3dpw_val_<backbone>.pth`不存在时，在运行main前停止，避免官方缺权重仅警告后随机初始化的路径。
- 要求绝对输出目录，输出`wham_3dpw_validation.json`，不覆盖已有同名JSON。
- 保留原forward、权重选择与metric代码；记录person-frame/accel样本数、flip、checkpoint路径、原始指标和单位。
- 官方GT首帧root用于世界轨迹初值；静态源码中不影响该camera-space指标，JSON保留这个限定披露。

**执行地点：以下评测与后续CPU/GPU核验只能在服务器执行，本地禁止运行。历史本地检查保留记录。**

## 执行前条件

补丁已通过11项stdlib/静态检查及Astra代码审查，只放行保存为开发适配。实际运行还需合法、不同性别的SMPL资产、H36M回归器及官方辅助资产；冻结3DPW validation序列/person/frame IDs，验证缓存schema、正常/flip内容和无test混入；固定权重SHA与变体；核验独立STaR-Body目录、环境、GPU资源及输出位置。旧C10的latest不是新baseline前提。

原官方预处理支持`python -m lib.data_utils.threedpw_eval_utils --split val`；它仍需真实原始数据/预测检测/特征和资产，当前未执行。不得将test缓存改名冒充val。

在独立官方checkout中先核对上述commit，再执行`git apply --check <本补丁>`、`git apply <本补丁>`。实际评测模板如下，变量由核验后的本项目运行记录提供；目前没有启动：

```bash
python -m lib.eval.evaluate_3dpw --cfg configs/yamls/demo.yaml \
  --eval-set 3dpw --eval-split val --metrics-output-dir "$RUN_DIR" \
  TRAIN.CHECKPOINT "$WHAM_CKPT" EXP_NAME g0_wham_val
```

使用全新的专属`RUN_DIR`，因为官方`prepare_output_dir`还会写配置文件；配置来源应为源码中的demo.yaml，避免SameFileError。模型权重选`wham_vit_w_3dpw`时显式覆盖demo的BEDLAM变体，不能混称。该开发分数不能作为最终benchmark/SOTA结果。

## 审查

调用请求模型：Astra xhigh，同家族暂定。结论：未发现阻止保存补丁的代码错误；不授权实际GPU执行。静态检查与运行验证分开记录，真实推断尚未运行。
