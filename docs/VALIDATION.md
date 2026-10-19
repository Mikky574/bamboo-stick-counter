# 验证与结果口径

## 无照片公共检查

```powershell
.venv\Scripts\python.exe -m unittest verify_runtime_safety verify_worker_timeout verify_probe_selection verify_scene_classifier verify_geometry_budget verify_startup_residency -v
node verify_frontend.cjs
```

这些检查使用自造像素、模拟结果和临时真实子进程，不加载权重或使用GPU。已完成35项核心检查。Node仅用于前端行为测试，不是服务运行依赖；它是模拟DOM测试，不是浏览器截图验证。

可另用具备Torch和网页依赖的环境运行：

```powershell
.venv-countgdpp\Scripts\python.exe -m pip install -r requirements.txt
.venv-countgdpp\Scripts\python.exe -m unittest verify_model_startup verify_rice_prompt_fallback -v
.venv-countgdpp\Scripts\python.exe -m unittest verify_category_recovery verify_point_prompts verify_end_face_review verify_cpu_diagnostics -v
```

这两组只使用CPU张量及模拟模型，验证启动前后顺序、两种前向预热、至少三个不同种子、保持阈值及失败候选口径。Windows服务控制器还有独立临时目录集成测试，覆盖端口冲突、重复启动、预热失败／超时、启动中停止和无关哨兵存活。

新增四组验证有界类别恢复、点示例原像素缩放与真实视觉前向参数、端面确认的适用范围，以及CPU诊断子进程和超时清理。它们不读取开发照片、不加载权重；`verify_cpu_diagnostics`会创建自己的临时子进程和哨兵。页面脚本包含19项模拟交互检查，不能代替真实浏览器验收。

## 本地完整回归

`verify_smart`、`verify_runtime_rice`、`verify_application_cases`、`verify_rod_surfaces`、`verify_packaged_rods`、`verify_carded_rods`、`verify_rod_routing` 等还依赖开发期照片／输出，不作为不含该数据的公共克隆必过检查。未经核实重分发许可的商品图、训练图和目录提取图留本地；来源与收集代码保留。

人工真值与预测必须独立保存，计数代码不可读取 `truth`、文件名或样本ID决定数量。已有清晰开发图结果：分散米粒30、绿豆17、硬币8，规则礼盒10／10、近景可见端面4，橙色背板10；某些交叉、相贴、模糊和重压缩输入仍拒绝。

黄褐米原图343是候选数。独立先看原图在两块局部标17粒，全部匹配且未见同粒重复，但整图没有完整真值。同源变换输出345–348，边界、遮挡和检测仍有差异；局部核对不支持整图准确或100%精度结论。

2026-10-19两张端面开发照片的原图和JPEG92版本经真实接口验证为91／50，与独立可见端面参考逐个匹配。正面图的点选示例在GPU和CPU上均输出50；斜视图的手动示例仍可能漏检或误检，不能保证比自动路径更准。昆虫候选仍有漏检与翅膀误检，总数相同不算逐实例准确。详细范围见[点选示例](POINT_EXAMPLES.md)与[昆虫说明](INSECT_COUNTING.md)。

## 性能和发布验收

本机启动准备约8.1秒，第一张密集米粒1.24秒、再次0.93秒，棉头0.52秒、分散米粒0.02秒、盒装筷子0.76秒，两模型加载时间均0。特殊输入首次算子／分配仍可有额外开销。

发布验收检查模型恢复SHA-256、真实首张复用、不同类别结果、重复启动同实例、停止后端口及所属进程消失、其他项目存活。开发原始日志位于本地 `outputs/`，不入Git；不把分类准确率、模拟测试通过率或候选稳定性当整图计数精度。
