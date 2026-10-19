---
name: bamboo-counter
description: Maintain this repository's local image counter, including automatic counting diagnostics, model residency, Windows service start and stop, validation, and model release packaging. Use within bamboo-stick-counter when operating or extending the counter.
---

# 见数项目维护技能

在项目根目录读README，再按任务读相应docs。本技能仅针对此仓库，不配置其他项目GPU或系统服务。用户目标为上传图片后自动计数，不增加类别、模式、示例框必选步骤。

## 启动和停止

使用start.ps1/status.ps1/stop.ps1或cmd入口。以脚本返回状态和两个当前 `model_resident` 判定就绪；`startup.state` 只是当次启动历史。参数及故障处理读 `docs/OPERATIONS.md`。

启动预加载GPU CountGD++与CPU分类，空闲不卸载。只停止有身份和创建时间证明属于本实例的服务组；不要按Python名称、端口或显存数字推断归属。启动中停止、旧PID和端口冲突按控制器处理。若用户要求最终停止，验收后确认端口与所属模型进程退出。

## 慢请求与识别失败

读本地轮转请求／服务日志及模型状态，区分加载、分类、几何、深度推理和前端等待。`cache_reused=false` 与加载耗时说明发生重载；`no_initial_exemplars` 是未找到目标，不是GPU卡死。模块位置读 `docs/ARCHITECTURE.md`，不必加载所有历史记录。

用用户原像素和实际图片接口复现，不用样本ID或参考坐标决定结果。米粒默认零候选才允许黄褐米回退，至少三个去重种子、保持阈值，不扩大到其他类别或覆盖自定义提示。签类杆身和平端／尖端可任选受图像支持的方法；纸带上下同根不重复，不按成双规律乘二。

用户可选点选1–3个当前图片中的目标作为视觉示例，通过 `exemplar_points:[{x,y,diameter}]` 重算。该输入不等于离线人工真值，不能把点数直接加到结果；验证必须证明示例框按原像素缩放后进入模型，并绕过自动分类和自动示例。默认按钮仍仅发送图片。端面混入杆身的额外确认只适用于自动bamboo且示例紧凑的路径，均匀端面近邻抑制不能套到昆虫、尺寸混杂对象或整杆。操作及限制见 `docs/POINT_EXAMPLES.md`。

点选界面支持拖框移动、拖角或滑块单独调整每个示例、局部裁剪预览和图片缩放；这些动作不推理，收起编辑保留示例、待重数提醒及重数入口，只有提交重数才调用模型，结果仍保留候选口径。

系统自动路由可显式开启 `allow_category_recovery`：只在原类别初始或最终无候选时做一次支持目标的语义重选，同设备、同阈值，异常直接报错，自定义提示不改。统一 `insects` 目标不按物种扩展八分类器；这与米粒提示回退分开。昆虫候选须逐只核对，漏一只与多算一片翅膀不能抵消成正确。分割及点监督的后续方案读 `docs/INSECT_COUNTING.md`，未实测新模型不能声称已提高精度。

## 验证和口径

按 `docs/VALIDATION.md` 选择公开模拟／生成检查与本地照片回归。进程改动须验证复用、超时清理和无关哨兵存活；真实GPU验收记录第一张是否复用、加载耗时及不同类别图片，不把上一张答案缓存当预热。

人工参考先独立建立，再与预测匹配；局部17粒通过不证明整图343候选全对。保持候选标签、`count_is_total:false`、`review_required`和无结果“—”。完全遮挡库存不能由单张图推断，分类分数不是计数准确率。

## 显存与模型发布

文件摘要以 `models/release-manifest.json` 为准，恢复读 `docs/SETUP.md`，显存读 `docs/MODELS.md`。4GiB空闲是准入门槛，权重常驻不等于显存硬上限。

按 `docs/RELEASE.md` 构建校验包，当前必需CountGD++／BERT及活动ConvNeXt v2。保留原许可，不把训练环境、重复旧权重、日志或整批网络图混入发布；公开可访问不代表可重分发。用户已授权的发布任务内，完成测试和具体包内容后上传，核对提交、标签、体积和摘要。凭证只经既有安全机制读取，不写进源码、命令输出或日志。
