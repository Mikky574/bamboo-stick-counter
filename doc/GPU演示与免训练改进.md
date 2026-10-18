# GPU演示与免训练改进（2026-10-18）

服务地址 http://127.0.0.1:5080/ 。本轮用户已授权使用GPU，启动时RTX4090空闲约22GiB。仍逐请求独立进程，结束后释放模型；网页服务保持运行。演示页新增整盒棉头、密集木签、密集竹签三个真实样本，并在主下拉隐藏已排除的手持图。CountGD++不读取点标注真值，最多自动选3个同图示例；GeCo2对照为前两张提供3个已选提示框，仅在显式切换GeCo2时载入。

## 本轮真实推理

| 图片 | CountGD++自动示例 | GeCo2三个人选同图示例 |
|---|---:|---:|
| cotton_overhead 整盒棉头 | 107 | 108 |
| flat_craft 密集整捆木签 | 99 | 99 |

CountGD++内部计时约9.9秒；GeCo2约2.2–2.5秒。这不是完整HTTP请求耗时或严格速度基准。所有结果原始JSON和编号图位于outputs/gpu_prompt_comparison/。用户页面保留棉头自动结果107个点，截图位于outputs/gpu_demo/cotton_page.png。

这是两张图的预测对照，不是准确率报告。既有标注有模型辅助和单轮视觉审查，不能将它们作为同模型的独立测试真值；同数量也不代表所有位置都正确。

## 示例是否越存越多

现有自动模式每个请求只从当前图自动选最多3个示例，下一张重新选。它们是本次推理输入，不更新模型权重，也不作为下次图像必须读取的示例库。导出结果可包含示例坐标用于追溯；这不同于持续学习。网页加载新图时清空临时示例。

## 下一步优先级

1. **固定少量分类参考模板与负示例。** CountGD++官方test_dataset.py提供use_ext_pos_exemp/use_ext_neg_exemp，可将一张参考图的少量实例跨图片使用。为平头、尖头和棉头分别选少量模板，必要时提供杆身/盒沿负示例，不必每张图片追加提示。本地worker尚未接入这部分，跨材质/光线的效果仍需测试。外部参考boxes用xywh，app/infer_sahi提示用xyxy，不能直接混用。
2. **限定计数区域，比较高分辨率分块。** 先排除盒沿、反射和杆身，再比较整图与重叠分块。官方infer_sahi.py可参考；本地worker目前仍是整图短边800、长边最多1333，未调用分块推理。分块必须映射回原图并去重，不能直接相加，且会增加耗时、边界漏检风险。
3. **保留已有两种模型作对照，先不追加训练。** 本轮GeCo2在木签图得到99点、棉头108点，不能认定比自动模式好；可用于检查候选差异。DAVE官方已推荐后续GeCo，GeCo又推荐GeCo2，当前没有证据表明回装DAVE会改善此场景。P2PNet训练继续暂缓。

官方依据：[CountGD++](https://github.com/niki-amini-naieni/CountGDPlusPlus)、[外部模板参数](https://github.com/niki-amini-naieni/CountGDPlusPlus/blob/main/test_dataset.py)、[分块实现](https://github.com/niki-amini-naieni/CountGDPlusPlus/blob/main/infer_sahi.py)、[GeCo2](https://github.com/jerpelhan/GECO2)、[DAVE](https://github.com/jerpelhan/DAVE)。
