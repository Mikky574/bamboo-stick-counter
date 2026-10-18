# ConvNeXt-Small 场景分类 GPU 训练记录（2026-10-18）

已在本机 RTX 4090 完成训练。8类为平端、尖头、平铺长杆/筷子、棉头、生米、绿豆、硬币和其他。

## 数据与训练

100张真实照片，按来源组隔离：训练52张、验证20张、留出测试28张。同图SHA去重、dHash近似图与已知同拍摄系列合组，旧15张与同源图片仅在测试集。dHash有可能把不同图片保守合组，不据此认定同图。每类验证至少2个来源组，但测试每类仅1–11张，不能推断实际场景总体准确率。标签为单次人工视觉审阅，未使用模型预测反填；没有实例计数真值。

模型49,460,840参数，使用torchvision官方ImageNet预训练；18轮仅分类头，12轮微调最后阶段和分类头，可训练约1429万参数。224像素完整图补边，旋转/翻转/轻颜色增强，batch8、混合精度。最佳模型为第二阶段第1轮，完全由验证集选择；阈值和温度也仅用验证集选择，之后才评估留出图。30轮约79.65秒，PyTorch分配显存峰值0.419GiB、保留峰值0.506GiB（不含CUDA驱动上下文）。

## 同图对照

|指标|旧TinyCLIP|新ConvNeXt-Small|
|---|---:|---:|
|28张首选类别正确|24/28|27/28|
|接受路由|18/28|24/28|
|接受路由中正确|18/18|23/24|
|已知7类的接受覆盖|18/26|24/26|

新模型减少了拒绝识别，平端分类改善；同时增加了一个危险的误路由：rods_walmart_six.jpeg（完整平铺木杆）被接受为平端。米粒3张首选类别都正确，但仅1张超过接受阈值，不能宣传已解决米粒识别。验证集20张中首选18张正确，接受15张、其中14张正确。

因此本次已保存可加载训练模型和候选配置，但**没有替换网页默认TinyCLIP**，也没有声称计数准确率提升。要试验部署，可在释放当前场景worker后，将candidate_profile.json复制为上级active_profile.json；删除active_profile.json即回退。正式启用前需要更多来源的平铺杆/端面和小目标米粒照片，再用新的留出集评估。不要用本次测试集反复调阈值后继续宣称独立测试。

## 本地文件

- datasets/scene_classification_v1/manifest.json：固定数据划分、来源、许可备注和标签
- models/scene_classifier/convnext_small_v1/scene_convnext_small.pt：训练好的模型
- models/scene_classifier/convnext_small_v1/candidate_profile.json：候选配置，未激活
- outputs/scene_training_convnext_20261018/：训练历史、验证logits、测试逐图结果、新旧对比和效果图

运行：`.venv-countgdpp/Scripts/python.exe train_scene_classifier.py --manifest datasets/scene_classification_v1/manifest.json --output outputs/another_scene_run`。同一测试集反复用于设计决策后，应另采独立测试集。

预训练来源：https://download.pytorch.org/models/convnext_small-0c510722.pth；校验和及下载来源见模型目录pretrained_provenance.json。新Commons补图保留CC许可和作者。原有商品图的训练/再分发授权未独立核实，保留为本地研究实验素材，不作为可公开发布的数据包。
