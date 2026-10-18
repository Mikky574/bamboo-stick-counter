# 第二批真实图片与 V2 GPU 训练（2026-10-18）

本批实际新增46张经过目视筛选的实拍照片，没有生成合成照片。来自Wikimedia Commons，保留CC BY、CC BY-SA、CC0或公有领域许可、作者、原始网页、原图URL、下载图SHA256、逐图判断与排除记录。

|新增类别|照片数|
|---|---:|
|平铺竹签、木杆、筷子|13|
|平端|4|
|密集尖端|1|
|生米|15|
|绿豆|5|
|盒装棉头|4|
|其他对照：回形针、夹子、熟饭配菜|4|

原100张排除3张（手持竹签、手指生米极近特写、筷勺混合礼包），合并46张后143张。训练88、验证23、留出32。原28张留出仍隔离，只作旧例回归；新增4张留出仅来自2个作者族，不能当成4个独立来源。每图仅有场景标签，本次没有实例点位、检测框或人工数量真值。与原项目的709个端面/棉头点位数据集是不同用途。

修复了v1按dHash相似度误把绿豆与木杆/彩筷合组的问题；它们仍保留旧测试身份，不回流训练。新图作者与原数据同源时继承旧split。SHA精确去重、来源组隔离和6项数据准备契约检查通过；跨划分近似哈希候选为0。清单冻结SHA256：25ccb31f0619b21c5a64c5e03c4ea61d056723d0637b09c2cc09de6b81b60e8e。

## 实际训练与对比

相同约4950万参数ConvNeXt-Small，重新从官方ImageNet预训练出发，沿用18轮分类头、12轮最后阶段微调；未依据测试图改参数或接受阈值。RTX 4090约130秒，PyTorch显存峰值allocated0.419GiB/reserved0.506GiB（不含CUDA驱动上下文）。最佳checkpoint为第二阶段第9轮，验证23张首选22张正确，接受19张且19张正确。

|留出组|V1首选正确|V2首选正确|V2接受且正确|
|---|---:|---:|---:|
|旧28张回归|27/28|28/28|25/25|
|新增4张，2来源族|3/4|3/4|3/3|
|合计32张|30/32|31/32|28/28|

V2纠正了rods_walmart_six的端面误分类。新增的一张平端牙签束（rods_v2_43）仍首选为平铺，但置信不足，拒绝路由；另两张其他类和一张旧生米图也拒绝。28/28是这批小样本里接受路由的观测结果，不能解释成通用100%准确率。分类正确也不等于实例计数正确。

## 已接入本地网页

已通过项目现有/api/release-model接口释放旧worker后启用models/scene_classifier/active_profile.json，未停止网页进程或其他项目。真实HTTP只传image，没有类别、文件名或人工框，响应meta确认为project/ConvNeXt-Small-scene-v2。四张端到端检查：

- DAISO筷子原有10根参考图：仍返回10根；首次约3.98秒含分类worker加载。
- rods_walmart_six：分类已变为laid_rods；长杆分离器尚未分清贴合木杆，返回count:null，页面显示—；约0.38秒。
- rice_gray：CPU颗粒方法仍返回30，约0.04秒，本例无需分类模型。
- 新牙签平端束rods_v2_43：分类不确定，返回count:null；约0.49秒。

本次训练进程已退出，GPU显存释放。分类器仍在CPU上按需推理。

## 文件与缺口

- datasets/scene_classification_v2_20261018.zip：143张原图、固定划分、来源记录与审核说明，约41MiB。按项目相对路径解压；训练环境与模型权重另在项目中。
- datasets/scene_classification_v2/manifest.json：完整冻结清单。
- datasets/scene_classification_v2/new_photos_preview.jpg：本批真实照片预览。
- models/scene_classifier/convnext_small_v2/scene_convnext_small.pt：新权重，SHA256 2f953daafdbcac755fac1e2a26f5ca11b6394c4495020f92c99df54aa5a2201f。
- outputs/scene_training_convnext_v2_20261018/：历史、验证logits、逐图预测、三模型同图对比和HTTP响应。

密集尖头只新增1张，深色背景单层散开生米仍缺新来源照片。这些缺口应优先补真实拍摄；旋转、调色等增强仍按同一原图计，不虚增独立样本。
