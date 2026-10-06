# 开源框架、权重和数据集核查（2026-10-06）

## 本项目实际采用：GeCo2

- 官方代码：https://github.com/jerpelhan/GECO2
- 论文：AAAI 2026，Generalized-Scale Object Counting with Gradual Query Aggregation。
- 官方公开示例权重：https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/main/weights/CNTQG_multitrain_ca44.pth
- 权重已下载到 models/CNTQG_multitrain_ca44.pth（350,785,678 字节）。来源、代码版本和 SHA-256 记录在 models/provenance.json。
- 少样本计数：在待识别照片上框选少量示例，提示模型寻找同类目标。示例不是本地训练数据，不需要先训练专用竹签 YOLO。
- 官方模型预测高分辨率中心响应图和框；官方演示还使用 SAM2 细化与框 NMS。它不是 P2PNet 式纯点回归模型。
- 本地默认使用原模型的中心峰值：关闭框 NMS，也关闭 SAM2 掩膜细化。界面另提供框 NMS 对照模式。内部仍有 3×3 局部极大值选峰，不能宣称任何相邻目标都不会被漏掉。
- 关闭掩膜细化是本地推理配置，与论文完整评测配置不同，不能照搬论文准确率。

## 其他真实可用方案

| 框架 | 官方代码 | 适用方式 | 本次状态 |
| --- | --- | --- | --- |
| P2PNet | https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet | 直接预测点和分数，一对一匹配训练 | 未安装；已有权重主要面向人群，不能当作现成竹签模型 |
| CountGD | https://github.com/niki-amini-naieni/CountGD | 文本、示例框或两者提示计数，公开计数权重 | 查证有权重；未在本机测试，其部署依赖更重 |
| GeCo2 | https://github.com/jerpelhan/GECO2 | 少量示例提示，预测中心响应和框 | 已安装、本地 GPU 跑通、接入页面 |
| FIDT | https://github.com/dk-liang/FIDTM | 学习定位图，从峰值定位个体 | 未安装；需要目标领域验证或训练 |

## 数据集

### FSC-147：通用点标注计数数据集

发布方：https://github.com/cvlab-stonybrook/LearningToCountEverything

6,135 张图片、147 个类别，提供每个目标的点标注和每张图中的少量示例框。按类别划分训练/验证/测试集。

本次已下载发布方的类别表、点标注和划分文件到 docs/。核对类别表后，相关类别有：

| 类别 | 图片数 | 划分 |
| --- | --- | --- |
| chopstick（筷子） | 6 | train |
| straws（吸管） | 7 | train |
| pencils（铅笔） | 58 | train |
| matches（火柴） | 8 | train |
| tree logs（木材端面） | 32 | test |

这些是相关外观类别，不等于大规模细竹签/棉签端面专用训练集；棉球也不是棉签。不能把训练集样例上的好结果当成独立泛化证据。需要专用模型时，点标注可用于 P2PNet 或中心定位网络训练。

### FSCD-147 / FSCD-LVIS

发布方：https://github.com/VinAIResearch/Counting-DETR

提供少样本计数和检测数据；FSCD-147 扩展 FSC-147 的框标注。发布方明确限定研究/教育用途，并限制再分发。没有下载这两个完整数据集。

### 竹子计数社区数据集

页面：https://universe.roboflow.com/object-counting-redi5/bamboo-counting-test/dataset/8

页面显示 88 张图片、目标检测任务、CC BY 4.0，支持 YOLO 等格式导出。该页面也有模型部署入口，但未确认存在可直接下载到本地的专用权重。图片究竟是竹竿还是细竹签、与用户场景是否匹配、标注质量如何，尚未逐图核查。没有下载或拿它训练本项目模型。

未确认可直接用于本场景、质量可靠的专用细竹签或棉签公开 checkpoint，因此本次采用有公开权重的通用计数框架进行实际验证。

## 对用户图片的验证

samples/user_bamboo.png：705×704。框选 3 个完整端面提示，未用这张图进行本地训练。无法排除该公共来源照片与开源模型预训练数据重叠，不能据此宣称独立测试准确率。

| 设置 | 输出数量 |
| --- | --- |
| 中心峰值，阈值 0.20，关闭框 NMS | 97 |
| 中心峰值，阈值 0.33，关闭框 NMS | 96 |
| 中心峰值，阈值 0.50，关闭框 NMS | 95 |
| 阈值 0.33，框 NMS IoU=0.50 | 96，删除 0 个候选 |

阈值是相对中心响应，不是校准过的概率。96 是预测值，尚未有独立人工真值标注。输出点、PNG、API 结果和模型检查报告均保存在 outputs/。通过了 GPU 加载、严格参数匹配、有限坐标、示例框校验、缓存筛选、NMS 对照和标注导出检查。

## Windows 适配

原始源码及其 MIT / Apache 许可证保留在 vendor/geco2。新增适配包括：复制作者仓库中的 Deformable-DETR ops；CUDA 扩展不可用时调用该仓库原有的 PyTorch grid_sample 实现；完整计数 checkpoint 加载时省略单独下载 SAM2 backbone；点模式不实例化未加载的 SAM 掩膜头。推理所用 backbone、中心响应和框头的全部权重均以 strict=True 加载，训练辅助头明确排除，没有随机初始化检测头冒充预训练模型。
