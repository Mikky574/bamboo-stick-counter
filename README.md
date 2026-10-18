# 见数 · 本地图片自动计数

上传图片，点击一次识别，自动选择计数方法并在原图上标点。面向清晰的竹签端面／尖头、棉签头、平铺筷子／木杆、米粒、绿豆和硬币场景，无需选择类别或手动画参考框。图片在本机处理。

当前版本 **v0.2.0**：自动场景分类、严格几何分根、CountGD++ 通用候选计数、启动预加载、模型持续常驻，以及完整 Windows 启停与模型恢复脚本。

> 当前是可核对的计数原型。密集接触、遮挡、反光、断粒和裁边仍可能漏检或重复；“候选数”不能直接当作准确库存总数，分类正确也不代表数对。

## 已安装的本机

在项目根目录运行：

```powershell
.\start.ps1
.\status.ps1
.\stop.ps1
```

也可双击 `start.cmd` / `stop.cmd`。启动成功后访问 [http://127.0.0.1:5080/](http://127.0.0.1:5080/)。默认只监听本机。

启动在后台运行，等待计数和分类模型均预热后才报告成功。重复启动复用同一实例；停止会关闭本项目网页与所属模型进程并释放资源，不按 Python 名称或端口盲杀其他项目。完整参数及故障处理见 [运行维护](docs/OPERATIONS.md)。

## 新机器安装

建议 Windows 10/11、Python 3.12、Git、NVIDIA GPU 和兼容 CUDA 12.6 的驱动。通用 GPU 深度计数新建会话前要求至少 **4 GiB 空闲显存**。

```powershell
git clone https://github.com/Mikky574/bamboo-stick-counter.git
cd bamboo-stick-counter
.\setup_auto.ps1
.\start.ps1
```

安装准备网页 `.venv`、模型 `.venv-countgdpp` 两个环境、固定版本 CountGD++ 源码和 Windows 适配，以及带校验的模型。具体选项见 [安装和离线恢复](docs/SETUP.md)。`.cmd` 入口不会修改系统全局执行策略。

## 包含当前模型

- [v0.2.0 模型发布附件](https://github.com/Mikky574/bamboo-stick-counter/releases/tag/v0.2.0)
- [文件清单、来源与 SHA-256](models/release-manifest.json)

| 必需资产 | 原始体积 | 用途 |
| --- | ---: | --- |
| CountGD++ + BERT 配置／分词器 | 约 1.251 GB | 文本定位、自动示例、密集目标候选计数 |
| 本项目 ConvNeXt-Small v2 | 约 198 MB | CPU 场景分类和方法路由 |
| 合计 | 约 1.45 GB / 1.35 GiB | 恢复当前默认识别流程 |

源码留在 Git，权重作为同仓库 Release 附件，克隆源码不会下载大模型。虚拟环境、重复的旧训练权重、临时输出及未核实重分发许可的商品／训练图留在本地。TinyCLIP 为可选历史回退；旧 GeCo2 权重仍在 [v0.1.0](https://github.com/Mikky574/bamboo-stick-counter/releases/tag/v0.1.0)。

已有环境仅恢复／检查模型：

```powershell
.venv\Scripts\python.exe tools/fetch_models.py --tag v0.2.0
.venv\Scripts\python.exe tools/fetch_models.py --verify-only
```

## 显存与速度

启动时计数模型加载到 GPU、分类模型加载到 CPU 内存，之后持续常驻，不按空闲时间卸载。常驻的是模型权重，每张图仍重新计算；停止、主动释放、进程失败或超时清理会释放对应资源。

**显存不是固定上限。** 权重之外还有输入张量、注意力中间结果、CUDA 上下文和分配器缓存；图片形状、处理路径、框数量及软件版本可能改变峰值。4 GiB 是准入门槛，不是显存硬配额。

RTX 4090 / Torch 2.6、当前单请求和 800／1333 缩放上限下，预热后 PyTorch reserved 约 2002 MiB；已测最大形状峰值约 **2690 MiB reserved / 2303 MiB allocated**。整卡 `nvidia-smi` 总量还包含其他进程。处理更多图片不保存全部图像，但分配器可能保留遇到过的较大缓存块。

开发验收：并行启动准备约 8.1 秒；第一张密集黄褐米约 1.24 秒、再次 0.93 秒，棉头约 0.52 秒；分散米粒约 0.02 秒、盒装筷子约 0.76 秒。这些是特定输入的实测，不是所有图片或机器的速度承诺。详见 [模型和显存说明](docs/MODELS.md)。

## 计数路径

1. 简单分开的颗粒先走 CPU 几何计数。
2. CPU 场景分类给出路由提示；签类杆身、平端和尖端都可为合理视角。
3. 长杆优先核对轮廓、截面、端帽和连续轨迹，纸带上下不重复，不按“双数”乘二。
4. 通用模型先检测，再从当前图片自动选择示例并推理。
5. 米粒默认提示完全无初始候选时，才尝试黄褐米提示，保持阈值和至少三个去重种子。

界面支持标点、按住看原图、局部放大和编号核对；超过120个候选时主图只显示点。无法可靠计数显示“—”，不会把失败当作0。完全遮挡或容器内部的数量不能由单张照片确定。

## 文档与维护技能

- [安装](docs/SETUP.md) · [启停与故障](docs/OPERATIONS.md) · [架构和 API](docs/ARCHITECTURE.md)
- [模型与显存](docs/MODELS.md) · [验证和限制](docs/VALIDATION.md) · [发布流程](docs/RELEASE.md)
- [项目技能 SKILL.md](.agents/skills/bamboo-counter/SKILL.md) · [维护约定](AGENTS.md) · [变更记录](CHANGELOG.md)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)

早期研究 CLI 和 GeCo2 对照继续保留。开发过程说明归档在 [历史记录](doc/history/development-notes-before-v0.2.0.md)，当前操作以本页与 `docs/` 为准。
