# 安装、模型下载与离线恢复

## 环境

推荐 Windows 10/11、Python3.12（脚本接受3.11–3.12）、Git、NVIDIA驱动。模型环境使用Torch2.6.0／torchvision0.21.0 cu126；只运行预编译推理不要求另装完整CUDA Toolkit。新建通用GPU会话至少需4GiB空闲显存，详细峰值见MODELS.md。

在仓库根目录执行：

```powershell
.\setup_auto.ps1
```

指定Python或代理：

```powershell
.\setup_auto.ps1 -PythonCommand 'C:\Python312\python.exe' -Proxy 'http://127.0.0.1:7890'
```

上例路径和代理端口只是示例，按实际安装替换。安装会保留现有环境和模型，不会启动服务；失败时显示具体步骤，不要以最后一条文字代替命令退出状态。

脚本创建网页`.venv`，安装requirements.txt；创建独立`.venv-countgdpp`，安装CUDA Torch和requirements-auto.txt；克隆固定版本CountGD++并打Windows补丁；从v0.2.0发布清单下载和验证当前两个模型包。不会默认下载旧GeCo2、TinyCLIP和训练用重复源权重。

已存在vendor若不是固定提交会停止并保留目录，请核对本地修改后自行处理，不会强行reset。完整包约1.45GB，首次还需Torch等依赖的下载空间；应为下载ZIP、临时解包和安装后权重预留额外磁盘空间。

## 只处理模型

```powershell
.venv\Scripts\python.exe tools/fetch_models.py --tag v0.2.0
.venv\Scripts\python.exe tools/fetch_models.py --verify-only
```

下载使用源码中的models/release-manifest.json，验证整包体积／SHA-256与逐文件摘要，支持已校验下载进度恢复；校验失败不会直接加载权重。`--verify-only`不触网、不运行GPU、不安装依赖。

离线资产恢复：

```powershell
.venv\Scripts\python.exe tools/fetch_models.py --archive-dir 'D:\CounterModels\v0.2.0'
```

该目录放两个原名ZIP；manifest仍使用当前源码版本。安装脚本也支持`-ModelArchiveDirectory`。这只是模型离线恢复，不代替pip依赖和Git上游源码的离线准备。

已有哈希不同文件默认保留并报错；备份且确实打算恢复发布版本时使用`--repair`。工具仍会保留经过有效校验的用户自选分类profile／checkpoint，不悄悄覆盖自训练模型。不要随意跳过哈希校验或从不可信来源加载pickle权重。

## 运行

安装完成运行`.\start.ps1`，停止运行`.\stop.ps1`。优先使用受管理脚本，不以`python app.py`代替：手工前台实例不具有服务控制器的进程组身份，停止脚本不会猜测关闭它。详见OPERATIONS.md。

无可用GPU时完整启动脚本不会将“仅CPU降级”报告成完整就绪。研究场景可直接调用CPU几何模块或显式CLI；默认网页通用深度计数不会静默改为数分钟CPU运算。
