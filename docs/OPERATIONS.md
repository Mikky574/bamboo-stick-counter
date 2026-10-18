# 启动、状态、停止与故障

## 推荐入口

```powershell
.\start.ps1
.\status.ps1
.\stop.ps1
```

双击start.cmd／stop.cmd也可；cmd对本次进程使用PowerShell执行策略选项，不修改系统全局策略。脚本从自身路径定位项目，不要求终端当前目录刚好在仓库。后台助手使用隐藏窗口，不弹新终端或浏览器。

启动默认最多等待90秒，先检查端口、建立受管理实例，并预热GPU计数／CPU分类；只有两个模型完整就绪且HTTP实例身份吻合才成功。重复启动返回同一PID，不重复占GPU。默认网址http://127.0.0.1:5080/。

参数：

```powershell
.\start.ps1 -TimeoutSeconds 180
.\start.ps1 -Port 5081
.\stop.ps1 -TimeoutSeconds 30
```

一个项目默认管理一个实例。已运行时指定不同端口会拒绝；先停止再换端口。所有入口返回可供自动化判断的退出码，错误不会假报成功。

## 停止范围

停止先验证PID、创建时间、绝对项目路径、唯一实例标识及实际命令行，再请求优雅退出；默认等待15秒。超时仅终止已验证的本实例Windows Job Object，包含网页及模型后代进程。支持模型预热阶段停止，也保护陈旧PID不误杀后来复用该PID的进程。

端口被其他程序占用时启动拒绝，不结束占用者。停止脚本不按python.exe名称、端口或显存量批量结束。手工启动的`python app.py`显示为非受管理实例，需在原终端Ctrl+C关闭；控制器不猜测它属于谁。

停止后status应报告stopped／无本实例存活，本实例不再提供网页，对应GPU占用释放。端口可能被其他程序占用或复用，不能据此猜测结束进程。其他项目和桌面GPU占用仍可能存在，所以整卡显存不会必然归零。

## 状态与文件

- `outputs/service.json`：实例身份、PID、创建时间、端口及状态。
- `outputs/service.stdout.log`、`service.stderr.log`：后台启动／服务日志，追加写入。
- `outputs/startup_models.json`：本次模型准备结果。
- `outputs/count_requests.jsonl`：轮转的请求开始／结束、耗时及诊断，不含上传像素／base64。
- `outputs/countgdpp_server.log`、`scene_classifier.log`：模型子进程库输出。

这些文件仅在本地，不入Git。运行中不要手改实例状态；若身份无法验证，控制器宁可拒绝停止，也不对其他进程猜测操作。

## 常见问题

**首次启动慢：** 模型读取、CUDA初始化及基础前向预热在启动阶段完成，本机约8秒；磁盘、驱动和硬件不同可更久。之后模型持续常驻，换图仍重新计算。

**页面等待久：** 核对请求ID、`model_load_seconds`、分类加载耗时、`cache_reused`和诊断。资源释放或worker失败后可能重新加载；`no_initial_exemplars`表示没找到目标，不等于GPU卡死。网页100秒后恢复按钮，后台可能仍在处理，不要连续重复提交。

**启动失败：** 看stderr与startup报告；任一模型未ready，完整启动会清理本实例并返回失败。检查模型校验、GPU资源和固定上游源码。显存不够时不要结束其他项目，先由用户安排资源。

**仍有显存占用：** 先核对所属PID，不看整卡数字猜归属。PyTorch reserved含缓存，推理峰值可高于空闲权重；详见MODELS.md。明确需要释放模型但保留网页时可调用`POST /api/release-model`，之后再推理将重新加载。

取证时先保存当前模型状态、对应请求ID与耗时，以及service.json记录的supervisor PID。用Windows进程树找该实例模型后代，再对照`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`；WDDM模式可能返回N/A，应说明该指标不可量化。区分空闲和推理峰值，不用整卡前后差额推断本实例。状态／进程记录含本机路径和实例标识，仅本地保存。

**无法停止：** 确认使用同一份仓库脚本与原`.venv`，保留状态及日志排查；不要改成taskkill全部Python。本次交付要求最终保持停止，验证后再次执行stop。
