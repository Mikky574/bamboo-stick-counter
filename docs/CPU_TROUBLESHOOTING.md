# 迁移后 CPU 推理排查

目标是确认CPU子进程是否启动、依赖是否可用、权重是否加载以及前向是否完成。计数准确性另行核验；`no_initial_exemplars`不代表CPU没运行。

## 原版入口的限制

- `start.ps1`通过服务控制器预加载GPU计数与CPU分类；两个模型均就绪才启动网页。没有GPU时，原版完整启动会失败并清理所属进程。
- `smart_counter.inference_device()`只选择CUDA或拒绝深度计数，没有自动CPU深度回退；简单的CPU几何分支仍存在。
- `countgdpp_worker.py --device cpu`支持独立CPU深度推理。CPU内核可运行不等于原版网页自动支持CPU部署。
- `setup_auto.ps1`面向CUDA环境；不能用它验证或保留纯CPU wheel。旧`auto_cli.py`不支持绿豆，不要用作本次入口。

2026-10-18本机实测：模型Python来自`.venv-countgdpp`，但`torch.__file__`和`torchvision.__file__`实际来自Miniconda系统包；虚拟环境启用了`include-system-site-packages=true`。迁移包没有Python环境，目标机必须独立准备依赖，不能假设本机全局包也会跟过去。本机CPU测试使用Torch2.6.0+cu126并隐藏GPU，没有另装纯CPU wheel或卸掉物理GPU做测试。

## 一键分层取证

将`diagnose_cpu.ps1`放到原项目根目录、`diagnose_cpu.py`放到`tools/`。诊断包不含模型或Python；与完整原项目一起使用，不要求网页启动。

先进入迁移后的项目根目录，图片路径换成目标机实际路径：

```powershell
# 只查解释器、依赖、真实CPU算子和模型文件校验，不加载计数模型。
.\diagnose_cpu.ps1

# 用原始绿豆预设，另测CPU分类，再实际加载和推理。
.\diagnose_cpu.ps1 -Image 'E:\识别数量项目\02-mung-beans-raw.jpg' -Category mung -CheckScene

# 用本地验证可出候选的提示词做执行对照；不把候选数当准确总数。
.\diagnose_cpu.ps1 -Image 'E:\识别数量项目\02-mung-beans-raw.jpg' -Category mung -Text 'green mung bean' -CheckScene

# 无图片也可以测试两种前向路径；使用合成灰图和合成框，只验证模型执行。
.\diagnose_cpu.ps1 -RunModel

# 如果模型安装在另外一个Python环境，必须明确指定。
.\diagnose_cpu.ps1 -Python 'D:\project\.venv-countgdpp\Scripts\python.exe' -RunModel
```

执行策略阻止ps1时，仅给这一次进程参数：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\diagnose_cpu.ps1 -RunModel
```

脚本将GPU只对子进程隐藏，使用本地模型，不安装包、不下载、不启动网页，不释放或终止其他项目的模型。每次创建独立报告目录`outputs/cpu_diagnostics/日期_进程_随机号/`，防止读到上次结果。CPU模型结束后退出；超时只清理由本次Popen持有的存活进程树。终端每约10秒显示当前阶段，默认模型期限600秒，可用`-TimeoutSeconds`调整。

`-CheckScene`分别验证分类器和显式类别计数；它不是完整`smart`路由。诊断JSON保留`full_smart_route_executed:false`。

## 根据状态定位

| 最终状态或证据 | 含义 | 下一步 |
| --- | --- | --- |
| `environment_failed`、`launched:false` | 解释器路径/创建进程失败，尚未进入模型 | 检查`-Python`，确认没有沿用旧机器绝对路径 |
| `environment_failed`且`failed_stage=import_torch` | Python已启动，Torch未成功导入 | 看`environment.log`的缺包/DLL错误及实际解释器路径 |
| `cpu_tensor_and_torchvision_ops`失败 | Torch/vision/其他依赖或CPU算子不可用 | 核对Torch2.6与torchvision0.21是否成对、来源是否混杂 |
| `assets_failed` | 必需模型缺失或SHA-256不符 | 查看`assets.log`，恢复相应模型文件；此工具不会自动下载 |
| `model_failed`、`load_countgd_weights_on_cpu` | CPU进程可启动，checkpoint/BERT/上游代码加载失败 | 看完整`model.log`及报错栈，不把它改成“换图” |
| `cpu_model_warmup_passed` | CPU模型和两种合成前向均完成 | 再测用户图片；仍未验证计数准确率 |
| `executed_no_initial_exemplars` | **CPU第一轮前向已经完成**，文本定位无初始目标 | 检查提示词/图片/路由，不能写“CPU没跑起来” |
| `executed_with_candidates` | CPU实际前向完成并返回候选 | 核验候选点与计数，不等于准确总数 |
| `model_timeout` | 在指定CPU期限内未完成 | 看`progress.json`停在哪一步、RAM和日志；确认业务层是否更早超时 |
| `model_cleanup_failed` | 无法确认本次所属进程树清理完成 | 保留报告，核对本次PID；不要按Python名称批量结束 |

## 报告中必须保留的证据

- `diagnostics.json`：实际解释器、每个阶段的returncode/是否成功创建/是否超时/PID/路径和完整结果。
- `environment_result.json`：Python版本、prefix/base_prefix、Torch与torchvision版本和实际文件路径、是否继承环境外部包、CPU张量与NMS实测。
- `environment.log`、`assets.log`、`model.log`：完整stdout/stderr；不要仅截取最后一句业务提示。
- `model_result.json`：失败阶段、异常、参数实际设备、模型加载时间、分类结果与计数摘要。
- `prediction.json`：原始worker输出。只有实际模型前向完成才产生此文件；检查`meta.device=cpu`、`diagnosis`、`initial_pass_seconds`、`initial_count`和`prediction_valid`。
- `progress.json`：当前或最后阶段。旧`completed`文件不代表新请求成功，必须对应同一报告目录。

这次绿豆原图在本机的结果：默认`mung bean`为`executed_no_initial_exemplars`；`green mung bean`对照提示可以产生CPU候选。前者证明定位失败，后者证明执行链能跑；两者都不是整体准确率证据。

## 不依赖诊断工具的最小命令

```powershell
.\.venv-countgdpp\Scripts\python.exe -c "import sys,torch,torchvision;print(sys.executable);print(torch.__version__,torch.__file__);print(torchvision.__version__,torchvision.__file__);x=torch.ones((2,2),device='cpu');print(x@x)"

.\.venv\Scripts\python.exe tools\fetch_models.py --verify-only

.\.venv-countgdpp\Scripts\python.exe countgdpp_worker.py `
  'E:\识别数量项目\02-mung-beans-raw.jpg' --category mung --device cpu `
  --text 'green mung bean' --output outputs\cpu_direct.json

$LASTEXITCODE
Get-Content -LiteralPath outputs\cpu_direct.json -Encoding UTF8
```

CUDA版本Torch同样可以执行CPU张量；`torch.version.cuda`只是构建信息，不能拿它判定CPU是否能运行。本诊断刻意设置`CUDA_VISIBLE_DEVICES=-1`，此时GPU不可见是预期行为。上游`Failed to load custom C++ ops...CPU mode Only`警告本身也不是终止证据；原项目Windows补丁启用了PyTorch注意力回退，最终看退出码和实际JSON。

## 如果依赖确实没迁移完整

先保留失败日志。可另外建立不继承全局包的CPU检查环境，避免修改当前工作环境。下面是PyTorch官方2.6 CPU wheel组合；本项目只需要torch和torchvision，无需torchaudio。[官方旧版本安装说明](https://pytorch.org/get-started/previous-versions/#v260)

```powershell
py -3.12 -m venv .venv-cpu-check
.\.venv-cpu-check\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-cpu-check\Scripts\python.exe -m pip install -r requirements-auto.txt
.\diagnose_cpu.ps1 -Python '.\.venv-cpu-check\Scripts\python.exe' -RunModel
```

这一步只建立独立检查环境，不会把原版`start.ps1`和`smart`改造成CPU网页部署。若目标是CPU-only网页，需要明确调整启动预热设备、smart路由设备、推理锁/队列、超时和状态展示，保持先分类再按方法计数；不能只在最外层增加一句“GPU失败改CPU”。

## CPU直跑正常，业务页面仍失败时

按一次请求记录链路：图片解码→分类→选中方法→启动worker→实际设备→worker退出码→原始JSON→业务JSON。核对图片SHA-256、缩放/EXIF、category、text、阈值和最终用户提示，不能只比文件名。

重点检查：

1. 迁移后的CPU回退是否仍走`smart`中的GPU准入或GPU-only预加载，导致worker根本未创建。
2. 是否调用旧`auto_cli`或省略类别，使绿豆按默认竹签提示计数。
3. 是否只有网页Python安装了Torch，真正的模型子进程却用另一份缺包解释器。
4. 是否先被业务网关/HTTP客户端30秒期限切断；原版分类30秒、驻留GPU60秒、网页100秒、显式CPU子进程600秒是不同层。
5. 是否把worker异常、超时、无初始目标、后置校验拒绝全部转换成同一句“换清晰图片”。应返回不同错误码及可关联的request_id。
6. 内存不足时核对任务进程实际工作集/系统可用RAM。CPU模型主要占系统内存；完全没有GPU不会使CPU张量自动失效。

诊断脚本和本说明为发布后新增，旧v0.2.0迁移压缩包不含；独立诊断小包只添加这些诊断文件，不替换现有模型和业务计数实现。
