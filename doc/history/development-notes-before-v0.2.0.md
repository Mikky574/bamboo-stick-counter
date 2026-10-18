# 历史开发记录

以下为 v0.2.0 收敛前的开发说明，含当时的页面与缓存策略；当前操作以根目录 README 和 docs/OPERATIONS.md 为准。

# 签数：GeCo2 本地端面计数

## 图片一键计数（2026-10-18）

网页仅保留“上传图片”和“开始识别”，不需要选择类别、模式、参考框或运行设备。请求仅发送 `{image: dataURL}`，计数器先检查清晰分开的重复颗粒，再由 CPU 场景分类器辅助选择端面、尖头、平铺长杆、棉头、绿豆等方法。本机已激活微调的 ConvNeXt-Small V2；未配置训练权重时使用 TinyCLIP。分类不确定时回退计数模型的类别探测；没有可靠结果时显示“—”。

简单颗粒与支持的长杆几何方法在 CPU 计数；网页需要通用深度模型时检查 GPU 资源，已有本项目常驻模型可复用，否则要求至少4GiB空闲显存。资源不足会及时提示，保留图片稍后重试，不再悄悄启动可能等待数分钟的CPU深度推理。显式指定旧 `engine/category/device` 的研究 API 与 CLI 仍支持CPU。使用 `.venv\Scripts\python.exe app.py` 启动时，并行预加载 GPU 计数模型和 CPU 分类模型，做无用户图片的前向预热；完成启动准备后再监听网页端口。两模型持续常驻，不再因闲置自动退出，直至主动释放、服务退出或推理超时/失败清理。常驻的是权重，不缓存图片答案；GPU 不可用时会记录降级状态，仍可使用简单 CPU 计数。直接导入 `app` 不触发模型加载，其他 WSGI 启动方式须显式调用 `model_startup.prepare_models()`。

当前启动常驻验收：并行准备8.08秒，两会话在第一张真实图片前已就绪；密集米粒第一张1.24秒、再次0.93秒，换棉头图0.52秒，模型加载均为0且复用既有会话。分散米粒0.02秒、盒装筷子0.76秒保持既有结果。GPU预热后PyTorch reserved约2002MiB，不等于整机全部显存用量；其他输入形状仍有额外计算/显存分配耗时。启动报告为 `outputs/startup_models.json`，真实image-only HTTP响应及哈希见 `outputs/startup_residency_20261018/live_summary.json`。

密集黄褐色米粒新增有限的语义回退：分类已接受米粒，且默认 `rice grain` 在原阈值0.23下没有找到任何初始目标时，尝试 `brown rice grain`。至少3个经去重的初始目标才自动生成参考框；不降阈值、不读取人工参考点，也不修改自定义文本或自动类别探测。结果显示“可见谷粒候选”，包括可见断粒和图边截断部分，不能当完整谷粒总数。用户密集图默认提示0目标，回退初测343候选；独立选定两块局部人工核验17个清晰米粒均匹配，无同粒重复，整图未完成逐粒真值标注。证据在 `outputs/dense_grain_20261018`、`outputs/dense_grain_audit_20261018`，输入保存在 `samples/dense_grain_regression_20261018`。变换同一张图会得到345–348候选，说明仍需核对，不代表整体准确率。

重启后真实image-only HTTP验收：黄褐米原图343候选，首次11.78秒（GPU模型加载5.14秒，另含分类加载及进程启动），连续0.97秒；输出坐标与独立局部审计对应的标记完全一致。原分散米粒30个、盒装筷子10根和空白拒绝保持；该次验收时GPU空闲缓存为900秒，之后按用户要求改为启动预加载并持续常驻。完整响应、请求ID和模块校验值在 `outputs/dense_grain_20261018/live_summary.json`；这是接口验收，未自动操作浏览器。

长等待现场确认曾因两次请求间隔160.75秒超过旧120秒缓存，触发14.81秒冷请求，其中模型加载7.84秒。后续一次实际请求因闲置约75分钟再次触发冷启动，耗15.894秒，其中GPU加载8.210秒、CPU分类加载4.051秒；随后换棉头图仅1.191秒。现在改为启动预加载并常驻，加载时间移到服务启动阶段；页面和API响应增加 `no-store` 及UI版本头，旧页面需刷新获取新脚本。前端保留100秒等待期限及按钮恢复逻辑，GPU worker单次60秒、分类worker30秒期限保持。相关行为验证：`python -m unittest verify_runtime_safety verify_worker_timeout verify_rice_prompt_fallback`、`node verify_frontend.cjs`。 启动与常驻契约验证：`python -m unittest verify_model_startup verify_startup_residency`。`/api/model-status` 的 `automatic.model_resident` 与 `scene_model.model_resident` 表示当前驻留情况，`idle_timeout_seconds: null` 表示无空闲释放；`startup` 是当次启动历史，不能代替当前驻留状态。`outputs/startup_models.json` 保存启动准备结果。

显存门槛已由原统一8GiB调整：实测方图的PyTorch缓存峰值约2002–2212MiB，最大处理形状1333×800和800×1333分别约2690MiB reserved、2303MiB allocated；4GiB门槛为CUDA上下文及波动留余量，不是模型实际占用。网页路由、worker初始化与auto设备选择共用 `runtime_capacity.py` 的同一常量，避免外层放行、内层仍按8GiB拒绝。测量在 `outputs/orange_pack_rods_20261018/gpu_measurements*.json`，仅证明本机这些输入的资源用量。

本机启动 `./start.ps1`，打开 http://127.0.0.1:5080/ 。新机器运行 `./setup_auto.ps1` 准备计数模型和小分类模型，或在已有环境运行 `python prepare_scene_model.py --proxy <代理地址>` 下载、校验 TinyCLIP。分类权重约93.8MB，CPU推理；不把文本相似度当作可靠概率或数量。

真实简单图回归：米粒30、分散绿豆17、硬币8均按像素自动计数，不读取样本ID、参考点或文件名。平铺增加专用CPU横截面分根，四张筷子图均得到10根，两个事前参考图逐根通过；另外五张浅色、交叉或密集长杆图仍拒绝。密集图和尖头仍有漏检/重复，当前原型不能保证所有场景正确，分类器也不能替代实例分离。

前端已更新为“见数”：支持拖放/粘贴、按住查看原图、悬停局部放大；超过120个标记时主图只显示圆点，放大镜显示编号。GPU模型输出显示“检测候选数”，不把分类得分当作计数准确率。无可靠结果显示“—”。

长等待防护：场景worker等待预算30秒，常驻GPU计数worker等待预算60秒，超时会清理本项目所属worker并允许下一次重新启动；进程创建、编码和清理时间另计。网页对请求与JSON解析统一设置100秒等待上限，恢复按钮且不自动重试，并明确客户端停止等待不代表后台任务已取消。安装好的场景worker报错时直接提示，不再吞掉错误后追加昂贵的类别探测。HTTP开始/结束、错误正文及耗时写入轮转的 `outputs/count_requests.jsonl`，不记录上传图像或base64数据。验证：`python -m unittest verify_runtime_safety verify_worker_timeout`、`node verify_frontend.cjs`。

盒装分支还限制候选框边和框对数量，四个旋转共享1秒合作预算，超限返回无法可靠计数，不截断候选后少计。合作预算不能抢占单次OpenCV算子，处理尺寸仍限制为1200像素。合成40条长线压力图修复前耗17.57秒，修复后约0.47秒拒绝；已有礼盒10/10和近景4的结果保留。验证：`python -m unittest verify_geometry_budget verify_packaged_rods`。早先防卡HTTP验收在 `outputs/hang_diagnosis_20261018/live_summary.json`：首次分类加载约5.1秒，其后两张筷子0.62/0.32秒，分散米粒30个约0.04秒；当时仍采用8GiB门槛，7931MiB空闲时密集棉头约0.27秒提示资源不足。现已根据实际峰值统一调为4GiB，不再因这个显存水平拒绝新建GPU模型。

2026-10-18早先真实HTTP实测：GPU首次端面13.38秒（含模型加载），连续端面0.81秒、棉头0.65秒、尖头0.61秒。当时闲置120秒后自动释放，之后改为15分钟，再改为当前的启动预加载并持续常驻。7张已有人工参考开发图片中6张数对、1张贴合木杆拒绝；这些小样本不能推断整体计数准确率。详细响应在 `outputs/counting_overall_current_20261018` 和 `outputs/gpu_latency_20261018`。前端行为检查运行 `node verify_frontend.cjs`。

平铺长杆新增外轮廓一致性分支：先对完整、分离、细长的轮廓填充并在三种分割阈值下核对，避免把内部油漆线或印字当新根；不通过时再使用原横截面缝隙方法。用户带红线黑圆图案的条纹筷子已从无法计数恢复为10根，原9张图的已有结果保持。标记跨越多根、相贴成组、交叉和黄背景粘连米粒仍可能拒绝；没有按文件名或读取 `truth.json` 推断数量。回归运行 `python -m unittest verify_rod_surfaces verify_smart`。

规则礼盒与近景新增可见杆端候选分支：先检查完整分离轮廓和原横截面，再检查盒框内连续收尖轨迹、或有杆身支撑的近端端面。用户新发三张照片的真实 image-only HTTP 结果分别为10、10、4；纸带上下不重复，按单根可见轨迹计数，不按双数乘二。第三张上端裁切，不能推断整盒或遮挡总量。页面统一显示“可见筷子候选”。部分重压缩、缩放或曝光变化仍会拒绝；当前是开发照片回归，不是独立泛化准确率。真实输出和标记图见 `outputs/boxed_rods_20261018/after_summary.json` 与 `after_comparison.jpg`。回归运行 `python -m unittest verify_packaged_rods`。

倾斜彩色背板新增 `carded_rod_counter.py`：保留分类流程，杆身、平端、尖端都作为合理的签类视角。若这些视角之间有冲突，允许先检查严格的完整单根轨迹；成功只采用一个计数方法，失败保留原路由。新橙色背板原图识别10个可见候选，真实像素移除后的9/8/7根分别输出9/8/7；不是按对数乘二。方法透视转正后联合检查亮暗边、收尖、独立端帽与跨阈值一致性，纸带上下不重复；不完整亮边的裁切及单板画线拒绝。30个同源审计案例为22正确、8拒绝、0错计，仅属开发回归，不能当独立准确率。回归：`python -m unittest verify_carded_rods verify_rod_routing`；真实接口及标记图在 `outputs/orange_pack_rods_20261018/`。

最终真实image-only HTTP：橙色背板原图10根，首次分类加载4.22秒、连续请求0.28秒；原两张礼盒10/10、近景4、分散米粒30保持。棉头深度模型在CUDA重新加载7.85秒、常驻后0.67秒，输出仍是未逐头核验的107个候选。记录在 `outputs/orange_pack_rods_20261018/live_summary.json` 和 `live_warm_summary.json`。

详见 [图片自动路由与小分类器](doc/图片自动路由与小分类器.md)。原有模型对照与 API 研究记录见 [免框选使用说明](doc/免框选自动计数使用说明.md)。

以下是历史 GeCo2 对照接口及早期页面的安装、验证记录；当前网页操作以上面的上传识别流程为准。

已经接入官方 GeCo2 预训练计数模型，在本机 RTX 4090 上运行。上传图片，框选少量完整端面作为提示，模型输出目标中心，再逐点编号计数。默认关闭框 NMS，可切换 NMS 对照；保留 OpenCV 基线。

## 安装与启动

推荐 Python 3.12。Windows + NVIDIA 显卡：

```powershell
git clone https://github.com/Mikky574/bamboo-stick-counter.git
cd bamboo-stick-counter
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install -r requirements-model.txt
.venv/Scripts/python.exe download_weights.py
.venv/Scripts/python.exe app.py
```

访问 http://127.0.0.1:5080 。服务仅监听本机，推理不上传图片。环境准备好后也可运行 `./start.ps1`。当前原开发电脑已准备好环境和权重。

无 NVIDIA 显卡时可安装对应平台的 PyTorch CPU 版本，速度会更慢。

## Release 文件

下载入口：https://github.com/Mikky574/bamboo-stick-counter/releases/tag/v0.1.0

- `CNTQG_multitrain_ca44.pth`：GeCo2 官方预训练权重，放到 models/；下载脚本可自动获取并校验。
- `fsc147-annotations.zip`：FSC-147 点标注、类别和划分文件及原许可证，解压至项目根目录；不含完整图片集。
- `validation-results.zip`：本次竹签模型输出 PNG、坐标 JSON、页面截图和检查报告。
- `SHA256SUMS.txt` / `manifest.json`：各资产的大小、来源与校验值。

未重新分发厂商棉签目录和无明确许可的木材参考照片；本机原有文件保留，公开仓库保留来源链接。公开克隆可直接运行竹签模型验证和合成图验证，缺少棉签图片时对应验证明确跳过。

![GeCo2 demo](docs/assets/geco2-demo.png)

## 使用模型

1. 点击“加载你的竹签图”，会自动提供三个端面示例；新上传图片则需要自行拖动框选 1–3 个完整端面。
2. 选择 GeCo2 预训练模型，点击“自动识别并编号”。第一次需要加载权重，之后可以缓存筛选结果。
3. “筛选强度”提高会过滤低响应目标，降低会找回可能漏掉的端面。数值是相对中心响应，不能当作置信概率。
4. 默认中心点模式关闭框 NMS；可选择框 NMS 对照模式并设置 IoU。页面报告删除了多少候选。
5. 可以点击补点、删点、撤销；支持显示编号、轮廓、示例框和放大查看。导出 PNG 与 JSON，JSON 包含原图坐标、示例框、模型设置、自动数量和人工修改信息。

示例框是当前图片的推理提示，不会训练模型，也不需要每根手动画框。

## 实际验证结果

用户提供的 705×704 竹签图，选 3 个端面，阈值 0.33：

- 中心点模式：96 个预测点。
- 框 NMS IoU=0.5：仍为 96，删除 0 个候选。
- 阈值 0.20 / 0.50：分别输出 97 / 95 个点。

96 是预测值，尚未建立独立人工真值，不宣称 100% 准确。未在本项目用该图训练模型，但无法排除开源模型预训练时见过其公共来源图片。

模型、数据集和部署改动的详细核查见 docs/open_source_research.md；源码/权重版本和 SHA-256 见 models/provenance.json。

## 模型机制与适配

GeCo2 是有公开权重的少样本计数框架，官方模型预测中心响应图和框。这里使用其学习到的中心峰值，默认关闭框 NMS 和 SAM2 掩膜细化；这不是 P2PNet 式纯点回归，也不是完整官方分割评测配置。局部极大值选峰仍可能漏掉非常接近的目标。

Windows 下调用作者仓库中现有的 PyTorch deformable-attention 实现，不要求编译自定义 CUDA 扩展。推理使用的所有参数严格加载，只有训练辅助头被排除；没有用随机权重替代检测模型。源码及许可证保留在 vendor/geco2。

## 命令行和验证

```powershell
.venv/Scripts/python.exe model_cli.py samples/user_bamboo.png --demo-exemplars --output outputs/model_result
.venv/Scripts/python.exe model_cli.py samples/user_bamboo.png --demo-exemplars --nms-iou 0.5 --output outputs/nms_result
.venv/Scripts/python.exe verify_model.py
```

对其他照片，使用 `--exemplars boxes.json`，文件内容为 `[[x1,y1,x2,y2], ...]`，坐标单位为原图像素。`--demo-exemplars` 仅用于内置用户竹签图。

verify_model.py 检查真实 GPU 推理、坐标范围、缓存、NMS 对照、无效示例拒绝与 PNG 导出。它是功能验证，不是带人工真值的准确率评测。

## OpenCV 基线

页面切换 OpenCV，可用阈值、直径、筛选强度计数。适合对比明显、清晰分开的端面。运行 `python verify.py` 可复现：合成图 80/80、真实棉签裁剪图 3/3、完整粘连棉签图 4/7。合成图不是真实照片；成功裁剪图不代表完整照片正确。

原棉签照片来自安庆嘉欣医疗公开产品目录第 33 页：
https://pub-mediabox-storage.rxweb-prd.com/exhibitor/document/exh-14651c2f-63d7-40e0-9636-59660eaadac5/48e3843f-0482-41cb-9444-65c4f2655ee6.pdf

原目录与提取图片在 samples 中，仅作本地验证，权利归原作者。real_cotton_crop.jpg 裁剪范围：x=135:260，y=35:145。木材参考图来自 https://github.com/LelandYan/Image_processing ，未作为竹签验证证据；没有复制该仓库代码。

## 后续模型设计

客户侧目标为上传后全自动计数，不要求选择品类、框选示例或调整强度。新模型尚在设计阶段，当前页面仍运行上述 GeCo2 与 OpenCV 基线。

- [签类计数模型与算法框架设计](doc/签类计数模型与算法框架设计.md)：点预测、一对一训练、棉头粘连、分块去重、实验路线及数据来源。
- [签类全自动计数系统设计稿](doc/签类全自动计数系统设计稿.md)：产品范围、自动处理流程、部署与验收。

## 开源来源

- GeCo2：https://github.com/jerpelhan/GECO2
- FSC-147：https://github.com/cvlab-stonybrook/LearningToCountEverything
- 竹子计数社区数据页面：https://universe.roboflow.com/object-counting-redi5/bamboo-counting-test/dataset/8 （未下载训练，场景匹配尚未逐图核查）
