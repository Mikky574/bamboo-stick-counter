# 架构和 API

网页仅向 `POST /api/count` 发送 `{ "image": "data:image/jpeg;base64,..." }`，默认 `engine=smart`。`app.py` 解码验证图片，`smart_counter.py` 选择计数路径，返回原图尺寸、候选中心和元数据。分类分数只用于路由，不是计数准确率。

| 文件 | 职责 |
| --- | --- |
| `particle_gate.py`、`grain_counter.py`、`mung_counter.py` | 清晰分离颗粒和颜色几何检查 |
| `scene_classifier.py`、`supervised_scene.py` | CPU ConvNeXt-Small v2 场景判断 |
| `rod_silhouette.py`、`laid_rod_counter.py` | 分离长杆轮廓、横截面分根 |
| `boxed_rod_counter.py`、`closeup_rod_counter.py`、`carded_rod_counter.py` | 盒装、近端和背板支撑轨迹 |
| `countgdpp_worker.py`、`auto_counter.py` | 独立 CUDA 模型、自动示例和候选计数 |
| `inference_process.py`、`model_startup.py` | 所属进程、期限、常驻和启动预热 |
| `service_control.py` | Windows 后台服务身份、启停及资源组清理 |

## 生命周期

网页进程不持有 CUDA 模型。两个模型在 `.venv-countgdpp` 的独立进程中运行，以 JSON-lines 接收任务并写短期结果。启动预热使用合成灰图及合成框，不使用用户图片、人工参考或历史答案。

空闲超时为 `None`，成功请求后不创建卸载计时器；显式释放、停止和失败／超时清理仍生效。常驻复用同一模型实例，不缓存图片答案。GPU worker 请求等待期限60秒、分类30秒，前端网络与JSON读取合计100秒；启动、编码和清理另计。客户端超时不保证后台已取消，前端不自动重试。

## 结果结构

结果含 `count`、`points`、`width`、`height`、`meta`。点至少有原图像素 `x/y`，也可能含半径 `r`、框 `box`、杆轴线 `segment` 和未校准 `score`。

`prediction_valid:false`、`count:null` 或 `no_initial_exemplars` 等诊断表示无法报告数量，前端清空旧点并显示“—”。深度候选带 `count_kind`；`count_is_total:false` 与 `review_required:true` 不能被上层当准确总数覆盖。`grain_candidates` 包括可见谷粒和断粒，`visible_rod_candidates` 只指可见杆轨迹。

## 辅助接口

- `GET /api/model-status`：两个模型的当前 `model_resident`、常驻策略、资产及启动历史。`startup.state` 不能代替当前驻留状态。
- `GET /api/service-status`：受管理服务的实例身份与就绪信息；未受管理的前台实例不允许停止脚本猜测处理。
- `POST /api/release-model`：释放本项目两模型，网页不退出；正在识别时拒绝。之后再用深度模型仍需加载。
- `POST /api/export`：按原图和提交点导出标记 PNG。

旧 `engine/category/device` API 和 CLI 保留研究用途，不等同于默认网页路径。GPU 不可用时网页不静默使用可能等待数分钟的 CPU 深度推理；简单 CPU 路径仍可单独调用。

自动路由给 `detect_auto` 设置 `allow_category_recovery=true`。仅原类别得到 `no_initial_exemplars` 或 `no_final_detections` 时，以同一设备、权重和阈值做一次支持目标的语义重选；只有不同类别且确有候选才采用。重选失败仍拒绝，不递归重试，不把模型加载、运行或超时异常当类别错误。元数据 `category_recovery` 保留原类别、失败原因、语义证据及是否采用。首次加载耗时与重选耗时合计，模型不重复加载。

显式 API 默认不重选。主站自动 CPU 回退可发送 `{ "engine":"countgdpp", "category":"rice", "device":"cpu", "allow_category_recovery":true, "image":"..." }`；CLI 对应 `--recover-category`。手工 `--text` 不会被覆盖。此能力与原本严格限定的黄褐米提示回退独立，不放宽后者的阈值、三个去重种子或显式米粒要求。

新增统一语义类别 `insects`（提示 `insect`），无需为每种昆虫增加分类权重。结果为 `insect_candidates`，必须保留 `review_required:true` 和 `count_is_total:false`；这不是专用昆虫模型，也不代表已解决翅膀重复和遮挡漏检。

可选视觉输入 `exemplar_points` 的格式、坐标、严格校验和示例单次前向见[点选示例](POINT_EXAMPLES.md)。有当前请求选点时绕过分类、传统几何、文本自动示例与类别恢复；不能将离线人工参考作为输入。`point_prompts.py`只做无模型输入校验和方框换算，CountGD++仍在原隔离worker中运行。

自动 `bamboo` 端面路径的 `end_face_review.py` 只在至少两个紧凑示例与明显长杆框同时存在时触发额外确认，使用同一图、同一自动示例和原阈值。确认提示为 `circular wooden end face`。仅大小接近且紧凑的密集端面采用0.7倍中位短边的近邻中心抑制；每对还要求短边尺寸比不超过1.5。尺寸混杂不作该合并，端面遮挡仍需核对。它不参与整杆、尖头、昆虫或手动点示例路径。

上传上限20 MiB、解码后3000万像素；网页和API `no-store`。轮转请求日志只记状态、耗时与诊断，不记录图片或base64。服务只绑定本机；公网部署需另外设计鉴权、配额、TLS和队列。
