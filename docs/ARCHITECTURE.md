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

上传上限20 MiB、解码后3000万像素；网页和API `no-store`。轮转请求日志只记状态、耗时与诊断，不记录图片或base64。服务只绑定本机；公网部署需另外设计鉴权、配额、TLS和队列。
