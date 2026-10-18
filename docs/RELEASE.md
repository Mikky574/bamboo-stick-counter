# 源码与模型发布

## 布局

源码、文档、脚本、模型元数据和维护技能进入Git。大模型放同仓库GitHub Release附件。当前发布v0.2.0必需两包：CountGD++／BERT约1.251GB，活动ConvNeXt v2约198MB，总1.45GB。模型包不含Python环境、训练图和临时输出。

GitHub普通Git文件超过100MiB会拒绝，Release单附件需小于2GiB；本次两包均满足。规则来源：[大文件](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Release附件](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)。旧v0.1.0与其GeCo2资产不覆盖。

## 构建与校验

```powershell
.venv\Scripts\python.exe tools/package_models.py --tag v0.2.0 --output release_assets/v0.2.0
.venv\Scripts\python.exe -m unittest verify_model_release -v
.venv\Scripts\python.exe tools/fetch_models.py --verify-only
```

打包使用固定成员和可重复ZIP格式，生成两包、release-manifest.json、SHA256SUMS.txt；源码清单位于models/release-manifest.json。改版本需同步打包目标、源码清单、下载脚本及setup版本，不要让“最新源码”恢复另一个未验证模型。

原许可证随包保留，包只引用相对路径。恢复工具先验证整包、成员集合、体积、链接／特殊成员与路径穿越，再逐文件校验并替换；哈希不同的已有文件默认保留。应另在空目录以`--archive-dir`验证真正恢复，并确认active profile仍为v2。

## 发布顺序

1. 停止服务，检查源码变更、排除凭证、环境、大blob和未经整理的网络原图；用户本地数据不删除。
2. 运行公开行为检查、模型恢复测试及服务控制器测试；必要时真实GPU启停验收，最后停止。
3. 构建模型包，把四个资产先上传到draft Release；核对远端asset size及sha256 digest。
4. 提交并推送具体源码，使用同一提交创建版本标签，把draft目标设为该提交再发布。
5. 核对公开release链接、标签提交、附件摘要和下载URL；检查Git工作区及最终服务状态。

GitHub访问使用用户已有凭证机制，令牌不能进入仓库、日志或发布正文。中断上传时先检查同名远端资产是否完整，不盲目重复创建Release或删除旧资产。文档与提交不得将局部候选核对宣称为总体计数精度。

## 数据范围

当前公开包提供运行模型，不提供整个网络图片训练集。商品图和目录提取图的公开可访问性不等于重分发授权；来源收集代码与许可说明保留。已有合法入库的夹具保留，后来未整理数据继续在本地samples／datasets，不执行递归清理。需要完整训练复现时先明确各来源许可，再单独发布可分发的数据说明和划分。
