# FrameLedger 文档索引

本公开候选只保存当前可维护 contract，不包含私有研究历史、真实媒体、人工 annotation、benchmark evidence 或 freeze archive。

## 核心文档

| 文档 | 用途 | 更新触发 |
|---|---|---|
| [项目入口](../README.md) | 问题、边界、安装、运行和结构 | CLI、环境、用户入口变化 |
| [架构说明](ARCHITECTURE.md) | 技术栈、模块边界、数据流和债务 | 模块、contract、外部依赖变化 |
| [测试与复现](TESTING.md) | 测试、CI、人工门和数据边界 | 测试、CI、复现方法变化 |
| [版本与发布](VERSIONING_AND_RELEASES.md) | SemVer、tag、release 和 rollback | 版本或发布流程变化 |
| [第三方依赖许可证](THIRD_PARTY_LICENSES.md) | 两个 lockfile 的版本与许可证 metadata | 依赖或分发义务变化 |
| [公开快照说明](PUBLIC_SNAPSHOT.md) | allowlist、排除项和公开前门禁 | 公开范围或权利状态变化 |
| [安全政策](../SECURITY.md) | 支持状态、报告渠道、安全范围 | 暴露面、渠道或安全承诺变化 |
| [变更记录](../CHANGELOG.md) | Unreleased 与可验证版本历史 | 每个用户可见、依赖或安全变化 |
| [贡献指南](../CONTRIBUTING.md) | 修改、验证和数据边界 | 协作流程变化 |

## Helper 文档

- [Apple Vision OCR helper](../phase2/apple_vision_ocr/README.md)
- [MLX Whisper ASR helper](../phase2/mlx_whisper_asr/README.md)

## 文档维护规则

1. README 只保留当前入口和最小路径；细节写入专题文档；
2. 所有用户可见、依赖或安全变化先进入 CHANGELOG `Unreleased`；
3. 命令必须从当前 CLI/manifest 验证，版本必须来自 manifest/lockfile；
4. 结果必须带版本、输入边界、验证日期和局限；
5. 无法从公开 Git 历史确认的内容不得从私有文件名或记忆推断；
6. 不把源媒体、真实识别文本、人工标注、模型、生成输出、密钥或账户信息加入 Git；
7. 新增、移动或删除文档时运行 `python scripts/check_docs.py`。
