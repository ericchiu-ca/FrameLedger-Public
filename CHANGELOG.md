# 变更记录

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 分类；版本规则见 [docs/VERSIONING_AND_RELEASES.md](docs/VERSIONING_AND_RELEASES.md)。

这个公开候选有意不继承私有研究仓库的提交历史。无法从公开候选验证的早期演进、日期和决策不会回填。

## [Unreleased]

### Added

- 建立无旧 Git 历史的本地公开候选，只包含允许公开的源码、通用测试、helper、依赖清单和治理文档。
- 增加 macOS arm64 / Python 3.12、3.13 CI，以及 wheel/sdist 和 isolated wheel 验证。
- 增加固定版本 Gitleaks、`pip-audit`、依赖许可证 inventory 和 Markdown 本地链接检查。
- 增加公开快照 allowlist、媒体/模型/私钥、绝对路径、symlink 和文件大小检查。
- 增加公开快照范围、排除项和发布门说明。
- 采用 MIT License，并在 package metadata 中声明 SPDX expression `MIT`。

### Changed

- 文档改为只描述公开候选，不引用私有 annotation、benchmark、freeze 或研究讨论。
- Git 历史从未来公开候选的首个 commit 开始，不推断或复制私有仓库历史。

### Fixed

- 无业务代码修复；本区域只记录公开候选整理和治理变化。

### Removed

- 从公开范围排除真实媒体、人工标注、私有 benchmark、冻结归档和生成输出。

### Security

- 保留源视频只读、输出不覆盖、模型离线、loopback-only 和 fail-closed 安全政策。
- GitHub Private Vulnerability Reporting URL 已绑定 `ericchiu-ca/FrameLedger-Public`；启用和端到端测试结果须另行验证。

## [0.1.0] - 尚未发布的 research preview

当前 package、wheel 和 `src/frameledger/__init__.py` 使用 `0.1.0`。此版本没有 tag、Release 或可验证发布日期。

### Added

- 提供视觉候选帧、Apple Vision OCR、本地 MLX Whisper ASR、时间对齐、确定性章节和 Markdown 导出。
- 提供单视频完整工作流和 loopback-only 串行批次控制器。
- 提供 Python `unittest`、合成 fixture 和严格 helper protocol 测试。
- 主环境与隔离 ASR 环境的锁定依赖基线。

### Changed

- 无法从当前公开历史确认；公开候选尚无首个 commit。

### Fixed

- 无法从当前公开历史确认。

### Removed

- 无法从当前公开历史确认。

### Security

- 代码包含路径、symlink、输出覆盖、源文件邻接写入、helper protocol、模型离线和 loopback 请求的 fail-closed 检查；这不等于独立安全审计。
