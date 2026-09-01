# 贡献与文档维护指南

FrameLedger 处理可能包含受限制媒体、识别文本和研究证据的本地文件。修改必须保护来源只读边界、会话授权和既有产物。

## 开始前

```bash
git status --short --branch
git log -1 --format=fuller
git tag --list --sort=version:refname
```

- 阅读 `README.md`、`SECURITY.md` 和相关 `docs/`；
- 保留全部未提交修改，不回滚或整理无关文件；
- 不提交源媒体、音频、模型、真实 OCR/ASR、人工标注、benchmark evidence、生成输出、密钥或账户信息；
- 不处理目录、`.part` 或无权使用的视频；
- 不执行 commit、push、tag、Release、仓库公开或云端变更，除非维护者明确授权。

## 变更边界

- 每次实验写入新的输出目录；
- helper/runtime 保持隔离，不把 MLX 加入主环境；
- 不把原始 OCR/ASR 升格为正确金融事实；
- 不把时间邻近称为语义匹配；
- 不把单视频命令、目录选择或 browser drop 扩展成目录扫描；
- UI 保持 `127.0.0.1`、显式文件、单 active batch、最多 50 项、严格串行；
- 公开测试必须使用合成或有明确再分发许可的 fixture。

## 文档同步

| 变化 | 必须检查/更新 |
|---|---|
| CLI、安装或最小使用方式 | `README.md`、CLI help 测试 |
| 技术栈、模块或外部依赖 | `docs/ARCHITECTURE.md` |
| 测试、fixture、CI 或人工 gate | `docs/TESTING.md` |
| package version 或发布流程 | `docs/VERSIONING_AND_RELEASES.md` |
| 用户可见、依赖、安全或删除变化 | `CHANGELOG.md` 的 `Unreleased` |
| 依赖版本或许可证 metadata | `docs/THIRD_PARTY_LICENSES.md`、inventory check |
| 公开快照范围 | `docs/PUBLIC_SNAPSHOT.md` |

结果必须区分 Git 证据、仓库文件、当前实测和无法确认项。没有 commit/tag/记录时直接写“无法从当前公开历史确认”。

## 验证

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv lock --check
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv sync --frozen
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen \
  python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger --help
python scripts/check_docs.py
python scripts/check_license_inventory.py
python scripts/check_public_snapshot.py
```

文档-only 变更也必须检查本地链接、lockfile/版本一致性、CHANGELOG `Unreleased` 和敏感信息。公开前用专用 secret scanner 扫描完整历史与工作树。

原生/真实视频变更的人工 gate 见 [docs/TESTING.md](docs/TESTING.md)。网络、native runtime 或沙箱阻塞必须与代码失败分开报告。

## 安全报告与发布

- 疑似漏洞遵循 [SECURITY.md](SECURITY.md)，不要创建公开 issue；
- 在新公开仓库名称确定后，必须更新并验证 `SECURITY.md` 的 Private Vulnerability Reporting URL；
- 贡献按项目 MIT License 提交；提交者必须有权贡献相关代码、文档和 fixture；
- 发布条件和首个版本规则见 [docs/VERSIONING_AND_RELEASES.md](docs/VERSIONING_AND_RELEASES.md)。
