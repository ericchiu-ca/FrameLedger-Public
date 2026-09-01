# 版本与发布

## 1. 当前事实

- `pyproject.toml`、`src/frameledger/__init__.py` 和 `uv.lock` 中的 package version 为 `0.1.0`；
- 公开候选有意不继承私有仓库的 Git 历史；
- 当前本地候选没有 commit、tag、GitHub Release 或生产部署；
- 项目采用 MIT License，版权声明为 `Copyright (c) 2026 Eric Chiu`；
- `SECURITY.md` 的 PVR URL 已绑定 `ericchiu-ca/FrameLedger-Public`，仍需在该仓库做端到端测试。

因此现状应称为 **`0.1.0` 尚未发布的 research preview 候选**。首个正式可追溯版本建议为 `v0.1.0`，但只能在全部发布门满足后创建 tag。

## 2. 版本号

采用 Semantic Versioning：

- `MAJOR`：破坏 CLI、机器可读 schema、证据完整性或来源只读边界；
- `MINOR`：向后兼容地增加命令、策略、字段或可选工作流；
- `PATCH`：不破坏 contract 的修复、校验加强或文档修正；
- 预发布：使用 `v0.2.0-rc.1` 等标准 SemVer 标识。

版本必须同步：

1. `pyproject.toml`；
2. `src/frameledger/__init__.py`；
3. 重新生成并检查的 `uv.lock`；
4. `CHANGELOG.md`；
5. 确实发生 contract 变化时对应的 schema/policy version。

Phase 名称和 workflow policy version 不等于 package release version。

## 3. CHANGELOG

- 用户可见行为、CLI、依赖、安全边界、schema、测试/CI 和文档体系变化先写入 `[Unreleased]`；
- 发布时移入 `[X.Y.Z]` 并写实际 ISO 日期；
- 使用 Added、Changed、Fixed、Removed、Security；
- 依赖升级记录包名、前后版本、原因和影响；
- 没有公开 commit/tag 证据的历史不得回填。

## 4. Tag 与 Release

Release tag 使用 annotated `vMAJOR.MINOR.PATCH`，必须：

- 指向已审阅、工作树干净且通过发布门的 commit；
- 与 package version、lockfile 和 CHANGELOG 一致；
- 不移动、不复用、不覆盖；
- tag message 只陈述已验证事实和人工限制；
- 只有获得明确授权后才 push tag 或创建 GitHub Release。

Release asset 不得包含媒体、音频、模型、真实识别文本、annotation、benchmark evidence、账户信息或受限数据。

## 5. 发布前验证

### Repository 与治理

```bash
git status --short --branch
git log -1 --format=fuller
git tag --list --sort=version:refname
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv lock --check
```

- 工作树没有意外文件；
- version、CHANGELOG 和目标 tag 一致；
- 项目所有者已选择 `LICENSE`；
- `SECURITY.md` 已绑定实际目标仓库；
- GitHub Private Vulnerability Reporting 已启用，并由无仓库权限的测试账号确认入口和私密接收；
- `docs/PUBLIC_SNAPSHOT.md` 的所有阻塞项有证据和 owner。

### 测试和构建

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv sync --frozen
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen \
  python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger --help
python scripts/check_docs.py
python scripts/check_license_inventory.py

release_dist_dir="$(mktemp -d /tmp/frameledger-dist.XXXXXX)"
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv build --out-dir "$release_dist_dir"
```

在 `--isolated --no-project` 环境安装并验证 wheel import、版本和 CLI。记录所有 pass/fail/skip；native skip 不等于 native gate 通过。

### 安全、依赖和权利

- 运行 `.github/workflows/security.yml` 中固定版本的 Gitleaks 和 `pip-audit`；
- 检查 dependency license inventory，并人工复核许可证全文、NOTICE、模型和 FFmpeg；
- 确认仓库只含合成或可再分发 fixture；
- 检查 Git author、remote 和文档是否暴露不期望公开的身份或路径；
- 在目标 macOS/Apple Silicon 上完成人工 OCR、ASR、完整 workflow 和 UI 授权边界验证。

## 6. 发布记录

每次发布至少保存：

- version、tag、commit SHA、发布日期和仓库可见性；
- Python、uv、macOS、Xcode、FFmpeg、MLX 版本；
- 两个 lockfile hash 和 model ID/revision/tree hash；
- 自动测试数量、失败和 skip；
- native/真实视频 gate 的非敏感结果；
- wheel/sdist 文件名、size 和 SHA-256；
- 已知限制、安全/许可证状态和 rollback tag。

## 7. 回滚与历史复现

发布后从目标 tag 建立独立 worktree，而不是覆盖当前 checkout：

```bash
git worktree add ../FrameLedger-v0.1.0 v0.1.0
```

旧源视频和输出继续保持只读；复现实验使用新输出目录和该 tag 对应的 lockfile、helper、模型 revision。公开历史从候选的首个 commit 开始，不能从中恢复被有意排除的私有研究历史或数据。
