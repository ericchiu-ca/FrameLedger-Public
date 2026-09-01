# 公开快照说明

## 1. 状态

本目录是 2026-09-01 建立并发布到 `ericchiu-ca/FrameLedger-Public` 的**干净公开快照**。它有意不复制来源私有仓库的 `.git` 历史；公开历史从 root commit `1d7d37265ad96202649584153d2efa8389598d67` 开始。

候选目标是保留可审阅、可测试的软件实现，同时不公开真实研究素材或来源绑定信息。

## 2. Allowlist

候选只包含：

- `src/frameledger/` Python package；
- `tests/` 通用 unittest 和合成 fixture；
- `phase2/apple_vision_ocr/` Swift helper；
- `phase2/mlx_whisper_asr/` helper、runner 和两个 requirements 文件；
- `pyproject.toml`、`uv.lock`、`.python-version`、`.gitignore` 和 MIT `LICENSE`；
- README、CHANGELOG、CONTRIBUTING、SECURITY 和核心维护文档；
- GitHub Actions CI/security workflows；
- 文档链接与 dependency license inventory 检查脚本。

## 3. 明确排除

候选不包含：

- 来源仓库的 `.git` 目录、commit、tag 或 branch history；
- 真实视频、音频、帧、截图、模型、cache 或生成输出；
- 真实文件 basename、source/media hash、时间段和 OCR/ASR 文本锚点；
- 人工 annotation、私有 benchmark、代表性 case 记录或 freeze archive；
- 私有讨论、研究库存、账户、联系人、密钥或本地绝对路径；
- 未确认再分发权的素材和商业内容。

排除私有研究历史意味着公开使用者无法仅凭该仓库复现原有真实 benchmark；这是有意的隐私/权利边界，不得通过公开文档补写相关内容。

## 4. 已执行的静态筛选

复制前，源码、测试和 helper 已检查以下类型：

- 已知私有 case 标识和专题文档引用；
- 64 位十六进制来源 fingerprint；
- 用户绝对路径；
- annotation/freeze 文件依赖；
- 常见 secret pattern 和媒体扩展名。

源码中保留的媒体扩展名、合成 `source.mp4` fixture、通用中文 Unicode 文件名和内容过滤词是功能/测试 contract，不是来源媒体或真实研究证据。

## 5. 发布前仍未完成

| 门禁 | 当前状态 |
|---|---|
| 项目 `LICENSE` | MIT；`LICENSE` 与 package metadata 已一致 |
| Git 历史 | 独立 `main`，只含公开快照历史；完整 history Gitleaks scan 已通过；没有 tag 或 Release |
| Hosted CI | CI run `33527890487` 与 Security run `33527890473` 通过 |
| PVR URL | 已设置并绑定 `ericchiu-ca/FrameLedger-Public` |
| PVR 端到端 | API `enabled: true`；无登录页面显示报告入口并正确跳转登录；独立非协作者提交/接收仍待验证 |
| Native/真实视频 | Apple Vision、MLX、FFmpeg、完整 workflow 和 UI 文件授权仍需人工验证 |
| 法律/权利 | 自有代码权属、依赖、模型、FFmpeg 和未来公开 fixture 仍需人工确认 |

## 6. 候选验证命令

```bash
python scripts/check_docs.py
python scripts/check_license_inventory.py
python scripts/check_public_snapshot.py
UV_CACHE_DIR=/tmp/frameledger-public-cache uv lock --check
UV_CACHE_DIR=/tmp/frameledger-public-cache uv run --frozen \
  python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/frameledger-public-cache uv run --frozen frameledger --help
```

本地构建、isolated wheel、Gitleaks working-tree/history scan 和两套 dependency audit 的实测结果记录在 [TESTING.md](TESTING.md)。不能把来源私有仓库的结果当作本公开快照结果。

## 7. 发布顺序

1. 完成本地验证并复核候选 diff；
2. 核对 MIT `LICENSE`、版权主体和 package metadata；
3. 使用已确认的 GitHub repository `ericchiu-ca/FrameLedger-Public`；
4. 确认 `SECURITY.md` 中的精确 PVR URL 与目标仓库一致；
5. 以 `1d7d37265ad96202649584153d2efa8389598d67` 创建公开 root commit；
6. 在明确授权后创建并推送 `ericchiu-ca/FrameLedger-Public`；
7. 启用 PVR，并在取得独立非协作者账号后补做实际提交/接收；
8. Hosted CI、权利和人工 native gate 全部通过后再创建 `v0.1.0` tag。
