# FrameLedger

FrameLedger 是一个本地运行、证据优先（evidence-first）的金融视频处理工具。它把一个明确选择的视频转换为可追溯的视觉帧、OCR、原始 ASR、时间对齐记录、确定性章节和最小 Markdown 截图报告，并用 SHA-256、绝对时间戳和离线审阅页面保留来源链。

## 当前状态与边界

这是一个**尚未发布的 `0.1.0` research preview 公开候选**。本目录有意不继承私有研究仓库的 Git 历史，也不包含真实源媒体、人工标注、私有 benchmark、冻结归档或生成输出。项目代码和仓库文档采用 [MIT License](LICENSE)；该许可不延伸到仓库外的源媒体、模型或第三方素材。

当前能力：

- Phase 1：有界视频区间的候选帧选择、离线审阅和外部标注评估；
- Phase 2a：对已选 `presentation` / `table` 原图执行本地 Apple Vision OCR；
- Phase 2b：对一个显式视频区间执行本地 MLX Whisper ASR；
- Phase 2c：在源视频绝对时钟上对齐视觉、OCR 和 ASR 证据；
- 确定性章节划分和只包含已验证关键截图的 Markdown 导出；
- 单视频完整 `workflow`，以及只绑定 `127.0.0.1`、最多显式排队 50 个视频且严格串行的 `workflow-ui`。

硬边界：

- 源视频只读；拒绝 `.part`、目录和不支持的格式；
- 输出必须写入新的独立目录，不覆盖既有运行；
- 不提供云端 OCR/ASR、LLM 总结、说话人识别、财务事实纠错或语义匹配结论；
- `chart` 和 `unknown` 路由在 OCR 阶段按策略跳过；
- 完整流程依赖 macOS、Apple Vision、Apple Silicon/MLX、本地固定模型和 FFmpeg；
- 项目没有生产 Web 服务、数据库或云端部署。

## 环境要求

基础流程：

- Python `>=3.12,<3.14`；
- [`uv`](https://docs.astral.sh/uv/)；
- `pyproject.toml` 和 `uv.lock` 中声明及锁定的主依赖。

OCR、ASR 和完整流程还需要：

- macOS 12 或更高版本及 Xcode Command Line Tools；
- Apple Silicon 与可用的 MLX Metal runtime；
- FFmpeg；
- 本地固定 revision 的 `mlx-community/whisper-large-v3-turbo` 模型。

## 安装

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv sync --frozen
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger --help
```

ASR runtime 与主环境隔离：

```bash
uv venv phase2/mlx_whisper_asr/.venv --python 3.12
UV_CACHE_DIR=.cache/uv uv pip install \
  --python phase2/mlx_whisper_asr/.venv/bin/python \
  -r phase2/mlx_whisper_asr/requirements.lock.txt
chmod +x phase2/mlx_whisper_asr/run
```

在转写前下载一次固定模型 revision；实际转写被强制为离线：

```bash
phase2/mlx_whisper_asr/.venv/bin/hf download \
  mlx-community/whisper-large-v3-turbo \
  --revision a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb \
  --local-dir .cache/models/whisper-large-v3-turbo
```

## 最小运行示例

只读探测视频：

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger probe \
  /absolute/path/to/video.mp4
```

运行一个明确的有界 Phase 1 benchmark；输出目录必须尚不存在：

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger benchmark \
  /absolute/path/to/video.mp4 \
  --start 00:05:00 \
  --duration 00:10:00 \
  --output outputs/example
```

本地原生能力和模型准备完成后，可执行完整单视频流程：

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger workflow \
  /absolute/path/to/video.mp4 \
  --output outputs/my-complete-run
```

本地控制器：

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger workflow-ui \
  --output-root outputs/workflows \
  --port 8765
```

然后打开 `http://127.0.0.1:8765/`。控制器只接受显式选择的文件，不扫描目录。

## 仓库结构

```text
src/frameledger/                 Python 主包与 CLI
tests/                           unittest 与合成 fixture 测试
phase2/apple_vision_ocr/         Swift/Apple Vision helper
phase2/mlx_whisper_asr/          隔离的 MLX Whisper helper 与锁定依赖
docs/                            架构、测试、版本、依赖和快照说明
.github/workflows/               CI 与安全扫描
scripts/                         文档和依赖清单检查
```

真实媒体、模型、标注、benchmark evidence、音频和生成输出不属于公开仓库内容。

## 测试

```bash
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv lock --check
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen \
  python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/frameledger-uv-cache uv run --frozen frameledger --help
python scripts/check_docs.py
python scripts/check_license_inventory.py
python scripts/check_public_snapshot.py
```

自动测试大量使用临时目录、合成帧和 fake helper，只能验证代码 contract、路径边界、协议和 fail-closed 行为，不能证明真实 OCR/ASR 质量、完整视频覆盖或素材权利。

## 已知限制

- 公开候选没有私有研究历史、真实 benchmark 数据、annotation 或 freeze package；
- 跨机器复现实验需要使用者自行合法取得并记录完全相同的输入；
- 变量帧率视频使用 nominal FPS 时间戳，可能与逐帧 presentation timestamp 有差异；
- OCR、ASR、路由和章节输出属于有边界的研究证据，不构成金融事实或投资结论；
- GitHub Actions 尚未在这个新候选仓库中运行；
- GitHub Actions 和 Private Vulnerability Reporting 仍需在新公开仓库中取得验证证据。

## 文档

- [文档索引](docs/README.md)
- [架构说明](docs/ARCHITECTURE.md)
- [测试与复现](docs/TESTING.md)
- [版本与发布](docs/VERSIONING_AND_RELEASES.md)
- [第三方依赖许可证](docs/THIRD_PARTY_LICENSES.md)
- [公开快照说明](docs/PUBLIC_SNAPSHOT.md)
- [安全政策](SECURITY.md)
- [变更记录](CHANGELOG.md)
- [贡献指南](CONTRIBUTING.md)
