# 测试与复现

## 1. 测试体系

项目使用 Python 标准库 `unittest`。测试大量使用临时目录、合成图像、fake OCR/ASR helper 和模拟阶段结果，以验证 contract、路径保护、protocol 和 fail-closed 行为。

| 类型 | 代表测试 | 覆盖内容 |
|---|---|---|
| 纯逻辑 | `test_features.py`、`test_selection.py`、`test_routing.py` | 图像特征、选择器、路由 |
| Phase 1 | `test_pipeline.py`、`test_routed_pipeline.py`、`test_report.py` | 有界范围、产物、cap、review |
| Annotation protocol | `test_annotations.py`、`test_asr_annotations.py` | schema、review level、hash 绑定、锚点评分 |
| OCR/ASR | `test_ocr.py`、`test_asr.py`、helper tests | stdio JSON、失败 ledger、质量门 |
| 下游 | `test_alignment.py`、`test_semantic.py`、`test_markdown_export.py` | 同源、coverage、确定性分章、原子导出 |
| Workflow/UI | `test_workflow.py`、`test_local_ui.py` | 阶段短路、源/输出保护、串行批次、路径逃逸 |
| Native smoke | Apple Vision helper、loopback socket tests | 环境不允许时明确 skip |

## 2. 可复制命令

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

首次 `uv sync` 需要依赖注册表网络或完整本地缓存。网络失败必须记录为环境阻塞，不能改写为测试失败或通过。

## 3. 公开候选验证快照

生成日期：2026-09-01。Package version：`0.1.0`。本候选尚无 commit 或 tag。

本地候选验证结果：

| 门禁 | 结果 |
|---|---|
| `uv lock --check` | 通过；解析 9 个 main package |
| 全量 unittest | 167 个测试通过；0 failure/error；2 skip；6.814 秒 |
| Skip 原因 | 当前沙箱不提供 Apple Vision runtime 和 loopback socket；不是通过证明 |
| CLI help | 通过；显示 13 个子命令 |
| Build | `frameledger-0.1.0-py3-none-any.whl` 与 `frameledger-0.1.0.tar.gz` 构建成功 |
| Isolated wheel | import、`__version__ == "0.1.0"` 和 CLI help 通过 |
| Public snapshot | 68 个 allowlisted UTF-8 文件；无媒体、模型、私钥、绝对私人路径、symlink 或超限文件 |
| Gitleaks | v8.30.1 working-tree scan：no leaks found |
| Dependency audit | pip-audit 2.10.1：main 与 ASR 均 no known vulnerabilities found |
| License inventory | 8 个 main、35 个 ASR、合计 39 个唯一 locked package 一致 |
| 文档链接 | 12 个 Markdown 文件中的 26 个本地链接均有效 |
| Workflow syntax | 两个 YAML 文件均可解析 |

候选 repository 仍是 0 commit，因此当前不存在可供 Gitleaks 扫描的 Git history；首个 commit 形成后必须补做 `gitleaks git --log-opts=--all .`。GitHub hosted CI 也必须在未来目标仓库中另行取得 run 证据。

依赖审计的 `--no-deps` 依赖两份输入已完整锁定这一前提；ASR lock 尚未记录下载 hash，见发布文档中的人工门禁。

## 4. CI 范围

### `.github/workflows/ci.yml`

- push、pull request、手工 dispatch；
- `macos-15` arm64；Python 3.12 与 3.13；
- lock、sync、全量 unittest、CLI help；
- Markdown link 和 dependency license inventory；
- Python 3.12 wheel/sdist 及 isolated wheel 验证。

### `.github/workflows/security.yml`

- push、pull request、手工 dispatch、每周 schedule；
- 固定版本及 SHA-256 的 Gitleaks 完整历史扫描；
- 固定版本 `pip-audit` 扫描主 frozen export 和 ASR lock；
- dependency license inventory drift check。

Workflow 只授予 `contents: read`。ASR lock 当前固定版本但不含下载 hash。

## 5. 自动测试不能证明什么

自动测试通过不代表：

- Apple Vision 在目标机器可用或 OCR 正确；
- MLX Metal、固定模型和真实转写通过质量门；
- FFmpeg 能解码目标视频；
- selector 捕获了真实视频全部关键视觉状态；
- OCR/ASR 是正确金融事实，或时间重叠是语义关联；
- 完整长视频的资源成本可接受；
- native picker/browser fallback 在真实环境符合授权边界；
- 项目无漏洞、具备媒体权利或许可证兼容。

## 6. 真实运行人工门

### Phase 1 / OCR

1. 使用合法视频、显式时间范围和新输出目录；
2. 记录 source SHA-256、size、mtime；
3. 使用仓库外的独立 annotation，记录 review coverage；
4. 核对 OCR 每张输入 PNG hash，且上游目录不变；
5. 不把 observation count 当作 accuracy。

### ASR / alignment

1. 使用显式范围、固定 model ID/revision/tree hash 和新输出目录；
2. 核对 mono 16 kHz PCM、source/audio/helper/model provenance；
3. 检查 quality gate、失败 ledger 和合法的人工 anchor；
4. alignment 必须保留 ASR 范围外 frame，不得自动补写或称为语义匹配。

### 完整 workflow / UI

1. range 必须从 `0` 到 probe 得到的 source duration；
2. Apple Vision 和 MLX Metal preflight 成功；
3. 每个视觉 frame 都有 OCR success 或 policy skip；
4. ASR quality、alignment coverage、semantic assignment 和 Markdown coverage 完整；
5. source fingerprint、size、mtime 前后不变；
6. UI 验证 51st rejection、一个 active batch、严格串行、失败后继续、无目录扫描/覆盖，以及 fallback/ledger 授权路径。

## 7. 数据与实验复现

公开仓库不提供真实输入或 benchmark 结论。复现实验必须在仓库外记录：

| 项目 | 必须记录 |
|---|---|
| 源数据 | 合法来源、非敏感标识、SHA-256、size、mtime、duration、codec/FPS |
| 时间边界 | start/end、lookahead、long-range opt-in |
| 算法 | strategy、policy/schema、analysis FPS/width、threshold、max frames |
| OCR | helper hash、平台、语言、ROI、每图 hash |
| ASR | model ID/revision/tree hash、helper/FFmpeg/runtime、decoding profile |
| 输出 | manifest、strategy、OCR、ASR、alignment、semantic、Markdown manifests |
| 人工评估 | annotation schema、review coverage、输入/candidate binding、局限 |

当前证据算法没有随机种子参数；loopback session token 使用安全随机数，但不影响算法结果。
