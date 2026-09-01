# 架构说明

## 1. 定位与规模

FrameLedger 是中型、本地优先的研究型应用：一个 Python package、一个 Swift OCR helper 和一个隔离的 MLX ASR runtime。它不是 Monorepo，也没有数据库、云服务或多租户后端。

架构目标是把金融视频转换为可审计的原始证据，而不是生成金融结论：

1. **输入不可变**：源视频只读，每次处理一个显式文件和显式范围；
2. **产物可绑定**：输入、helper、模型和中间产物以 SHA-256、时间戳和 policy version 绑定；
3. **推断有边界**：OCR/ASR 保留原始输出，时间邻近不升级为语义匹配，失败和未覆盖范围必须显式保留。

## 2. 技术栈

| 层 | 当前实现 | 版本来源 |
|---|---|---|
| Runtime | Python `>=3.12,<3.14` | `pyproject.toml`、`.python-version` |
| 打包 | Hatchling；`uv` 管理主环境 | `pyproject.toml`、`uv.lock` |
| 图像/视频 | NumPy、OpenCV headless、PySceneDetect | 主 manifest 与 `uv.lock` |
| 配置/标注协议 | PyYAML | 主 manifest 与 `uv.lock` |
| OCR | Swift + Apple Vision/CoreGraphics/CoreImage/ImageIO | `phase2/apple_vision_ocr/main.swift` |
| ASR | 独立 Python 3.12 venv + `mlx-whisper==0.4.3` | ASR requirements lock |
| 音频 | 外部 FFmpeg | 系统或显式路径；不在 Python lockfile |
| 本地 UI | Python `ThreadingHTTPServer` + 内嵌 HTML/CSS/JS | `src/frameledger/local_ui.py` |
| 测试 | Python `unittest` | `tests/` |

主 lockfile 当前锁定 NumPy `2.5.2`、PyYAML `6.0.3`、scenedetect-headless `0.7.1` 和 OpenCV headless `5.0.0.93`。ASR 完整传递依赖由独立 requirements lock 固定。Hatchling 没有版本上界，是构建可复现性缺口。

## 3. 数据流

```text
显式源视频（只读）
  │
  ├─ probe / scan
  └─ benchmark（显式范围或 workflow 探测的完整范围）
       │
       ├─ manifest + source fingerprint
       ├─ strategy ledger
       ├─ source-resolution PNG evidence
       └─ offline review
            │
            ├─ OCR（presentation/table；Apple Vision）
            └─ ASR（单独 audio.wav；本地 MLX Whisper）
                 │
                 ▼
              alignment
                 │
                 ▼
              semantic（确定性）
                 │
                 ▼
              markdown report
```

`workflow` 在昂贵视觉阶段前探测完整时长、定位 helper/model 并验证 Apple Vision 和 MLX Metal。它按 `visual → ocr → asr → alignment → semantic → markdown` 串行运行；失败写入机器可读状态并停止当前视频。

`workflow-ui` 只负责编排：绑定 `127.0.0.1`，一个 active batch，最多 50 个显式项目，严格串行；一个项目失败后继续下一个。它不扫描目录。

## 4. 模块边界

| 模块 | 职责 | 不负责 |
|---|---|---|
| `cli.py` | 命令、参数和错误退出 | 算法、隐式扫描 |
| `media.py` | 输入校验、视频探测、帧读取、fingerprint | 转写、语义判断 |
| `features.py`、`selection.py`、`routing.py` | 视觉特征、选择器和固定路由 | OCR/ASR 和金融语义 |
| `pipeline.py` | Phase 1 编排、产物和外部标注评估 | 修改源视频 |
| `annotations.py`、`asr_annotations.py` | 外部 annotation schema 和有限锚点评分 | 未审阅数据的准确率主张 |
| `ocr.py` | helper protocol、OCR ledger 和 review | 单元格关系、值纠错 |
| `asr.py` | PCM WAV、local helper、模型/质量 provenance | 云转写、说话人识别 |
| `alignment.py` | 同源/hash 验证、时间关系和 coverage | 语义匹配 |
| `semantic.py` | 确定性章节和完整事件分配 | LLM 总结、embedding |
| `markdown_export.py` | 完整性复核和原子 Markdown 发布 | 修改图片、生成事实 |
| `workflow.py` | 原生 preflight 和六阶段串行编排 | 批量调度 |
| `local_ui.py` | 会话输出授权、显式队列、配额和 ledger | 远程服务、目录发现、并行处理 |

Swift 和 MLX helper 使用严格 JSON-over-stdio protocol；主 Python 环境不直接依赖 MLX。

## 5. 数据和网络边界

- 公开仓库只跟踪源码、通用测试、helper、依赖清单和治理文档；
- `.gitignore` 排除媒体、音频、模型、`outputs/`、cache、browser import 和 batch ledger；
- 真实 annotation、benchmark evidence 和 freeze archive 不属于公开仓库；
- 新输出目录必须不存在；完整 workflow 拒绝在源视频旁边写入；
- Apple Vision 和 MLX Whisper 本机执行，HTML review 不引用外部 asset；
- 仅依赖安装和预先下载固定模型 revision 可能联网；证据处理本身不得传输媒体或识别文本。

## 6. 环境差异

| 环境 | 实际范围 |
|---|---|
| 本地基础开发 | macOS + Python/uv；纯逻辑、CLI 和 Phase 1 |
| 本地原生完整运行 | Apple Vision、Apple Silicon/MLX、固定模型、FFmpeg、合法真实视频 |
| 自动测试 | 临时目录、合成帧、fake helper、原生不可用时明确 skip |
| CI | macOS arm64 / Python 3.12、3.13；Ubuntu secret/dependency scan |
| 生产 | 不适用；没有生产服务、数据库或云环境 |

## 7. 已知技术债务

1. 公开候选没有私有研究历史、真实 benchmark 或独立有效性数据；
2. 真实 OCR/ASR/selector 回归必须由使用者用合法输入独立完成；
3. CI 尚未在未来目标 GitHub 仓库获得 hosted run 证据；
4. OCR/ASR 完整链依赖 macOS/Apple Silicon；
5. 变量帧率源使用 nominal FPS，不是逐帧 presentation timestamp；
6. table/presentation/router 缺少公开 blind holdout；OCR/ASR 锚点不是整体准确率；
7. Hatchling build backend 未限制版本；ASR lock 固定版本但没有 hash；
8. 已采用 MIT License；PVR 已启用且公开入口可见，但独立非协作者提交/接收仍需人工验证。
