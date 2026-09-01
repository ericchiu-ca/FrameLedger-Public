# 第三方依赖许可证清单

本清单用于公开发布前的依赖治理和版本漂移检查，不是法律意见，也不替代对许可证全文、NOTICE、源码分发义务、模型许可证或素材权利的人工审查。

## 清单口径

- `main` 来自 `uv.lock`，不包含本地 `frameledger` package；
- `ASR` 来自 `phase2/mlx_whisper_asr/requirements.lock.txt`；
- 版本取当前 lockfile；许可证表达式取 2026-09-01 当前已安装 distribution metadata，缺少 SPDX 表达式时使用其 PyPI classifier；
- 本表只记录 Python distributions。FFmpeg、Apple/Xcode runtime、macOS frameworks、Whisper 模型权重、测试媒体和文档素材必须分别确认；
- `python scripts/check_license_inventory.py` 只检查包名、版本、runtime 归属和许可证字段是否随 lockfile 漂移，不能判断许可证兼容性。

## 当前锁定依赖

| Package | Version | Runtime | Metadata license |
|---|---:|---|---|
| `anyio` | `4.14.2` | ASR | `MIT` |
| `certifi` | `2026.7.22` | ASR | `MPL-2.0` |
| `charset-normalizer` | `3.5.1` | ASR | `MIT` |
| `click` | `8.4.2` | main + ASR | `BSD-3-Clause` |
| `colorama` | `0.4.6` | main | `BSD-3-Clause` |
| `filelock` | `3.32.3` | ASR | `MIT` |
| `fsspec` | `2026.7.0` | ASR | `BSD-3-Clause` |
| `h11` | `0.16.0` | ASR | `MIT` |
| `hf-xet` | `1.6.0` | ASR | `Apache-2.0` |
| `httpcore` | `1.0.9` | ASR | `BSD-3-Clause` |
| `httpx` | `0.28.1` | ASR | `BSD-3-Clause` |
| `huggingface-hub` | `1.28.0` | ASR | `Apache-2.0` |
| `idna` | `3.19` | ASR | `BSD-3-Clause` |
| `jinja2` | `3.1.6` | ASR | `BSD-3-Clause` |
| `llvmlite` | `0.49.0` | ASR | `BSD-2-Clause AND Apache-2.0 WITH LLVM-exception` |
| `markupsafe` | `3.0.3` | ASR | `BSD-3-Clause` |
| `mlx` | `0.32.1` | ASR | `MIT` |
| `mlx-metal` | `0.32.1` | ASR | `MIT` |
| `mlx-whisper` | `0.4.3` | ASR | `MIT` |
| `more-itertools` | `11.1.0` | ASR | `MIT` |
| `mpmath` | `1.3.0` | ASR | `BSD` classifier |
| `networkx` | `3.6.1` | ASR | `BSD-3-Clause` |
| `numba` | `0.67.0` | ASR | `BSD` classifier |
| `numpy` | `2.5.2` | main + ASR | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` |
| `opencv-python-headless` | `5.0.0.93` | main | `Apache-2.0` |
| `packaging` | `26.3` | ASR | `Apache-2.0 OR BSD-2-Clause` |
| `platformdirs` | `4.11.3` | main | `MIT` |
| `pyyaml` | `6.0.3` | main + ASR | `MIT` |
| `regex` | `2026.7.19` | ASR | `Apache-2.0 AND CNRI-Python` |
| `requests` | `2.34.2` | ASR | `Apache-2.0` |
| `scenedetect-headless` | `0.7.1` | main | `BSD-3-Clause` |
| `scipy` | `1.18.1` | ASR | `BSD` classifier |
| `setuptools` | `84.0.0` | ASR | `MIT` |
| `sympy` | `1.14.0` | ASR | `BSD` classifier |
| `tiktoken` | `0.14.0` | ASR | `MIT` |
| `torch` | `2.13.0` | ASR | `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` |
| `tqdm` | `4.70.0` | main + ASR | `MPL-2.0 AND MIT` |
| `typing-extensions` | `4.16.0` | ASR | `PSF-2.0` |
| `urllib3` | `2.7.0` | ASR | `MIT` |

## 发布前人工复核

1. 从每个 distribution 的实际 `METADATA`、许可证文件和上游源码复核表达式；不能只依赖 classifier；
2. 对多许可证表达式确认实际适用文件、组合方式和分发义务；
3. 确认 wheel/sdist 是否需要包含第三方 NOTICE 或许可证副本；
4. 单独确认 `mlx-community/whisper-large-v3-turbo` 模型卡、权重和上游训练数据条款；
5. 单独确认 FFmpeg build 的编译选项与许可证；
6. FrameLedger 自身采用 MIT License；公开分发前仍需做一次完整兼容性法律复核。
