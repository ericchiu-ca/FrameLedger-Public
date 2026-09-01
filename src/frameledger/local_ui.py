from __future__ import annotations

import hashlib
import html
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import unicodedata
import urllib.parse
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from .media import MediaError, SUPPORTED_EXTENSIONS, probe_video, validate_video_path
from .workflow import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    default_asr_helper,
    default_model_path,
    project_root,
    run_complete_workflow,
)


RUN_NAME_PATTERN = re.compile(r"[\w][\w.-]{0,79}\Z", re.UNICODE)
MAX_JSON_BODY_BYTES = 256 * 1024
MAX_BATCH_ITEMS = 50
MAX_UPLOAD_BYTES = 32 * 1024 * 1024 * 1024
MAX_SESSION_IMPORT_BYTES = 64 * 1024 * 1024 * 1024
MIN_FREE_AFTER_IMPORT_BYTES = 8 * 1024 * 1024 * 1024


def _validate_publication_safe_output_root(root: Path) -> None:
    project = project_root().resolve()
    project_outputs = (project / "outputs").resolve()
    if root.is_relative_to(project) and not root.is_relative_to(project_outputs):
        raise ValueError(
            "项目目录内只能选择已被 Git 忽略的 outputs 文件夹或其子目录"
        )


def _macos_picker(script: str) -> list[str]:
    if sys.platform != "darwin":
        raise RuntimeError("本地文件选择器目前仅支持 macOS")
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("macOS 文件选择器等待超时") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if "(-128)" in detail or "User canceled" in detail:
            return []
        raise RuntimeError(f"无法打开 macOS 文件选择器：{detail}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _choose_local_videos() -> list[str]:
    return _macos_picker(
        'set selectedFiles to choose file with prompt "选择要批量处理的视频" '
        "with multiple selections allowed\n"
        'set outputText to ""\n'
        "repeat with selectedFile in selectedFiles\n"
        "set outputText to outputText & POSIX path of selectedFile & linefeed\n"
        "end repeat\n"
        "return outputText"
    )


def _choose_output_directory() -> str | None:
    values = _macos_picker(
        'set selectedFolder to choose folder with prompt "选择 FrameLedger 输出文件夹"\n'
        "return POSIX path of selectedFolder"
    )
    return values[0] if values else None


def _safe_run_name(source: Path) -> str:
    normalized = unicodedata.normalize("NFKC", source.stem)
    value = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip("._-")
    value = value[:72].rstrip("._-")
    return value or "video"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _frontend_html(
    *,
    token: str,
    default_video: str | None,
    output_root: str = "",
    output_choice_id: str = "default",
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
) -> str:
    return f"""<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FrameLedger 本地完整工作流</title>
<style>
:root {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#16212b; background:#edf2f6; }}
body {{ margin:0; }} main {{ max-width:980px; margin:36px auto; padding:0 18px; }}
.card {{ background:#fff; border:1px solid #d4dde5; border-radius:14px; padding:24px; box-shadow:0 8px 28px #24334412; }}
h1 {{ margin:0 0 8px; }} p {{ color:#536270; line-height:1.55; }} label {{ display:block; margin:16px 0 6px; font-weight:650; }}
input {{ box-sizing:border-box; width:100%; padding:11px 12px; border:1px solid #b9c5cf; border-radius:8px; font:inherit; }}
.actions {{ display:flex; gap:10px; margin-top:18px; flex-wrap:wrap; }} button,a.primary {{ border:0; border-radius:8px; padding:11px 15px; background:#08775e; color:#fff; font:inherit; cursor:pointer; text-decoration:none; }}
button.secondary {{ background:#e8eef3; color:#253645; }} button:disabled {{ opacity:.55; cursor:not-allowed; }}
#message {{ margin-top:18px; white-space:pre-wrap; }} .error {{ color:#8a2525; }} .ok {{ color:#087059; }}
ol,ul {{ padding-left:24px; }} li {{ margin:9px 0; }} code {{ font-size:11px; overflow-wrap:anywhere; }}
.notice {{ padding:11px 13px; border-left:4px solid #08775e; background:#eaf7f3; }}
.drop {{ margin-top:18px; padding:28px 18px; border:2px dashed #91a5b4; border-radius:12px; text-align:center; color:#425666; background:#f7fafc; cursor:pointer; }}
.drop.dragging {{ border-color:#08775e; background:#eaf7f3; }} .drop:focus {{ outline:3px solid #74cbb6; outline-offset:2px; }}
.queue {{ margin:18px 0 0; padding:0; list-style:none; }} .queue li {{ padding:12px 14px; border:1px solid #d9e1e7; border-radius:9px; background:#fafcfd; }}
.queue strong,.queue span {{ display:block; }} .queue span {{ color:#62717d; font-size:13px; margin-top:4px; overflow-wrap:anywhere; }}
.queue a {{ display:inline-block; margin-top:8px; color:#08775e; }} .hidden {{ display:none; }}
</style></head><body><main id="app" data-token="{html.escape(token, quote=True)}" data-video="{html.escape(default_video or "", quote=True)}" data-output-choice="{html.escape(output_choice_id, quote=True)}"><section class="card">
<h1>FrameLedger 本地完整工作流</h1>
<p class="notice">显式选择或拖入多个视频后，工具会逐片串行完成视觉、OCR、本机 Whisper、对齐、语义分段和 Markdown 证据导出。所有处理仅在本机进行；无需聊天模型，也不会连接云端。</p>
<div id="dropZone" class="drop" role="button" tabindex="0" aria-label="拖入或选择多个视频">把一个或多个视频拖到这里<br><small>浏览器无法提供原路径时，会流式复制到本机受管导入区；原视频不变。较大批次建议使用下方“选择视频”，可避免复制。</small></div>
<input id="browserFiles" class="hidden" type="file" multiple accept=".mp4,.mov,.mkv,.m4v,.webm,video/*">
<div class="actions"><button id="chooseVideos" class="secondary">选择视频</button><button id="clearQueue" class="secondary">清空列表</button></div>
<ul id="queue" class="queue" aria-live="polite"></ul>
<label for="outputPath">输出文件夹</label>
<input id="outputPath" readonly value="{html.escape(output_root, quote=True)}">
<div class="actions"><button id="chooseOutput" class="secondary">选择输出文件夹</button><button id="run" disabled>批量生成</button></div>
<div id="message"></div><ol id="stages"></ol><div id="result"></div>
<p>固定本地模型：<code>{html.escape(model_id)}</code><br>revision：<code>{html.escape(model_revision)}</code></p>
</section></main><script src="/app.js" defer></script></body></html>"""


def _frontend_js() -> str:
    return r"""
const app = document.getElementById('app');
const token = app.dataset.token;
const message = document.getElementById('message'); const stages = document.getElementById('stages');
const result = document.getElementById('result'); const runButton = document.getElementById('run');
const queue = document.getElementById('queue'); const dropZone = document.getElementById('dropZone');
const browserFiles = document.getElementById('browserFiles'); const outputPath = document.getElementById('outputPath');
let outputChoice = app.dataset.outputChoice; let entries = []; let busy = false;
const droppedFileKeys = new Set();
const allowedExtensions = new Set(['.mp4','.mov','.mkv','.m4v','.webm']);
const labels = {visual:'视觉路由与关键帧',ocr:'本地 OCR',asr:'整片本地语音识别',alignment:'视觉与语音对齐',semantic:'本地语义分段',markdown:'Markdown 关键截图'};
async function api(path, body) {
  const response = await fetch(path, {method: body ? 'POST':'GET', headers: body ? {'Content-Type':'application/json','X-FrameLedger-Token':token} : {}, body: body ? JSON.stringify(body) : undefined});
  const payload = await response.json(); if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`); return payload;
}
function setMessage(text, kind='') { message.className=kind; message.textContent=text; }
function updateButtons() { runButton.disabled=busy || entries.length===0 || !outputChoice; document.getElementById('clearQueue').disabled=busy || entries.length===0; }
function basename(path) { const parts=path.split('/'); return parts[parts.length-1] || path; }
function renderQueue(job=null) {
  queue.replaceChildren();
  const sourceItems = job ? job.items : entries;
  sourceItems.forEach((item,index) => {
    const li=document.createElement('li'); const title=document.createElement('strong');
    title.textContent=`${index+1}. ${item.display_name || basename(item.video_path)}`; li.appendChild(title);
    const detail=document.createElement('span');
    if (job) detail.textContent=`${item.status}${item.current_stage ? ' · '+labels[item.current_stage] : ''}${item.error ? ' · '+item.error : ''}${item.output_path ? ' · '+item.output_path : ''}`;
    else detail.textContent=item.note || item.video_path;
    li.appendChild(detail);
    if (job && item.status==='ok' && item.report_url) { const link=document.createElement('a'); link.href=item.report_url; link.download='report.md'; link.textContent='下载关键截图 Markdown'; li.appendChild(link); }
    queue.appendChild(li);
  });
  updateButtons();
}
function addPaths(paths,note='直接读取原文件（只读）') {
  const known=new Set(entries.map(item=>item.video_path));
  let added=0; let skipped=0;
  paths.forEach(path => { if (!path || known.has(path)) return; if (entries.length>=50) { skipped+=1; return; } entries.push({video_path:path,display_name:basename(path),note}); known.add(path); added+=1; });
  renderQueue(); return {added,skipped};
}
function uploadFile(file) {
  return new Promise((resolve,reject) => {
    const xhr=new XMLHttpRequest(); xhr.open('POST',`/api/import?filename=${encodeURIComponent(file.name)}&output_choice_id=${encodeURIComponent(outputChoice)}`);
    xhr.setRequestHeader('X-FrameLedger-Token',token); xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.upload.onprogress=event => { if (event.lengthComputable) setMessage(`正在导入 ${file.name}：${Math.round(event.loaded/event.total*100)}%`); };
    xhr.onerror=()=>reject(new Error(`无法导入 ${file.name}`));
    xhr.onload=()=>{ let payload={}; try { payload=JSON.parse(xhr.responseText); } catch (_) {} if (xhr.status<200||xhr.status>=300) reject(new Error(payload.error||`HTTP ${xhr.status}`)); else resolve(payload); };
    xhr.send(file);
  });
}
async function addDroppedFiles(files) {
  if (!files.length || busy) return; busy=true; updateButtons();
  try {
    if (entries.length+files.length>50) throw new Error('每个批次最多只能加入 50 个显式视频');
    for (const file of files) {
      const extension=file.name.includes('.') ? `.${file.name.split('.').pop().toLowerCase()}` : '';
      if (!allowedExtensions.has(extension)) throw new Error(`不支持的视频格式：${file.name}`);
      const fileKey=`${file.name}\u0000${file.size}\u0000${file.lastModified}`;
      if (droppedFileKeys.has(fileKey)) continue;
      const directPath=typeof file.path==='string' && file.path.startsWith('/') ? file.path : '';
      if (directPath) addPaths([directPath]);
      else { const imported=await uploadFile(file); addPaths([imported.video_path],'浏览器拖入 · 已复制到本机受管导入区'); }
      droppedFileKeys.add(fileKey);
    }
    setMessage(`已加入 ${entries.length} 个视频`,'ok');
  } catch(error) { setMessage(error.message,'error'); }
  finally { busy=false; updateButtons(); }
}
function renderStatus(job) {
  const summary=job.summary || {};
  const counts=job.total_items ? ` · ${summary.ok||0} 成功 / ${summary.error||0} 失败 / ${summary.pending||0} 等待` : '';
  setMessage(`批次状态：${job.status}${job.current_item_index ? ` · 第 ${job.current_item_index}/${job.total_items} 个` : ''}${counts}${job.error ? '\n'+job.error : ''}`,job.status==='error'||job.status==='partial'?'error':job.status==='ok'?'ok':'');
  renderQueue(job); stages.replaceChildren(); const state = job.stages || {};
  Object.keys(labels).forEach(key => { const li=document.createElement('li'); li.textContent=`${labels[key]}：${(state[key]||{}).status||'pending'}`; stages.appendChild(li); });
  if (['ok','partial','error'].includes(job.status)) { busy=false; updateButtons(); }
}
async function poll(id) { try { const job=await api(`/api/status?job_id=${encodeURIComponent(id)}`); renderStatus(job); if (job.status==='running'||job.status==='queued') setTimeout(()=>poll(id),1200); } catch(error) { busy=false; setMessage(error.message,'error'); updateButtons(); } }
document.getElementById('chooseVideos').onclick=async()=>{ if (busy) return; try { const data=await api('/api/choose-videos',{}); const outcome=addPaths(data.video_paths||[]); if (outcome.skipped) setMessage(`已保留前 50 个视频，另有 ${outcome.skipped} 个未加入`,'error'); else if (outcome.added) setMessage(`已加入 ${entries.length} 个视频`,'ok'); } catch(error) { setMessage(error.message,'error'); } };
document.getElementById('chooseOutput').onclick=async()=>{ if (busy) return; try { const data=await api('/api/choose-output',{}); if (!data.cancelled) { outputChoice=data.choice_id; outputPath.value=data.path; updateButtons(); } } catch(error) { setMessage(error.message,'error'); } };
document.getElementById('clearQueue').onclick=()=>{ if (!busy) { entries=[]; droppedFileKeys.clear(); result.replaceChildren(); stages.replaceChildren(); setMessage(''); renderQueue(); } };
dropZone.onclick=()=>{ if (!busy) browserFiles.click(); }; dropZone.onkeydown=event=>{ if (event.key==='Enter'||event.key===' ') { event.preventDefault(); dropZone.click(); } };
browserFiles.onchange=async()=>{ await addDroppedFiles(Array.from(browserFiles.files||[])); browserFiles.value=''; };
['dragenter','dragover'].forEach(name=>dropZone.addEventListener(name,event=>{ event.preventDefault(); if (!busy) dropZone.classList.add('dragging'); }));
['dragleave','drop'].forEach(name=>dropZone.addEventListener(name,event=>{ event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop',event=>addDroppedFiles(Array.from(event.dataTransfer.files||[])));
runButton.onclick=async()=>{ try { busy=true; updateButtons(); result.replaceChildren(); const job=await api('/api/run-batch',{video_paths:entries.map(item=>item.video_path),output_choice_id:outputChoice}); renderStatus(job); poll(job.job_id); } catch(error) { busy=false; setMessage(error.message,'error'); updateButtons(); } };
if (app.dataset.video) addPaths([app.dataset.video]); else renderQueue();
api('/api/active').then(data=>{ if (data.job) { busy=true; renderStatus(data.job); poll(data.job.job_id); } }).catch(error=>setMessage(error.message,'error'));
"""


class WorkflowJobManager:
    def __init__(
        self, *, output_root: Path, workflow_options: Mapping[str, Any]
    ) -> None:
        self.output_root = output_root.resolve()
        _validate_publication_safe_output_root(self.output_root)
        self.default_output_choice_id = "default"
        self.output_choices: dict[str, Path] = {
            self.default_output_choice_id: self.output_root
        }
        self.import_root = self.output_root / ".frameledger-imports"
        self.workflow_options = dict(workflow_options)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.active_job_id: str | None = None
        self.session_import_bytes = 0
        self.reserved_import_bytes = 0
        self.session_import_items = 0
        self.reserved_import_items = 0

    def _ensure_inactive_locked(self) -> None:
        if self.active_job_id is None:
            return
        active = self.jobs.get(self.active_job_id)
        if active is not None and active.get("status") in {"queued", "running"}:
            raise RuntimeError("已有一个本地批次正在运行")

    def register_output_root(self, value: str | Path) -> dict[str, str]:
        root = Path(value).expanduser().resolve()
        if root == Path(root.anchor):
            raise ValueError("不能把文件系统根目录作为输出文件夹")
        if not root.is_dir():
            raise ValueError(f"输出文件夹不存在：{root}")
        if not os.access(root, os.W_OK | os.X_OK):
            raise ValueError(f"输出文件夹不可写：{root}")
        _validate_publication_safe_output_root(root)
        frozen = (project_root() / "freezes" / "phase1-v1").resolve()
        if root == frozen or frozen in root.parents:
            raise ValueError("输出文件夹不能位于冻结的 Phase 1 基线内")
        with self.lock:
            for choice_id, existing in self.output_choices.items():
                if existing == root:
                    return {"choice_id": choice_id, "path": str(root)}
            choice_id = secrets.token_urlsafe(18)
            self.output_choices[choice_id] = root
        return {"choice_id": choice_id, "path": str(root)}

    def output_choice(self, choice_id: str) -> Path:
        with self.lock:
            root = self.output_choices.get(choice_id)
        if root is None:
            raise ValueError("输出文件夹选择已失效，请重新选择")
        return root

    def save_import(
        self,
        *,
        filename: str,
        stream: BinaryIO,
        content_length: int,
        output_choice_id: str | None = None,
    ) -> dict[str, Any]:
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            raise ValueError("拖入的视频大小无效或超过本地导入上限")
        if not filename or Path(filename).name != filename:
            raise ValueError("拖入的视频文件名无效")
        if any(ord(character) < 32 for character in filename):
            raise ValueError("拖入的视频文件名包含控制字符")
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"不支持的视频扩展名；应为 {allowed}")
        choice_id = output_choice_id or self.default_output_choice_id
        import_root = self.output_choice(choice_id) / ".frameledger-imports"
        free_bytes = shutil.disk_usage(import_root.parent).free
        with self.lock:
            if self.session_import_items + self.reserved_import_items >= MAX_BATCH_ITEMS:
                raise ValueError(
                    "本次服务会话最多保留 50 个浏览器拖入视频；更多视频请使用原生选择按钮"
                )
            if (
                self.session_import_bytes
                + self.reserved_import_bytes
                + content_length
                > MAX_SESSION_IMPORT_BYTES
            ):
                raise ValueError(
                    "本次服务会话的浏览器拖入总量不能超过 64 GiB；请改用原生选择按钮"
                )
            if free_bytes - self.reserved_import_bytes - content_length < MIN_FREE_AFTER_IMPORT_BYTES:
                raise ValueError("磁盘剩余空间不足；导入后必须至少保留 8 GiB")
            self.reserved_import_bytes += content_length
            self.reserved_import_items += 1
        import_directory = import_root / secrets.token_hex(12)
        target = import_directory / filename
        temporary = import_directory / f".{filename}.uploading"
        remaining = content_length
        succeeded = False
        try:
            import_directory.mkdir(parents=True, exist_ok=False)
            with temporary.open("xb") as handle:
                while remaining:
                    chunk = stream.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("视频导入中断，未保留不完整文件")
                    handle.write(chunk)
                    remaining -= len(chunk)
            temporary.replace(target)
            validate_video_path(target)
            probe_video(target)
            succeeded = True
        except Exception:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            try:
                import_directory.rmdir()
            except OSError:
                pass
            raise
        finally:
            with self.lock:
                self.reserved_import_bytes -= content_length
                self.reserved_import_items -= 1
                if succeeded:
                    self.session_import_bytes += content_length
                    self.session_import_items += 1
        return {
            "video_path": str(target.resolve()),
            "display_name": filename,
            "size_bytes": content_length,
            "storage": "managed_local_import",
        }

    @staticmethod
    def _summary(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        ok = sum(1 for item in items if item.get("status") == "ok")
        error = sum(1 for item in items if item.get("status") == "error")
        pending = len(items) - ok - error
        return {"ok": ok, "error": error, "pending": pending}

    @staticmethod
    def _write_batch_snapshot(job: Mapping[str, Any]) -> None:
        root = Path(str(job["output_root"]))
        ledger_root = root / ".frameledger-batches"
        ledger_root.mkdir(parents=True, exist_ok=True)
        target = ledger_root / f"{job['job_id']}.json"
        temporary = ledger_root / f".{job['job_id']}.{secrets.token_hex(6)}.tmp"
        temporary.write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)

    def _snapshot_locked(self, job: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(job))

    def _unique_run_names(
        self,
        *,
        sources: Sequence[Path],
        output_root: Path,
        explicit_names: Sequence[str] | None,
    ) -> list[str]:
        reserved: set[str] = set()
        names: list[str] = []
        for index, source in enumerate(sources):
            if explicit_names is not None:
                candidate = explicit_names[index]
                if not RUN_NAME_PATTERN.fullmatch(candidate):
                    raise ValueError(
                        "运行名称只能包含 Unicode 字母、数字、点、下划线和连字符，最长 80 个字符"
                    )
                if candidate.casefold() in reserved:
                    raise ValueError(f"批次内运行名称重复：{candidate}")
                if (output_root / candidate).exists():
                    raise ValueError(f"运行输出已经存在，拒绝覆盖：{output_root / candidate}")
            else:
                base = _safe_run_name(source)
                candidate = base
                suffix = 2
                while (
                    candidate.casefold() in reserved
                    or (output_root / candidate).exists()
                ):
                    marker = f"-{suffix}"
                    candidate = f"{base[: 80 - len(marker)]}{marker}"
                    suffix += 1
            reserved.add(candidate.casefold())
            names.append(candidate)
        return names

    def start_batch(
        self,
        *,
        video_paths: Sequence[str],
        output_choice_id: str,
        explicit_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self._ensure_inactive_locked()
        if not video_paths or len(video_paths) > MAX_BATCH_ITEMS:
            raise ValueError(f"每个批次必须包含 1–{MAX_BATCH_ITEMS} 个显式视频")
        if explicit_names is not None and len(explicit_names) != len(video_paths):
            raise ValueError("视频和运行名称数量不一致")
        sources: list[Path] = []
        seen: set[Path] = set()
        for value in video_paths:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("每个视频路径都必须是非空字符串")
            expanded = Path(value).expanduser()
            if not expanded.is_absolute():
                raise ValueError(f"视频路径必须是绝对路径：{value}")
            source = validate_video_path(expanded)
            if source in seen:
                raise ValueError(f"同一源视频不能在一个批次中重复：{source}")
            seen.add(source)
            sources.append(source)
        output_root = self.output_choice(output_choice_id)
        names = self._unique_run_names(
            sources=sources,
            output_root=output_root,
            explicit_names=explicit_names,
        )
        frozen = (project_root() / "freezes" / "phase1-v1").resolve()
        prepared: list[dict[str, Any]] = []
        for source, run_name in zip(sources, names, strict=True):
            output = (output_root / run_name).resolve()
            if not output.is_relative_to(output_root):
                raise ValueError("运行输出逃离了选择的输出文件夹")
            source_parent = source.parent.resolve()
            if output == source_parent or source_parent in output.parents:
                raise ValueError(
                    f"输出不能写在源视频旁边或其下方：{source.name}"
                )
            if output == frozen or frozen in output.parents:
                raise ValueError("运行输出不能修改冻结的 Phase 1 基线")
            prepared.append(
                {
                    "video_path": str(source),
                    "display_name": source.name,
                    "run_name": run_name,
                    "output_path": str(output),
                    "status": "queued",
                    "queued_at": _now(),
                    "started_at": None,
                    "completed_at": None,
                    "current_stage": None,
                    "stages": {},
                    "error": None,
                    "report_url": None,
                }
            )

        with self.lock:
            self._ensure_inactive_locked()
            job_id = secrets.token_hex(12)
            for index, item in enumerate(prepared):
                item["report_url"] = (
                    f"/artifacts/{job_id}/{index}/markdown/report.md"
                )
            job: dict[str, Any] = {
                "kind": "workflow_batch",
                "job_id": job_id,
                "status": "queued",
                "created_at": _now(),
                "completed_at": None,
                "output_root": str(output_root),
                "current_item_index": None,
                "total_items": len(prepared),
                "current_stage": None,
                "stages": {},
                "error": None,
                "items": prepared,
                "summary": self._summary(prepared),
                "batch_manifest": str(
                    output_root / ".frameledger-batches" / f"{job_id}.json"
                ),
            }
            if len(prepared) == 1:
                job["run_name"] = prepared[0]["run_name"]
            snapshot = self._snapshot_locked(job)
            self._write_batch_snapshot(snapshot)
            self.jobs[job_id] = job
            self.active_job_id = job_id

        def persist_snapshot(value: Mapping[str, Any]) -> None:
            try:
                self._write_batch_snapshot(value)
            except OSError as error:
                with self.lock:
                    job["ledger_error"] = f"{type(error).__name__}: {error}"

        def worker() -> None:
            unexpected_error: str | None = None
            try:
                with self.lock:
                    job["status"] = "running"
                    first_snapshot = self._snapshot_locked(job)
                persist_snapshot(first_snapshot)
                for index, item in enumerate(prepared):
                    with self.lock:
                        job["current_item_index"] = index + 1
                        item["status"] = "running"
                        item["started_at"] = _now()
                        job["summary"] = self._summary(prepared)
                        item_snapshot = self._snapshot_locked(job)
                    persist_snapshot(item_snapshot)

                    def update(progress: Mapping[str, Any]) -> None:
                        with self.lock:
                            item["current_stage"] = progress.get("current_stage")
                            item["stages"] = json.loads(
                                json.dumps(progress.get("stages") or {})
                            )
                            job["current_stage"] = item["current_stage"]
                            job["stages"] = item["stages"]
                            progress_snapshot = self._snapshot_locked(job)
                        persist_snapshot(progress_snapshot)

                    try:
                        run_complete_workflow(
                            Path(item["video_path"]),
                            output=Path(item["output_path"]),
                            progress_callback=update,
                            **self.workflow_options,
                        )
                        with self.lock:
                            item["status"] = "ok"
                            item["current_stage"] = None
                            item["error"] = None
                            item["completed_at"] = _now()
                    except Exception as error:
                        with self.lock:
                            item["status"] = "error"
                            item["current_stage"] = None
                            item["error"] = f"{type(error).__name__}: {error}"
                            item["completed_at"] = _now()
                    with self.lock:
                        job["summary"] = self._summary(prepared)
                        completed_snapshot = self._snapshot_locked(job)
                    persist_snapshot(completed_snapshot)
            except BaseException as error:
                unexpected_error = f"{type(error).__name__}: {error}"
            finally:
                with self.lock:
                    if unexpected_error is not None:
                        for item in prepared:
                            if item["status"] in {"queued", "running"}:
                                item["status"] = "error"
                                item["current_stage"] = None
                                item["error"] = (
                                    "批次执行器意外中断：" + unexpected_error
                                )
                                item["completed_at"] = _now()
                    summary = self._summary(prepared)
                    job["summary"] = summary
                    job["current_item_index"] = None
                    job["current_stage"] = None
                    job["completed_at"] = _now()
                    if summary["error"] == 0:
                        job["status"] = "ok"
                        job["error"] = None
                    elif summary["ok"] == 0:
                        job["status"] = "error"
                        job["error"] = (
                            "批次中的所有视频均处理失败"
                            if unexpected_error is None
                            else "批次执行器意外中断"
                        )
                    else:
                        job["status"] = "partial"
                        job["error"] = "部分视频处理失败，其余结果已保留"
                    if self.active_job_id == job_id:
                        self.active_job_id = None
                    final_snapshot = self._snapshot_locked(job)
                persist_snapshot(final_snapshot)

        thread = threading.Thread(
            target=worker, name=f"frameledger-batch-{job_id}", daemon=True
        )
        try:
            thread.start()
        except BaseException as error:
            with self.lock:
                job["status"] = "error"
                job["error"] = f"无法启动批次线程：{type(error).__name__}: {error}"
                for item in prepared:
                    item["status"] = "error"
                    item["error"] = job["error"]
                    item["completed_at"] = _now()
                job["summary"] = self._summary(prepared)
                job["completed_at"] = _now()
                if self.active_job_id == job_id:
                    self.active_job_id = None
                failed_snapshot = self._snapshot_locked(job)
            persist_snapshot(failed_snapshot)
            raise
        return snapshot

    def start(self, *, video_path: str, run_name: str) -> dict[str, Any]:
        return self.start_batch(
            video_paths=[video_path],
            output_choice_id=self.default_output_choice_id,
            explicit_names=[run_name],
        )

    def status(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self._snapshot_locked(job)

    def active_status(self) -> dict[str, Any] | None:
        with self.lock:
            if self.active_job_id is None:
                return None
            job = self.jobs.get(self.active_job_id)
            return self._snapshot_locked(job) if job is not None else None

    def artifact_path(self, job_id: str, item_index: int, relative: Path) -> Path:
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("结果路径无效")
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            items = job.get("items") or []
            if item_index < 0 or item_index >= len(items):
                raise KeyError(item_index)
            job_root = Path(str(job["output_root"])).resolve()
            output_path = Path(str(items[item_index]["output_path"]))
            if output_path.is_symlink():
                raise FileNotFoundError(output_path)
            output = output_path.resolve()
        if not output.is_relative_to(job_root):
            raise FileNotFoundError(output)
        target = (output / relative).resolve()
        if not target.is_relative_to(output) or not target.is_file():
            raise FileNotFoundError(target)
        return target


class _WorkflowServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler,
        *,
        token,
        manager,
        default_video,
        model_id,
        model_revision,
        bootstrap_token=None,
    ):
        super().__init__(server_address, handler)
        self.token = token
        self.bootstrap_token = bootstrap_token or secrets.token_urlsafe(32)
        self.bootstrap_consumed = False
        self.bootstrap_lock = threading.Lock()
        self.manager = manager
        self.default_video = default_video
        self.model_id = model_id
        self.model_revision = model_revision
        self.verified_sources: dict[str, tuple[int, int, str]] = {}

    def consume_bootstrap(self, path: str) -> bool:
        expected = f"/session/{self.bootstrap_token}"
        with self.bootstrap_lock:
            if self.bootstrap_consumed or not secrets.compare_digest(path, expected):
                return False
            self.bootstrap_consumed = True
            return True


class WorkflowRequestHandler(BaseHTTPRequestHandler):
    server: _WorkflowServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _host_allowed(self) -> bool:
        try:
            host = urllib.parse.urlsplit(
                f"//{self.headers.get('Host', '')}"
            ).hostname
        except ValueError:
            return False
        return (host or "").lower() in {"127.0.0.1", "localhost", "::1"}

    def _token_allowed(self) -> bool:
        header = self.headers.get("X-FrameLedger-Token")
        if isinstance(header, str) and secrets.compare_digest(
            header, self.server.token
        ):
            return True
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except ValueError:
            return False
        value = cookie.get("frameledger_token")
        return value is not None and secrets.compare_digest(
            value.value, self.server.token
        )

    def _json(
        self, status: int, payload: Mapping[str, Any], *, cookie: bool = False
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        if cookie:
            self.send_header(
                "Set-Cookie",
                f"frameledger_token={self.server.token}; HttpOnly; SameSite=Strict; Path=/",
            )
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "media-src 'self'; object-src 'none'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'none'",
        )

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type.lower() != "application/json":
            raise ValueError("JSON 接口要求 Content-Type: application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length 无效") from error
        if length <= 0 or length > MAX_JSON_BODY_BYTES:
            raise ValueError("JSON 请求大小无效")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("请求不是有效 JSON") from error
        if not isinstance(value, dict):
            raise ValueError("JSON 请求必须是对象")
        return value

    def _post_authorized(self) -> bool:
        header = self.headers.get("X-FrameLedger-Token")
        if (
            not self._host_allowed()
            or not isinstance(header, str)
            or not secrets.compare_digest(header, self.server.token)
        ):
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed = urllib.parse.urlparse(origin)
                allowed = (
                    parsed.scheme == "http"
                    and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                    and parsed.port == self.server.server_address[1]
                )
            except ValueError:
                allowed = False
            if not allowed:
                return False
        return True

    def _run_document(self, run_name: str) -> dict[str, Any]:
        if not RUN_NAME_PATTERN.fullmatch(run_name):
            raise ValueError("结果运行名称无效")
        path = (self.server.manager.output_root / run_name / "workflow.json").resolve()
        if (
            not path.is_relative_to(self.server.manager.output_root)
            or not path.is_file()
        ):
            raise FileNotFoundError(run_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "complete_local_workflow"
        ):
            raise ValueError("结果不是完整工作流")
        return payload

    def _source_for_run(self, run_name: str) -> Path:
        document = self._run_document(run_name)
        source = document.get("source")
        if not isinstance(source, dict):
            raise ValueError("结果缺少源视频绑定")
        path = Path(str(source.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if stat.st_size != int(source.get("size_bytes", -1)) or stat.st_mtime_ns != int(
            source.get("mtime_ns", -1)
        ):
            raise ValueError("源视频在工作流完成后发生变化")
        expected = str(source.get("fingerprint", ""))
        cache_key = str(path)
        cached = self.server.verified_sources.get(cache_key)
        identity = (stat.st_size, stat.st_mtime_ns, expected)
        if cached != identity:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("源视频 SHA-256 不再匹配工作流证据")
            self.server.verified_sources[cache_key] = identity
        return path

    def _send_bytes(self, content: bytes, *, mime: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_file(self, path: Path, *, mime: str) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self._security_headers()
                self.end_headers()
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = int(last) if last else end
            elif last:
                suffix = int(last)
                start = max(0, size - suffix)
            if start < 0 or end < start or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self._security_headers()
                self.end_headers()
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_POST(self) -> None:
        if not self._post_authorized():
            self._error(HTTPStatus.FORBIDDEN, "本地请求授权失败")
            return
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/import":
                content_type = (
                    self.headers.get("Content-Type", "").split(";", 1)[0].strip()
                )
                if content_type.lower() != "application/octet-stream":
                    raise ValueError(
                        "视频导入要求 Content-Type: application/octet-stream"
                    )
                query = urllib.parse.parse_qs(parsed.query)
                filename = query.get("filename", [""])[0]
                output_choice_id = query.get("output_choice_id", [""])[0]
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError as error:
                    raise ValueError("Content-Length 无效") from error
                imported = self.server.manager.save_import(
                    filename=filename,
                    stream=self.rfile,
                    content_length=content_length,
                    output_choice_id=output_choice_id,
                )
                self._json(HTTPStatus.CREATED, imported)
                return
            payload = self._read_json()
            if parsed.path == "/api/choose-videos":
                values = _choose_local_videos()
                paths = [str(validate_video_path(Path(value))) for value in values]
                self._json(HTTPStatus.OK, {"video_paths": paths})
                return
            if parsed.path == "/api/choose-output":
                value = _choose_output_directory()
                if value is None:
                    self._json(HTTPStatus.OK, {"cancelled": True})
                else:
                    choice = self.server.manager.register_output_root(value)
                    self._json(HTTPStatus.OK, {"cancelled": False, **choice})
                return
            if parsed.path == "/api/probe":
                video_path = payload.get("video_path")
                if not isinstance(video_path, str) or not video_path.strip():
                    raise ValueError("请提供源视频绝对路径")
                metadata = probe_video(
                    Path(video_path).expanduser().resolve()
                ).to_dict()
                metadata["duration_timecode"] = _format_duration(
                    metadata["duration_seconds"]
                )
                self._json(HTTPStatus.OK, metadata)
                return
            if self.path == "/api/run":
                video_path = payload.get("video_path")
                run_name = payload.get("run_name")
                if not isinstance(video_path, str) or not isinstance(run_name, str):
                    raise ValueError("video_path 和 run_name 必须是字符串")
                job = self.server.manager.start(
                    video_path=video_path, run_name=run_name
                )
                self._json(HTTPStatus.ACCEPTED, job)
                return
            if parsed.path == "/api/run-batch":
                video_paths = payload.get("video_paths")
                output_choice_id = payload.get("output_choice_id")
                if not isinstance(video_paths, list) or not all(
                    isinstance(value, str) for value in video_paths
                ):
                    raise ValueError("video_paths 必须是显式视频路径数组")
                if not isinstance(output_choice_id, str):
                    raise ValueError("请先选择输出文件夹")
                job = self.server.manager.start_batch(
                    video_paths=video_paths,
                    output_choice_id=output_choice_id,
                )
                self._json(HTTPStatus.ACCEPTED, job)
                return
            self._error(HTTPStatus.NOT_FOUND, "接口不存在")
        except (MediaError, ValueError, OSError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except RuntimeError as error:
            self._error(HTTPStatus.CONFLICT, str(error))

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host 不允许")
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/session/"):
            if not self.server.consume_bootstrap(parsed.path):
                self._error(HTTPStatus.FORBIDDEN, "本地会话入口无效或已使用")
                return
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            self.send_header("Cache-Control", "no-store")
            self._security_headers()
            self.send_header(
                "Set-Cookie",
                f"frameledger_token={self.server.token}; HttpOnly; SameSite=Strict; Path=/",
            )
            self.end_headers()
            return
        if parsed.path == "/":
            if not self._token_allowed():
                self._error(
                    HTTPStatus.FORBIDDEN,
                    "请使用启动命令输出的随机本地会话网址",
                )
                return
            body = _frontend_html(
                token=self.server.token,
                default_video=self.server.default_video,
                output_root=str(self.server.manager.output_root),
                output_choice_id=self.server.manager.default_output_choice_id,
                model_id=self.server.model_id,
                model_revision=self.server.model_revision,
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/app.js":
            self._send_bytes(
                _frontend_js().encode("utf-8"),
                mime="text/javascript; charset=utf-8",
            )
            return
        if not self._token_allowed():
            self._error(HTTPStatus.FORBIDDEN, "本地请求授权失败")
            return
        if parsed.path == "/api/status":
            job_id = urllib.parse.parse_qs(parsed.query).get("job_id", [""])[0]
            try:
                self._json(HTTPStatus.OK, self.server.manager.status(job_id))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "任务不存在")
            return
        if parsed.path == "/api/active":
            self._json(HTTPStatus.OK, {"job": self.server.manager.active_status()})
            return
        if parsed.path.startswith("/artifacts/"):
            relative_text = urllib.parse.unquote(
                parsed.path[len("/artifacts/") :]
            )
            parts = Path(relative_text).parts
            if len(parts) < 4:
                self._error(HTTPStatus.BAD_REQUEST, "结果路径无效")
                return
            job_id, index_text, *artifact_parts = parts
            try:
                target = self.server.manager.artifact_path(
                    job_id, int(index_text), Path(*artifact_parts)
                )
                mime = (
                    "text/plain; charset=utf-8"
                    if target.suffix.lower() == ".md"
                    else mimetypes.guess_type(target.name)[0]
                    or "application/octet-stream"
                )
                self._send_file(target, mime=mime)
            except (KeyError, FileNotFoundError, ValueError):
                self._error(HTTPStatus.NOT_FOUND, "结果文件不存在")
            return
        if parsed.path.startswith("/runs/"):
            relative_text = urllib.parse.unquote(parsed.path[len("/runs/") :])
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                self._error(HTTPStatus.BAD_REQUEST, "结果路径无效")
                return
            parts = relative.parts
            if len(parts) < 2 or not RUN_NAME_PATTERN.fullmatch(parts[0]):
                self._error(HTTPStatus.BAD_REQUEST, "结果运行名称无效")
                return
            if len(parts) == 2 and parts[1] == "source-video":
                try:
                    source = self._source_for_run(parts[0])
                    self._send_file(source, mime="video/mp4")
                except FileNotFoundError:
                    self._error(HTTPStatus.NOT_FOUND, "源视频不存在")
                except (ValueError, OSError, json.JSONDecodeError) as error:
                    self._error(HTTPStatus.CONFLICT, str(error))
                return
            try:
                self._run_document(parts[0])
            except FileNotFoundError:
                self._error(HTTPStatus.NOT_FOUND, "结果运行不存在")
                return
            except (ValueError, OSError, json.JSONDecodeError) as error:
                self._error(HTTPStatus.CONFLICT, str(error))
                return
            target = (self.server.manager.output_root / relative).resolve()
            if (
                not target.is_relative_to(self.server.manager.output_root)
                or not target.is_file()
            ):
                self._error(HTTPStatus.NOT_FOUND, "结果文件不存在")
                return
            mime = (
                "text/plain; charset=utf-8"
                if target.suffix.lower() == ".md"
                else mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            )
            if len(parts) == 3 and parts[1:] == ("visual", "review.html"):
                try:
                    source = self._source_for_run(parts[0])
                    rewritten = target.read_text(encoding="utf-8").replace(
                        source.as_uri(), "../source-video"
                    )
                    self._send_bytes(
                        rewritten.encode("utf-8"), mime="text/html; charset=utf-8"
                    )
                except (ValueError, OSError, json.JSONDecodeError) as error:
                    self._error(HTTPStatus.CONFLICT, str(error))
                return
            self._send_file(target, mime=mime)
            return
        self._error(HTTPStatus.NOT_FOUND, "页面不存在")


def _format_duration(value: Any) -> str:
    seconds = float(value)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def serve_workflow_ui(
    *,
    output_root: str | Path,
    port: int = 8765,
    default_video: str | Path | None = None,
    model_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    mlx_whisper_helper: str | Path | None = None,
    apple_vision_helper: str | Path | None = None,
    ffmpeg: str | Path | None = None,
) -> None:
    if port < 0 or port > 65535:
        raise ValueError("Port must be between 0 and 65535")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    options = {
        "model_path": model_path or default_model_path(),
        "model_id": model_id,
        "model_revision": model_revision,
        "mlx_whisper_helper": mlx_whisper_helper or default_asr_helper(),
        "apple_vision_helper": apple_vision_helper,
        "ffmpeg": ffmpeg,
    }
    manager = WorkflowJobManager(output_root=root, workflow_options=options)
    token = secrets.token_urlsafe(32)
    bootstrap_token = secrets.token_urlsafe(32)
    server = _WorkflowServer(
        ("127.0.0.1", port),
        WorkflowRequestHandler,
        token=token,
        manager=manager,
        default_video=str(Path(default_video).expanduser().resolve())
        if default_video
        else None,
        model_id=model_id,
        model_revision=model_revision,
        bootstrap_token=bootstrap_token,
    )
    actual_port = server.server_address[1]
    print(
        "FrameLedger local workflow UI: "
        f"http://127.0.0.1:{actual_port}/session/{bootstrap_token}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
