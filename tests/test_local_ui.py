from __future__ import annotations

import io
import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from frameledger.local_ui import (
    WorkflowJobManager,
    WorkflowRequestHandler,
    _WorkflowServer,
)


class _ChunkedStream:
    def __init__(self, content: bytes, *, maximum_chunk: int) -> None:
        self._stream = io.BytesIO(content)
        self.maximum_chunk = maximum_chunk
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            size = self.maximum_chunk
        return self._stream.read(min(size, self.maximum_chunk))


class LocalWorkflowUiTests(unittest.TestCase):
    @staticmethod
    def _video(path: Path, content: bytes = b"not-decoded-test-video") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path.resolve()

    @staticmethod
    def _manager(output_root: Path) -> WorkflowJobManager:
        output_root.mkdir(parents=True, exist_ok=True)
        return WorkflowJobManager(
            output_root=output_root,
            workflow_options={"model_id": "local-test-model"},
        )

    @staticmethod
    def _wait_terminal(
        manager: WorkflowJobManager, job_id: str, *, timeout: float = 3.0
    ) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = manager.status(job_id)
            if status["status"] in {"ok", "partial", "error"}:
                return status
            threading.Event().wait(0.005)
        raise AssertionError(f"Batch {job_id} did not finish within {timeout} seconds")

    def test_batch_runs_in_order_and_continues_after_middle_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = self._manager(root / "outputs")
            videos = [
                self._video(root / "sources" / f"{name}.mp4")
                for name in ("first", "second", "third")
            ]
            calls: list[str] = []

            def fake_workflow(source: Path, *, output: Path, progress_callback, **options):
                calls.append(source.name)
                self.assertEqual(options["model_id"], "local-test-model")
                progress_callback(
                    {
                        "current_stage": "visual",
                        "stages": {"visual": {"status": "running"}},
                    }
                )
                output.mkdir(parents=True, exist_ok=False)
                if source.name == "second.mp4":
                    raise RuntimeError("intentional middle failure")
                return {"kind": "complete_local_workflow", "output": str(output)}

            with patch(
                "frameledger.local_ui.run_complete_workflow",
                side_effect=fake_workflow,
            ):
                started = manager.start_batch(
                    video_paths=[str(path) for path in videos],
                    output_choice_id=manager.default_output_choice_id,
                )
                finished = self._wait_terminal(manager, str(started["job_id"]))

            self.assertEqual(calls, ["first.mp4", "second.mp4", "third.mp4"])
            self.assertEqual(finished["status"], "partial")
            self.assertEqual(finished["summary"], {"ok": 2, "error": 1, "pending": 0})
            self.assertEqual(
                [item["status"] for item in finished["items"]],
                ["ok", "error", "ok"],
            )
            self.assertIn("intentional middle failure", finished["items"][1]["error"])
            self.assertIsNone(manager.active_job_id)

    def test_active_batch_rejects_another_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = self._manager(root / "outputs")
            first = self._video(root / "sources" / "first.mp4")
            second = self._video(root / "sources" / "second.mp4")
            entered = threading.Event()
            release = threading.Event()

            def blocking_workflow(source: Path, *, output: Path, **kwargs):
                entered.set()
                if not release.wait(2.0):
                    raise RuntimeError("test did not release blocked workflow")
                output.mkdir(parents=True, exist_ok=False)
                return {"kind": "complete_local_workflow"}

            with patch(
                "frameledger.local_ui.run_complete_workflow",
                side_effect=blocking_workflow,
            ) as workflow:
                started = manager.start_batch(
                    video_paths=[str(first)],
                    output_choice_id=manager.default_output_choice_id,
                )
                self.assertTrue(entered.wait(1.0))
                with self.assertRaisesRegex(RuntimeError, "批次正在运行"):
                    manager.start_batch(
                        video_paths=[str(second)],
                        output_choice_id=manager.default_output_choice_id,
                    )
                release.set()
                finished = self._wait_terminal(manager, str(started["job_id"]))

            self.assertEqual(finished["status"], "ok")
            self.assertEqual(workflow.call_count, 1)
            self.assertIsNone(manager.active_job_id)

    def test_duplicate_and_part_inputs_fail_closed_before_job_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._video(root / "sources" / "episode.mp4")
            incomplete = self._video(root / "sources" / "episode.mp4.part")

            cases = (
                ("duplicate", [str(source), str(source)], "重复"),
                ("part", [str(source), str(incomplete)], "Incomplete download"),
            )
            for name, video_paths, message in cases:
                with self.subTest(case=name):
                    manager = self._manager(root / f"outputs-{name}")
                    with patch(
                        "frameledger.local_ui.run_complete_workflow"
                    ) as workflow:
                        with self.assertRaisesRegex((ValueError, RuntimeError), message):
                            manager.start_batch(
                                video_paths=video_paths,
                                output_choice_id=manager.default_output_choice_id,
                            )

                    workflow.assert_not_called()
                    self.assertEqual(manager.jobs, {})
                    self.assertIsNone(manager.active_job_id)
                    self.assertFalse(
                        (manager.output_root / ".frameledger-batches").exists()
                    )

    def test_auto_names_are_unique_and_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            manager = self._manager(output_root)
            existing = output_root / "episode"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("preserve me", encoding="utf-8")
            videos = [
                self._video(root / source_directory / "episode.mp4")
                for source_directory in ("source-a", "source-b")
            ]
            outputs: list[Path] = []

            def fake_workflow(source: Path, *, output: Path, **kwargs):
                outputs.append(output)
                output.mkdir(parents=True, exist_ok=False)
                return {"kind": "complete_local_workflow"}

            with patch(
                "frameledger.local_ui.run_complete_workflow",
                side_effect=fake_workflow,
            ) as workflow:
                started = manager.start_batch(
                    video_paths=[str(path) for path in videos],
                    output_choice_id=manager.default_output_choice_id,
                )
                self.assertEqual(
                    [item["run_name"] for item in started["items"]],
                    ["episode-2", "episode-3"],
                )
                finished = self._wait_terminal(manager, str(started["job_id"]))
                with self.assertRaisesRegex(ValueError, "已经存在"):
                    manager.start_batch(
                        video_paths=[str(videos[0])],
                        output_choice_id=manager.default_output_choice_id,
                        explicit_names=["episode"],
                    )

            self.assertEqual(finished["status"], "ok")
            self.assertEqual(
                [path.name for path in outputs], ["episode-2", "episode-3"]
            )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
            self.assertEqual(workflow.call_count, 2)

    def test_registered_output_choice_is_session_scoped_and_used_by_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default_root = root / "default-output"
            chosen_root = root / "chosen-output"
            chosen_root.mkdir()
            manager = self._manager(default_root)
            registration = manager.register_output_root(chosen_root)
            repeated = manager.register_output_root(chosen_root)
            self.assertEqual(registration, repeated)
            choice_id = registration["choice_id"]
            self.assertEqual(manager.output_choice(choice_id), chosen_root.resolve())
            self.assertEqual(manager.output_root, default_root.resolve())

            other_manager = self._manager(root / "other-session")
            with self.assertRaisesRegex(ValueError, "选择已失效"):
                other_manager.output_choice(choice_id)

            source = self._video(root / "sources" / "registered.mp4")

            def fake_workflow(source: Path, *, output: Path, **kwargs):
                output.mkdir(parents=True, exist_ok=False)
                return {"kind": "complete_local_workflow"}

            with patch(
                "frameledger.local_ui.run_complete_workflow",
                side_effect=fake_workflow,
            ):
                started = manager.start_batch(
                    video_paths=[str(source)], output_choice_id=choice_id
                )
                finished = self._wait_terminal(manager, str(started["job_id"]))

            item_output = Path(finished["items"][0]["output_path"])
            self.assertEqual(item_output.parent.resolve(), chosen_root.resolve())
            self.assertEqual(Path(finished["output_root"]), chosen_root.resolve())
            batch_manifest = Path(str(finished["batch_manifest"]))
            self.assertEqual(batch_manifest.parent.parent, chosen_root.resolve())
            self.assertTrue(batch_manifest.is_file())
            persisted = json.loads(batch_manifest.read_text(encoding="utf-8"))
            self.assertEqual(persisted["job_id"], finished["job_id"])
            self.assertEqual(Path(persisted["output_root"]), chosen_root.resolve())

    def test_project_tree_output_requires_ignored_outputs_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            allowed = project / "outputs"
            unsafe = project / "private-results"
            external = root / "external-results"
            for directory in (allowed, unsafe, external):
                directory.mkdir(parents=True)

            with patch("frameledger.local_ui.project_root", return_value=project):
                manager = self._manager(allowed)
                for directory in (project, unsafe):
                    with self.subTest(directory=directory):
                        with self.assertRaisesRegex(ValueError, "outputs"):
                            manager.register_output_root(directory)
                registration = manager.register_output_root(external)

            self.assertEqual(
                manager.output_choice(registration["choice_id"]), external.resolve()
            )

    def test_save_import_streams_and_cleans_interrupted_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = self._manager(root / "successful-output")
            content = b"streamed-local-video-bytes" * 5
            stream = _ChunkedStream(content, maximum_chunk=7)

            with patch("frameledger.local_ui.probe_video") as probe:
                imported = manager.save_import(
                    filename="dragged.mp4",
                    stream=stream,
                    content_length=len(content),
                )
            probe.assert_called_once()

            target = Path(imported["video_path"])
            self.assertEqual(target.read_bytes(), content)
            self.assertTrue(target.is_relative_to(manager.import_root.resolve()))
            self.assertEqual(imported["storage"], "managed_local_import")
            self.assertEqual(imported["size_bytes"], len(content))
            self.assertGreater(len(stream.read_sizes), 1)
            self.assertTrue(all(size <= 4 * 1024 * 1024 for size in stream.read_sizes))
            self.assertFalse(any(manager.import_root.rglob("*.uploading")))

            interrupted_manager = self._manager(root / "interrupted-output")
            short_stream = _ChunkedStream(b"only-a-prefix", maximum_chunk=4)
            with self.assertRaisesRegex(ValueError, "导入中断"):
                interrupted_manager.save_import(
                    filename="interrupted.mp4",
                    stream=short_stream,
                    content_length=100,
                )

            self.assertTrue(interrupted_manager.import_root.is_dir())
            self.assertEqual(list(interrupted_manager.import_root.iterdir()), [])
            self.assertFalse(any(interrupted_manager.import_root.rglob("*.uploading")))

    def test_artifact_path_rejects_traversal_absolute_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = self._manager(root / "outputs")
            run_output = manager.output_root / "episode"
            markdown_directory = run_output / "markdown"
            markdown_directory.mkdir(parents=True)
            report = markdown_directory / "report.md"
            report.write_text("# local report\n", encoding="utf-8")
            outside = root / "outside.md"
            outside.write_text("outside", encoding="utf-8")
            escape_link = run_output / "escape.md"
            escape_link.symlink_to(outside)
            job_id = "job-for-artifacts"
            with manager.lock:
                manager.jobs[job_id] = {
                    "output_root": str(manager.output_root),
                    "items": [{"output_path": str(run_output)}]
                }

            self.assertEqual(
                manager.artifact_path(
                    job_id, 0, Path("markdown") / "report.md"
                ),
                report.resolve(),
            )
            for relative in (Path("../outside.md"), outside.resolve(), Path()):
                with self.subTest(relative=str(relative)):
                    with self.assertRaisesRegex(ValueError, "路径无效"):
                        manager.artifact_path(job_id, 0, relative)
            with self.assertRaises(FileNotFoundError):
                manager.artifact_path(job_id, 0, Path("escape.md"))
            with self.assertRaises(FileNotFoundError):
                manager.artifact_path(job_id, 0, Path("missing.md"))
            with self.assertRaises(KeyError):
                manager.artifact_path("unknown-job", 0, Path("report.md"))
            with self.assertRaises(KeyError):
                manager.artifact_path(job_id, 1, Path("report.md"))

            replaced_output = manager.output_root / "replaced"
            replaced_output.mkdir()
            replacement_job = "job-with-replaced-output"
            with manager.lock:
                manager.jobs[replacement_job] = {
                    "output_root": str(manager.output_root),
                    "items": [{"output_path": str(replaced_output)}],
                }
            replaced_output.rmdir()
            replaced_output.symlink_to(root)
            with self.assertRaises(FileNotFoundError):
                manager.artifact_path(replacement_job, 0, Path("outside.md"))

    def test_loopback_http_batch_import_auth_and_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = self._manager(root / "outputs")
            token = "test-loopback-token"
            try:
                server = _WorkflowServer(
                    ("127.0.0.1", 0),
                    WorkflowRequestHandler,
                    token=token,
                    manager=manager,
                    default_video=None,
                    model_id="local-test-model",
                    model_revision="test-revision",
                    bootstrap_token="test-bootstrap-token",
                )
            except PermissionError:
                self.skipTest("loopback sockets are unavailable in this sandbox")
            port = int(server.server_address[1])
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()

            def request(
                method: str,
                path: str,
                *,
                body: bytes = b"",
                headers: dict[str, str] | None = None,
            ) -> tuple[int, bytes]:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                connection.request(method, path, body=body, headers=headers or {})
                response = connection.getresponse()
                result = response.status, response.read()
                connection.close()
                return result

            origin = f"http://127.0.0.1:{port}"
            json_headers = {
                "Content-Type": "application/json",
                "Origin": origin,
                "X-FrameLedger-Token": token,
            }
            try:
                root_status, root_body = request("GET", "/")
                self.assertEqual(root_status, 403)
                self.assertNotIn(token.encode(), root_body)
                bad_bootstrap_status, _ = request(
                    "GET", "/session/wrong-bootstrap-token"
                )
                self.assertEqual(bad_bootstrap_status, 403)
                bootstrap_status, _ = request(
                    "GET", "/session/test-bootstrap-token"
                )
                self.assertEqual(bootstrap_status, 303)
                reused_bootstrap_status, _ = request(
                    "GET", "/session/test-bootstrap-token"
                )
                self.assertEqual(reused_bootstrap_status, 403)
                authorized_root_status, authorized_root_body = request(
                    "GET",
                    "/",
                    headers={"Cookie": f"frameledger_token={token}"},
                )
                self.assertEqual(authorized_root_status, 200)
                self.assertIn(token.encode(), authorized_root_body)

                denied_status, _ = request(
                    "POST",
                    "/api/run-batch",
                    body=b"{}",
                    headers={
                        "Content-Type": "application/json",
                        "Cookie": f"frameledger_token={token}",
                        "Origin": origin,
                    },
                )
                self.assertEqual(denied_status, 403)
                wrong_origin_status, _ = request(
                    "POST",
                    "/api/run-batch",
                    body=b"{}",
                    headers={
                        **json_headers,
                        "Origin": f"http://127.0.0.1:{port + 1}",
                    },
                )
                self.assertEqual(wrong_origin_status, 403)
                wrong_type_status, _ = request(
                    "POST",
                    "/api/run-batch",
                    body=b"{}",
                    headers={**json_headers, "Content-Type": "text/plain"},
                )
                self.assertEqual(wrong_type_status, 400)

                upload = b"loopback-video-test"
                with patch("frameledger.local_ui.probe_video") as probe:
                    import_status, import_body = request(
                        "POST",
                        "/api/import?filename=dragged.mp4&output_choice_id=default",
                        body=upload,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(len(upload)),
                            "Origin": origin,
                            "X-FrameLedger-Token": token,
                        },
                    )
                probe.assert_called_once()
                self.assertEqual(import_status, 201)
                imported = json.loads(import_body)
                self.assertEqual(Path(imported["video_path"]).read_bytes(), upload)

                def fake_workflow(source: Path, *, output: Path, **kwargs):
                    markdown = output / "markdown"
                    markdown.mkdir(parents=True, exist_ok=False)
                    (markdown / "report.md").write_text(
                        "![frame](frame.png)\n", encoding="utf-8"
                    )
                    return {"kind": "complete_local_workflow"}

                payload = json.dumps(
                    {
                        "video_paths": [imported["video_path"]],
                        "output_choice_id": "default",
                    }
                ).encode()
                with patch(
                    "frameledger.local_ui.run_complete_workflow",
                    side_effect=fake_workflow,
                ):
                    batch_status, batch_body = request(
                        "POST",
                        "/api/run-batch",
                        body=payload,
                        headers=json_headers,
                    )
                    self.assertEqual(batch_status, 202)
                    started = json.loads(batch_body)
                    finished = self._wait_terminal(manager, started["job_id"])

                self.assertEqual(finished["status"], "ok")
                report_url = str(finished["items"][0]["report_url"])
                report_status, report_body = request(
                    "GET",
                    report_url,
                    headers={"Cookie": f"frameledger_token={token}"},
                )
                self.assertEqual(report_status, 200)
                self.assertIn(b"frame.png", report_body)
            finally:
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
