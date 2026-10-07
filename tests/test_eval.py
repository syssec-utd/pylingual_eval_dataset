"""Runner regression tests without model downloads, GPUs, or a Redis service.

Run from the repository root: uv run python -m unittest discover -s tests -v
The upstream PythonVersion parser and decompiler are stubbed; Click is real.
"""
import contextlib
import csv
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


class StubPythonVersion:
    def __init__(self, value):
        match = re.fullmatch(r"3\.?([0-9]+)", value)
        if not match:
            raise ValueError(value)
        self.major = 3
        self.minor = int(match.group(1))


version_module = types.ModuleType("pylingual.utils.version")
version_module.PythonVersion = StubPythonVersion
spec = importlib.util.spec_from_file_location("dataset_eval_under_test", ROOT / "eval.py")
runner = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"pylingual.utils.version": version_module}):
    spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cli = CliRunner()
        self.context = MagicMock()
        self.pool = self.context.Pool.return_value.__enter__.return_value
        self.mp_patch = patch.object(runner.mp, "get_context", return_value=self.context)
        self.mp_patch.start()
        self.addCleanup(self.mp_patch.stop)
        self.gpu_patch = patch.object(runner, "detect_gpus", return_value=[])
        self.gpu_patch.start()
        self.addCleanup(self.gpu_patch.stop)
        self.root_patch = patch.object(runner, "DATASET_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.evaluate_patch = patch.object(runner, "evaluate")
        self.evaluate = self.evaluate_patch.start()
        self.addCleanup(self.evaluate_patch.stop)

    def manifest(self, name="313-pyc-list.txt"):
        path = self.root / name
        path.write_text("python-3.13/example/example.pyc\n")
        return path

    def invoke(self, args, **kwargs):
        result = self.cli.invoke(runner.main, args, **kwargs)
        self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
        return result

    def test_help_has_no_runtime_dataset_selector(self):
        result = self.invoke(["--help"])
        self.assertNotIn("--pylingual-version", result.output)
        self.assertIn("--redis-host", result.output)
        self.assertIn("--workers-per-gpu", result.output)

    def test_removed_dataset_selector_is_rejected(self):
        result = self.cli.invoke(runner.main, ["results", "-p", "v1"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No such option", result.output)

    def test_single_version_uses_checked_out_root(self):
        manifest = self.manifest()
        self.invoke(["results", "-v", "3.13"])
        self.evaluate.assert_called_once_with(
            self.pool, manifest, Path("results/python-3.13"), base_dir=self.root
        )
        self.context.Pool.assert_called_once_with(
            1, initializer=runner._init_worker,
            initargs=(self.context.Queue.return_value, None)
        )

    def test_all_versions_use_per_version_output(self):
        self.manifest("36-pyc-list.txt")
        self.manifest("313-pyc-list.txt")
        self.invoke(["results"])
        self.assertEqual(self.evaluate.call_count, 2)
        targets = {call.args[2] for call in self.evaluate.call_args_list}
        self.assertEqual(targets, {Path("results/python-3.6"), Path("results/python-3.13")})
        self.assertTrue(all(call.kwargs["base_dir"] == self.root for call in self.evaluate.call_args_list))

    def test_bundled_manifests_work_from_another_directory(self):
        manifest = self.manifest()
        other = self.root / "another-working-directory"
        other.mkdir()
        with contextlib.chdir(other):
            self.invoke(["results", "-v", "3.13"])
        self.assertEqual(self.evaluate.call_args.args[1], manifest)
        self.assertEqual(self.evaluate.call_args.kwargs["base_dir"], self.root)

    def test_custom_list_overrides_version_and_keeps_cwd_paths(self):
        manifest = self.manifest("custom.txt")
        self.invoke(["results", "-l", str(manifest), "-v", "9.99"])
        self.evaluate.assert_called_once_with(self.pool, manifest, Path("results"), base_dir=None)

    def test_missing_manifest_fails_before_starting_pool(self):
        result = self.cli.invoke(runner.main, ["results", "-v", "3.13"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Dataset manifest not found: 313-pyc-list.txt", result.output)
        self.context.Pool.assert_not_called()

    def test_empty_dataset_fails_before_starting_pool(self):
        result = self.cli.invoke(runner.main, ["results"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No dataset manifests found", result.output)
        self.context.Pool.assert_not_called()

    def test_explicit_redis_host_reaches_workers(self):
        self.manifest()
        result = self.invoke(["results", "--redis-host", "127.0.0.1"])
        self.assertIn("127.0.0.1:6379", result.output)
        self.assertEqual(self.context.Pool.call_args.kwargs["initargs"][1], "127.0.0.1")

    def test_redis_environment_is_read_at_invocation(self):
        self.manifest()
        self.invoke(["results"], env={"PYLINGUAL_REDIS_HOST": "cache.example"})
        self.assertEqual(self.context.Pool.call_args.kwargs["initargs"][1], "cache.example")

    def test_explicit_redis_host_overrides_environment(self):
        self.manifest()
        self.invoke(["results", "-r", "explicit"], env={"PYLINGUAL_REDIS_HOST": "environment"})
        self.assertEqual(self.context.Pool.call_args.kwargs["initargs"][1], "explicit")

    def test_parallel_gpu_slots_are_preserved(self):
        self.manifest()
        self.invoke(["results", "--gpus", "0,2", "--workers-per-gpu", "2"])
        self.assertEqual(self.context.Pool.call_args.args[0], 4)
        self.assertEqual(
            [call.args[0] for call in self.context.Queue.return_value.put.call_args_list],
            [0, 0, 2, 2]
        )


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.decompiler = types.ModuleType("pylingual.decompiler")
        self.decompiler.decompile = MagicMock()
        self.bytecode = types.ModuleType("pylingual.utils.generate_bytecode")
        self.bytecode.CompileError = type("CompileError", (Exception,), {})
        self.modules = patch.dict(sys.modules, {
            "pylingual.decompiler": self.decompiler,
            "pylingual.utils.generate_bytecode": self.bytecode,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.alarms = patch.object(runner.signal, "alarm")
        self.alarm = self.alarms.start()
        self.addCleanup(self.alarms.stop)
        self.host = patch.object(runner._process_file, "redis_host", "cache", create=True)
        self.host.start()
        self.addCleanup(self.host.stop)
        self.task = (Path("sample.pyc"), Path("result.py"))

    def test_initializer_assigns_gpu_and_redis_host(self):
        queue = MagicMock()
        queue.get.return_value = 2
        with patch.dict(os.environ, {}, clear=False), patch.object(runner.signal, "signal"):
            runner._init_worker(queue, "127.0.0.1")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(runner._process_file.redis_host, "127.0.0.1")

    def test_decompile_receives_redis_and_cancels_timeout(self):
        self.decompiler.decompile.return_value.equivalence_results = [
            types.SimpleNamespace(success=True, message="equal")
        ]
        result = runner._process_file(self.task)
        self.assertEqual(result, (self.task[0], None, [(True, "equal")]))
        self.decompiler.decompile.assert_called_once_with(
            *self.task, redis_cache_server_ip="cache", redis_port=6379
        )
        self.assertEqual([call.args[0] for call in self.alarm.call_args_list], [300, 0])

    def test_no_redis_host_disables_cache(self):
        runner._process_file.redis_host = None
        self.decompiler.decompile.return_value.equivalence_results = []
        runner._process_file(self.task)
        self.assertIsNone(self.decompiler.decompile.call_args.kwargs["redis_cache_server_ip"])

    def test_decompiler_error_is_reported_and_cancels_timeout(self):
        self.decompiler.decompile.side_effect = ValueError("bad bytecode")
        result = runner._process_file(self.task)
        self.assertEqual(result, (self.task[0], "ValueError('bad bytecode')", None))
        self.assertEqual([call.args[0] for call in self.alarm.call_args_list], [300, 0])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.progress = patch.object(runner.tqdm, "tqdm")
        self.progress.start()
        self.addCleanup(self.progress.stop)

    def test_bundled_paths_csv_and_statistics(self):
        manifest = self.root / "313-pyc-list.txt"
        manifest.write_text("python-3.13/one/one.pyc\n\npython-3.13/two/two.pyc\n")
        pool = MagicMock()
        def results(worker, tasks):
            self.assertIs(worker, runner._process_file)
            self.assertEqual(tasks[0][0], self.root / "python-3.13/one/one.pyc")
            return [(tasks[0][0], None, [(True, "equal")]), (tasks[1][0], "bad", None)]
        pool.imap_unordered.side_effect = results
        runner.evaluate(pool, manifest, self.root / "results", base_dir=self.root)
        output = next((self.root / "results").glob("pylingual-*"))
        with (output / "evaluation_results.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["category"] for row in rows], ["Equal", "", "DECOMPILER ERROR"])
        self.assertEqual(rows[2]["notes"], "bad")
        self.assertIn("File success: 1/2 50.00%", (output / "elapsed_time.txt").read_text())

    def test_custom_paths_keep_cwd_semantics(self):
        manifest = self.root / "custom.txt"
        manifest.write_text("relative/sample.pyc\n/absolute/sample.pyc\n")
        pool = MagicMock()
        pool.imap_unordered.return_value = []
        runner.evaluate(pool, manifest, self.root / "results")
        tasks = pool.imap_unordered.call_args.args[1]
        self.assertEqual([task[0] for task in tasks], [Path("relative/sample.pyc"), Path("/absolute/sample.pyc")])

    def test_empty_list_has_defined_statistics(self):
        manifest = self.root / "empty.txt"
        manifest.write_text("\n")
        pool = MagicMock()
        pool.imap_unordered.return_value = []
        runner.evaluate(pool, manifest, self.root / "results")
        output = next((self.root / "results").glob("pylingual-*"))
        self.assertIn("File success: 0/0 N/A", (output / "elapsed_time.txt").read_text())


class DatasetTests(unittest.TestCase):
    def test_every_bundled_pyc_exists_and_no_versioned_prefix_remains(self):
        manifests = sorted(ROOT.glob("*-pyc-list.txt"))
        self.assertEqual(len(manifests), 10)
        for manifest in manifests:
            for entry in manifest.read_text().splitlines():
                if not entry.strip():
                    continue
                with self.subTest(manifest=manifest.name, entry=entry):
                    self.assertFalse(entry.startswith(("pylingualv1/", "pylingualv2/")))
                    self.assertTrue((ROOT / entry).is_file())


if __name__ == "__main__":
    unittest.main()
