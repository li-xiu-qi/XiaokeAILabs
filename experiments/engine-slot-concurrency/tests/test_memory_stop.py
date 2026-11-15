import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_llamacpp_slots.py"


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True


class MemoryStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        spec = importlib.util.spec_from_file_location("slot_bench", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"BENCH_OUT": self.temp.name, "LLAMA_SERVER": "test", "MODEL_GGUF": "test"}):
            spec.loader.exec_module(self.module)

    def test_low_memory_stops_only_the_registered_process(self):
        proc = FakeProcess()
        unrelated = FakeProcess()
        self.module.register_service(proc)
        real_open = open

        def fake_open(path, *args, **kwargs):
            if path == "/proc/meminfo":
                return io.StringIO("MemAvailable: 1000 kB\nSwapFree: 1000 kB\n")
            return real_open(path, *args, **kwargs)

        with patch.object(self.module, "open", fake_open, create=True), patch.object(
            self.module.subprocess, "run", return_value=type("Result", (), {"stdout": ""})()
        ):
            self.module.monitor()
        self.assertTrue(self.module.stop.is_set())
        self.assertTrue(proc.terminated)
        self.assertFalse(unrelated.terminated)

    def test_stop_before_registration_terminates_new_service(self):
        proc = FakeProcess()
        self.module.memory_stop()
        self.module.register_service(proc)
        self.assertTrue(proc.terminated)

    def test_unregistered_service_is_not_terminated(self):
        proc = FakeProcess()
        self.module.register_service(proc)
        self.module.unregister_service(proc)
        self.module.memory_stop()
        self.assertFalse(proc.terminated)


if __name__ == "__main__":
    unittest.main()
