import os
import tempfile
import unittest

from runtime_paths import build_auto_trade_log_path, ensure_runtime_dirs


class RuntimePathTests(unittest.TestCase):
    def test_ensure_runtime_dirs_creates_expected_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                ensure_runtime_dirs()

                self.assertTrue(os.path.isdir("runtime/health"))
                self.assertTrue(os.path.isdir("runtime/orders"))
                self.assertTrue(os.path.isdir("runtime/signals"))
                self.assertTrue(os.path.isdir("runtime/logs"))
                self.assertTrue(build_auto_trade_log_path().startswith("runtime/logs/auto_trade_"))
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
