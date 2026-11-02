import subprocess
import sys
import unittest
from pathlib import Path

from nl2python_agent.server import CodingAgentHandler


NL2PYTHON_ROOT = Path(__file__).resolve().parents[1]


class TestStandaloneAgentServer(unittest.TestCase):
    def test_runtime_imports_without_microsoft_or_azure_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import nl2python_agent.agent; "
                    "assert not any(name == 'coreagent' or name.startswith('coreagent.') "
                    "for name in sys.modules); "
                    "assert not any(name == 'azure' or name.startswith('azure.') "
                    "for name in sys.modules); "
                    "assert 'msal' not in sys.modules"
                ),
            ],
            cwd=NL2PYTHON_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_server_rejects_non_local_or_cross_origin_requests(self):
        handler = CodingAgentHandler.__new__(CodingAgentHandler)
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {"Host": "localhost:8080"}
        handler.server = type("Server", (), {"server_port": 8080})()

        self.assertTrue(handler._is_local_request())

        handler.headers = {
            "Host": "localhost:8080",
            "Origin": "http://attacker.example",
        }
        self.assertFalse(handler._is_local_request())

        handler.headers = {"Host": "example.com:8080"}
        self.assertFalse(handler._is_local_request())

        handler.client_address = ("192.0.2.10", 12345)
        handler.headers = {"Host": "localhost:8080"}
        self.assertFalse(handler._is_local_request())
