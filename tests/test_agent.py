import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nl2python_agent.agent import (
    CodingAgentError,
    CodingWorkspace,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    _model_request,
    coding_agent_status,
    run_coding_agent,
)


class TestCodingWorkspace(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.workspace = CodingWorkspace(self.root)

    def test_map_repository_and_search_python_symbols(self):
        source = self.root / "src" / "sample.py"
        source.parent.mkdir()
        source.write_text(
            "class CsvReader:\n    def read_rows(self):\n        return []\n",
            encoding="utf-8",
        )
        (self.root / ".git").mkdir()
        (self.root / ".git" / "ignored.py").write_text("class Hidden: pass\n", encoding="utf-8")

        repository_map = self.workspace.map_repository()
        symbols = self.workspace.search_symbols("read_rows")

        self.assertIn("src/sample.py", repository_map["files"])
        self.assertNotIn(".git/ignored.py", repository_map["files"])
        self.assertEqual(
            symbols,
            [
                {
                    "path": "src/sample.py",
                    "name": "read_rows",
                    "kind": "function",
                    "line": 2,
                }
            ],
        )

    def test_read_file_returns_content_hash(self):
        path = self.root / "main.py"
        path.write_bytes(b"print('hello')\n")

        result = self.workspace.read_file("main.py")

        self.assertEqual(result["content"], "print('hello')\n")
        self.assertEqual(
            result["sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def test_patch_can_create_cpp_source_file(self):
        created = self.workspace.apply_file_patch(
            "test/helloworld.cpp",
            "",
            'cout<<"hello world!";\n',
            "",
        )

        self.assertTrue(created["created"])
        self.assertEqual(
            (self.root / "test" / "helloworld.cpp").read_text(encoding="utf-8"),
            'cout<<"hello world!";\n',
        )

    def test_status_reports_deepseek_configuration_without_exposing_key(self):
        with patch("nl2python_agent.agent.DEEPSEEK_API_KEY", "test-secret"):
            status = coding_agent_status(self.workspace)

        self.assertTrue(status["model_available"])
        self.assertEqual(status["model"], DEEPSEEK_MODEL)
        self.assertEqual(status["provider"], "DeepSeek")
        self.assertEqual(status["mode"], "autopilot")
        self.assertNotIn("test-secret", json.dumps(status))

    def test_status_reports_missing_deepseek_key(self):
        with patch("nl2python_agent.agent.DEEPSEEK_API_KEY", ""):
            status = coding_agent_status(self.workspace)

        self.assertFalse(status["model_available"])
        self.assertIn("DEEPSEEK_API_KEY", status["message"])

    def test_read_and_patch_reject_paths_outside_workspace(self):
        with self.assertRaises(CodingAgentError):
            self.workspace.read_file("../outside.py")

        with self.assertRaises(CodingAgentError):
            self.workspace.read_file(".env")

    def test_patch_requires_exact_content_and_current_hash(self):
        path = self.root / "main.py"
        path.write_bytes(b"VALUE = 1\nVALUE = 1\n")
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        with self.assertRaises(CodingAgentError):
            self.workspace.apply_file_patch(
                "main.py",
                "VALUE = 1\n",
                "VALUE = 2\n",
                current_hash,
            )
        with self.assertRaises(CodingAgentError):
            self.workspace.apply_file_patch("main.py", "VALUE = 1\n", "VALUE = 2\n", "stale")

    def test_patch_applies_one_exact_change_and_creates_new_file(self):
        path = self.root / "main.py"
        path.write_bytes(b"VALUE = 1\n")
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        result = self.workspace.apply_file_patch(
            "main.py",
            "VALUE = 1\n",
            "VALUE = 2\n",
            current_hash,
        )
        created = self.workspace.apply_file_patch(
            "tests/test_main.py",
            "",
            "def test_value():\n    assert True\n",
            "",
        )

        self.assertEqual(path.read_text(encoding="utf-8"), "VALUE = 2\n")
        self.assertIn("-VALUE = 1", result["diff"])
        self.assertTrue(created["created"])
        self.assertTrue((self.root / "tests" / "test_main.py").exists())

    @patch("nl2python_agent.agent.subprocess.run")
    def test_run_tests_uses_pytest_without_shell_and_strips_credentials(self, mock_run):
        test_file = self.root / "tests" / "test_example.py"
        test_file.parent.mkdir()
        test_file.write_text("def test_example(): pass\n", encoding="utf-8")
        mock_run.return_value = Mock(returncode=0, stdout="1 passed\n", stderr="")

        with patch.dict(os.environ, {"SAMPLE_API_TOKEN": "private-value", "VISIBLE_FLAG": "yes"}):
            result = self.workspace.run_tests(["tests/test_example.py"])

        command = mock_run.call_args.args[0]
        self.assertIn("-m", command)
        self.assertIn("pytest", command)
        self.assertNotIn("shell", mock_run.call_args.kwargs)
        self.assertNotIn("SAMPLE_API_TOKEN", mock_run.call_args.kwargs["env"])
        self.assertEqual(mock_run.call_args.kwargs["env"]["VISIBLE_FLAG"], "yes")
        self.assertTrue(result["passed"])


class TestCodingAgentLoop(unittest.TestCase):
    @patch("nl2python_agent.agent.requests.post")
    def test_model_request_uses_deepseek_openai_compatible_endpoint(self, mock_post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Ready."}}],
        }
        mock_post.return_value = response

        with patch("nl2python_agent.agent.DEEPSEEK_API_KEY", "test-secret"):
            message = _model_request([{"role": "user", "content": "Hello"}])

        self.assertEqual(message["content"], "Ready.")
        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args.args[0],
            f"{DEEPSEEK_BASE_URL}/chat/completions",
        )
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], DEEPSEEK_MODEL)
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-secret",
        )
        self.assertTrue(mock_post.call_args.kwargs["json"]["tools"])

    def test_model_request_requires_api_key(self):
        with patch("nl2python_agent.agent.DEEPSEEK_API_KEY", ""):
            with self.assertRaisesRegex(CodingAgentError, "DEEPSEEK_API_KEY"):
                _model_request([{"role": "user", "content": "Hello"}])

    @patch("nl2python_agent.agent._model_request")
    def test_model_message_returns_response_and_persistent_session_id(self, mock_model):
        mock_model.return_value = {"content": "Done.", "tool_calls": []}

        response, session_id = run_coding_agent(
            "Add a small helper.",
            workspace=Mock(),
        )

        self.assertEqual(response, "Done.")
        self.assertTrue(session_id)
        mock_model.assert_called_once()

    @patch("nl2python_agent.agent._model_request")
    def test_agent_runs_workspace_tools_before_final_response(self, mock_model):
        workspace = Mock()
        workspace.map_repository.return_value = {"files": ["app.py"]}
        mock_model.side_effect = [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "map-call",
                        "function": {
                            "name": "map_repository",
                            "arguments": json.dumps({"limit": 10}),
                        },
                    }
                ],
            },
            {"content": "The project contains app.py.", "tool_calls": []},
        ]

        response, _session_id = run_coding_agent(
            "Describe this project.",
            workspace=workspace,
        )

        self.assertEqual(response, "The project contains app.py.")
        workspace.map_repository.assert_called_once_with(10)
        self.assertEqual(mock_model.call_count, 2)

    @patch("nl2python_agent.agent._model_request")
    def test_agent_requires_a_plan_before_applying_file_patches(self, mock_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "main.py"
            path.write_bytes(b"VALUE = 1\n")
            workspace = CodingWorkspace(Path(temp_dir))
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            observed_tool_results = []

            def model_response(messages):
                if messages[-1].get("role") == "tool":
                    observed_tool_results.append(json.loads(messages[-1]["content"]))
                    return {"content": "I need to record a plan before editing."}
                if len(observed_tool_results) == 0:
                    return {
                        "tool_calls": [
                            {
                                "id": "early-edit",
                                "function": {
                                    "name": "apply_file_patch",
                                    "arguments": json.dumps(
                                        {
                                            "path": "main.py",
                                            "old_text": "VALUE = 1\n",
                                            "new_text": "VALUE = 2\n",
                                            "expected_sha256": file_hash,
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                return {"content": "I need to record a plan before editing."}

            mock_model.side_effect = model_response
            response, _session_id = run_coding_agent(
                "Update the value.",
                workspace=workspace,
            )
            final_content = path.read_bytes()

        self.assertIn("record_plan before editing", observed_tool_results[0]["error"])
        self.assertEqual(final_content, b"VALUE = 1\n")
        self.assertIn("record a plan", response)

    @patch("nl2python_agent.agent._model_request")
    def test_agent_can_edit_after_recording_plan(self, mock_model):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "main.py"
            path.write_bytes(b"VALUE = 1\n")
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            mock_model.side_effect = [
                {
                    "tool_calls": [
                        {
                            "id": "plan",
                            "function": {
                                "name": "record_plan",
                                "arguments": json.dumps(
                                    {"steps": ["Update the value.", "Run the focused tests."]}
                                ),
                            },
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "id": "edit",
                            "function": {
                                "name": "apply_file_patch",
                                "arguments": json.dumps(
                                    {
                                        "path": "main.py",
                                        "old_text": "VALUE = 1\n",
                                        "new_text": "VALUE = 2\n",
                                        "expected_sha256": file_hash,
                                    }
                                ),
                            },
                        }
                    ]
                },
                {"content": "Plan recorded; main.py updated."},
            ]

            response, _session_id = run_coding_agent(
                "Update the value.",
                workspace=CodingWorkspace(Path(temp_dir)),
            )
            final_content = path.read_bytes()

        self.assertEqual(final_content, b"VALUE = 2\n")
        self.assertIn("Plan recorded", response)
