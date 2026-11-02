"""Minimal local web server for the standalone NL2Python coding agent."""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import CodingAgentError, coding_agent_status, run_coding_agent


WEB_DIR = Path(__file__).parent.parent / "web"
DEFAULT_HOST = os.getenv("AGENT_WEB_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AGENT_WEB_PORT", "8080"))
MAX_REQUEST_BYTES = 50_000


class CodingAgentHandler(BaseHTTPRequestHandler):
    server_version = "NL2PythonAgent/1.0"

    def _is_local_request(self) -> bool:
        try:
            peer = self.client_address[0]
            if peer not in {"127.0.0.1", "::1"}:
                return False
            host = urlparse(f"//{self.headers.get('Host', '')}")
            host_port = host.port or (self.server.server_port if self.server.server_port in {80, 443} else None)
            if host.hostname not in {"localhost", "127.0.0.1", "::1"}:
                return False
            if host_port != self.server.server_port:
                return False

            origin = self.headers.get("Origin")
            if origin:
                parsed_origin = urlparse(origin)
                origin_port = parsed_origin.port or (
                    443 if parsed_origin.scheme == "https" else 80
                )
                if (
                    parsed_origin.scheme not in {"http", "https"}
                    or parsed_origin.hostname not in {"localhost", "127.0.0.1", "::1"}
                    or origin_port != self.server.server_port
                ):
                    return False
            return True
        except (ValueError, IndexError):
            return False

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(
            status,
            "application/json; charset=utf-8",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    def do_GET(self) -> None:
        if not self._is_local_request():
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not Found")
            return

        path = urlparse(self.path).path
        if path == "/api/coding/status":
            try:
                self._send_json(HTTPStatus.OK, coding_agent_status())
            except CodingAgentError as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return

        static_files = {
            "/": ("coding.html", "text/html; charset=utf-8"),
            "/coding.js": ("coding.js", "application/javascript; charset=utf-8"),
            "/coding.css": ("coding.css", "text/css; charset=utf-8"),
        }
        asset = static_files.get(path)
        if asset is None:
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not Found")
            return
        filename, content_type = asset
        try:
            body = (WEB_DIR / filename).read_bytes()
        except OSError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not read UI asset: {exc}"})
            return
        self._send(HTTPStatus.OK, content_type, body)

    def do_POST(self) -> None:
        if not self._is_local_request():
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not Found")
            return
        if urlparse(self.path).path != "/api/coding/chat":
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length."})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes."},
            )
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Expected a JSON request body."})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), str):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "message must be a string."})
            return

        try:
            response, session_id = run_coding_agent(
                payload["message"],
                payload.get("session_id"),
            )
        except CodingAgentError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.OK,
            {"response": response, "session_id": session_id},
        )

    def log_message(self, format: str, *args) -> None:
        print(f"[coding-agent] {self.address_string()} - {format % args}")


def main() -> None:
    if not WEB_DIR.is_dir():
        raise FileNotFoundError(f"Coding agent web client not found: {WEB_DIR}")
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), CodingAgentHandler)
    print(f"NL2Python coding agent listening at http://{DEFAULT_HOST}:{DEFAULT_PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down NL2Python coding agent.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
