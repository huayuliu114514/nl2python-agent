# NL2Python Repository Coding Agent

Standalone local coding agent powered by the DeepSeek API. Its runtime does not import the Pegasus/Microsoft
service, Azure SDKs, MISE, Microsoft identity, or enterprise telemetry.

## Run

From the `nl2python-agent` directory in PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"
$env:CODING_WORKSPACE_ROOT = (Get-Location).Path
python -m pip install -r requirements.txt
python -m pip install pytest
python -m nl2python_agent.server
```

Open http://127.0.0.1:8080. The server binds to loopback by default and only operates on files inside
`CODING_WORKSPACE_ROOT`. It supports repository mapping, Python symbol search, incremental text patches, and pytest
recovery. Supported source files include Python and C/C++.

Configuration:

- `DEEPSEEK_API_KEY`: required; read from the process environment and never returned by the status endpoint.
- `DEEPSEEK_MODEL`: defaults to `deepseek-chat`.
- `DEEPSEEK_BASE_URL`: defaults to `https://api.deepseek.com`.
- `CODING_WORKSPACE_ROOT`: defaults to the current working directory.
- `AGENT_WEB_HOST` and `AGENT_WEB_PORT`: defaults to `127.0.0.1` and `8080`.

The `.vscode` task starts this standalone server without starting the separate legacy Pegasus service. The agent
source is outside that service's Git repository and does not depend on its enterprise source or deployment files.
