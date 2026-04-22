

import subprocess
from pathlib import Path

from anthropic.types import ToolParam
from anthropic.types.tool_param import InputSchemaTyped

MAX_BASH_OUTPUT = 50000
WORKDIR = Path.cwd()

# ==============================================================================
# 基础工具
# ==============================================================================
def run_bash(command: str) -> str:
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120
        )
        out = (r.stdout + r.stderr).strip()
        return out[:MAX_BASH_OUTPUT] if out else '(no output)'
    except subprocess.TimeoutExpired:
        return 'Error: Timeout (120s)'
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ==============================================================================
# 工具注册表
# ==============================================================================
TOOL_HANDLERS: dict = {
    'bash':       lambda **kw: run_bash(kw['command']),
}
TOOL_PARAMS: dict[str, ToolParam] = {
    'bash': ToolParam(
        name='bash',
        description='Run a shell command in the current workspace.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={'command': {'type': 'string'}},
            required=['command'],
        ),
    ),
}


