

import subprocess
from pathlib import Path

from anthropic.types import ToolParam
from anthropic.types.tool_param import InputSchemaTyped

MAX_BASH_OUTPUT = 50000
WORKDIR = Path.cwd()


# ==============================================================================
# 调用函数
# ==============================================================================
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f'Path escapes workspace: {p}')
    return path



# ==============================================================================
# 基础工具
# ==============================================================================
def run_bash(command: str) -> str:
    dangerous = ['rm -rf /', 'sudo', 'shutdown', 'reboot', '> /dev/']
    if any(d in command for d in dangerous):
        return 'Error: Dangerous command blocked'
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
        return f'Error: {e}'

def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f'...({len(lines) - limit} more lines)']
        return '\n'.join(lines)[:MAX_BASH_OUTPUT]
    except Exception as e:
        return f'Error: {e}'

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f'Wrote {len(content)} bytes to {path}'
    except Exception as e:
        return f'Error: {e}'

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f'Error: Text not found in {path}'
        fp.write_text(content.replace(old_text, new_text, 1))
        return f'Edited {path}'
    except Exception as e:
        return f'Error: {e}'

# ==============================================================================
# 工具注册表
# ==============================================================================
TOOL_HANDLERS: dict = {
    'bash':       lambda **kw: run_bash(kw['command']),
    'read_file':  lambda **kw: run_read(kw['path'], kw.get('limit')),
    'write_file': lambda **kw: run_write(kw['path'], kw['content']),
    'edit_file':  lambda **kw: run_edit(kw['path'], kw['old_text'], kw['new_text']),
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
    'read_file': ToolParam(
        name='read_file',
        description='Read file contents.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={'path': {'type': 'string'}, 'limit': {'type': 'integer'}},
            required=['path'],
        ),
    ),
    'write_file': ToolParam(
        name='write_file',
        description='Write content to file.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={'path': {'type': 'string'}, 'content': {'type': 'string'}},
            required=['path', 'content'],
        ),
    ),
    'edit_file': ToolParam(
        name='edit_file',
        description='Replace exact text in file.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={
                'path':     {'type': 'string'},
                'old_text': {'type': 'string'},
                'new_text': {'type': 'string'},
            },
            required=['path', 'old_text', 'new_text'],
        ),
    ),
}


