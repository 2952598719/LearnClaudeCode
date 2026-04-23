import subprocess
from pathlib import Path

from anthropic.types import ToolParam
from anthropic.types.tool_param import InputSchemaTyped

# python不支持循环依赖，所以不要from agent import WORKDIR, THRESHOLD
WORKDIR = Path.cwd()
BASH_OUTPUT_THRESHOLD = 50000
READ_LINES_THRESHOLD = 50000
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
        return out[:BASH_OUTPUT_THRESHOLD] if out else '(no output)'
    except subprocess.TimeoutExpired:
        return 'Error: Timeout (120s)'
    except (FileNotFoundError, OSError) as e:
        return f'Error: {e}'

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f'...({len(lines) - limit} more lines)']
        return '\n'.join(lines)[:READ_LINES_THRESHOLD]
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

def run_compact():
    return 'Manual compression requested.'

# ==============================================================================
# 待办工具
# ==============================================================================
class TodoManager:
    VALID_STATUSES = ('pending', 'in_progress', 'completed')
    STATUS_MARKERS = {'pending': '[ ]', 'in_progress': '[>]', 'completed': '[x]'}

    def __init__(self):
        self.items: list[dict] = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError('Max 20 todos allowed')

        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            item_id = str(item.get('id', i + 1))
            text = str(item.get('text', '')).strip()
            status = str(item.get('status', 'pending')).lower()
            if not text:
                raise ValueError(f'Item {item_id}: text required')
            if status not in self.VALID_STATUSES:
                raise ValueError(f'Item {item_id}: invalid status {status!r}')
            if status == 'in_progress':
                in_progress_count += 1
            validated.append({'id': item_id, 'text': text, 'status': status})

        if in_progress_count > 1:
            raise ValueError('Only one task can be in_progress at a time')

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return 'No todos.'
        lines = [
            f"{self.STATUS_MARKERS[t['status']]} #{t['id']}: {t['text']}"
            for t in self.items
        ]
        done = sum(1 for t in self.items if t['status'] == 'completed')
        lines.append(f'\n({done}/{len(self.items)} completed)')
        return '\n'.join(lines)

# ==============================================================================
# 工具注册表
# ==============================================================================
TODO = TodoManager()
TOOL_HANDLERS: dict = {
    'bash':       lambda **kw: run_bash(kw['command']),
    'read_file':  lambda **kw: run_read(kw['path'], kw.get('limit')),
    'write_file': lambda **kw: run_write(kw['path'], kw['content']),
    'edit_file':  lambda **kw: run_edit(kw['path'], kw['old_text'], kw['new_text']),
    'compact':    lambda **kw: run_compact(),
    'todo':       lambda **kw: TODO.update(kw['items']),
}
TOOL_PARAMS: dict[str, ToolParam] = {
    'bash': ToolParam(
        name='bash',
        description='Run a shell command.',
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
    'compact': ToolParam(
        name='compact',
        description='Trigger manual conversation compression.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={'focus': {'type': 'string', 'description': 'What to preserve in the summary'}}
        )
    ),
    'todo': ToolParam(
        name='todo',
        description='Update task list. Track progress on multi-step tasks.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id':     {'type': 'string'},
                            'text':   {'type': 'string'},
                            'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed']},
                        },
                        'required': ['id', 'text', 'status'],
                    },
                },
            },
            required=['items'],
        ),
    ),
}


