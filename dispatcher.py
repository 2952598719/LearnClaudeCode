import json
import subprocess
import threading
import uuid
from pathlib import Path

from anthropic.types import ToolParam
from anthropic.types.tool_param import InputSchemaTyped

# python不支持循环依赖，所以不要from agent import WORKDIR, THRESHOLD
WORKDIR = Path.cwd()
TASKS_DIR = WORKDIR / '.tasks'
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
# 任务工具
# ==============================================================================
# --- 基础待办 ---
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

class TaskManager:
    VALID_STATUSES = ('pending', 'in_progress', 'completed')
    STATUS_MARKERS = {'pending': '[ ]', 'in_progress': '[>]', 'completed': '[x]'}

    def __init__(self, tasks_dir: Path):
        tasks_dir.mkdir(exist_ok=True)
        self.dir = tasks_dir
        self._next_id = self._max_id() + 1  # 对于持久化到硬盘上的文件，应该每次启动都能读到

    def _max_id(self) -> int:
        ids = [int(f.stem.split('_')[1]) for f in self.dir.glob('task_*.json')]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f'task_{task_id}.json'
        if not path.exists():
            raise ValueError(f'Task {task_id} not found')
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def _clear_dependency(self, completed_id: int):
        # 这里看上去就是那种最值得优化的地方，存个后续节点就不用遍历了，但是作者没优化是有原因的
        # 首先任务数量不会太多，而且大部分情况下只需要查当前任务的前驱即可
        # 其次现在只有在任务完成时遍历，如果存后续的话，每次update时就要遍历来更新其他节点的前驱了，出错概率猛增
        for f in self.dir.glob('task_*.json'):
            task = json.loads(f.read_text())
            if completed_id in task.get('blockedBy', []):
                task['blockedBy'].remove(completed_id)
                self._save(task)

    def create(self, subject: str, description: str='') -> str:
        task = {
            'id': self._next_id,
            'subject': subject,
            'description': description,
            'status': 'pending',
            'blockedBy': [],
            'owner': '',
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(self, task_id: int, status: str = None, add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        task = self._load(task_id)
        if status:  # 如果status不为None
            if status not in self.VALID_STATUSES:
                raise ValueError(f'Invalid status: {status}')
            task['status'] = status
            if status == 'completed':
                self._clear_dependency(task_id)
        if add_blocked_by:
            task['blockedBy'] = list(set(task['blockedBy'] + add_blocked_by))
        if remove_blocked_by:
            task['blockedBy'] = [x for x in task['blockedBy'] if x not in remove_blocked_by]
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def list_all(self) -> str:
        tasks = []
        files = sorted(
            self.dir.glob('task_*.json'),
            key=lambda f: int(f.stem.split('_')[1])
        )
        for f in files:
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return 'No tasks.'
        lines = []
        for t in tasks:
            marker = self.STATUS_MARKERS.get(t['status'], '[?]')
            blocked = f"(blocked by: {t['blockedBy']})" if t.get('blockedBy') else ''
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return '\n'.join(lines)

class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()   # 创建互斥锁，with自动加锁解锁

    def run(self, command):
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {'status': 'running', 'result': None, 'command': command}
        threading.Thread(target=self._execute, args=(task_id, command), daemon=True).start()    # daemon=True构建守护线程异步执行任务
        return f'Background task {task_id} started: {command[:80]}'

    def _execute(self, task_id, command):
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = 'completed'
        except subprocess.TimeoutExpired:
            output, status = 'Error: Timeout (300s)', 'timeout'
        except Exception as e:
            output, status = f'Error: {e}', 'error'
        self.tasks[task_id].update(status=status, result=output or '(no output)')
        with self._lock:
            self._notification_queue.append({
                'task_id': task_id,
                'status': status,
                'command': command[:80],
                'result': (output or '(no output)')[:500]
            })

    def check(self, task_id=None):
        if task_id:     # 传入task_id就是检查这个任务的内容
            t = self.tasks.get(task_id)
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}" if t else \
                f'Error: Unknown task{task_id}'
        else:           # 没有传入task_id就是检查所有任务内容
            lines = [f"{tid}: [{t['status']}]" for tid, t in self.tasks.items()]
            return '\n'.join(lines) if lines else 'No background tasks'

    def drain_notifications(self):
        with self._lock:
            notifs, self._notification_queue = list(self._notification_queue), []
        return notifs

# ==============================================================================
# 工具注册表
# ==============================================================================
TODO = TodoManager()
TASKS = TaskManager(TASKS_DIR)
BG = BackgroundManager()
TOOL_HANDLERS: dict = {
    'bash':         lambda **kw: run_bash(kw['command']),
    'read_file':    lambda **kw: run_read(kw['path'], kw.get('limit')),
    'write_file':   lambda **kw: run_write(kw['path'], kw['content']),
    'edit_file':    lambda **kw: run_edit(kw['path'], kw['old_text'], kw['new_text']),
    'compact':      lambda **kw: run_compact(),
    # 任务-基础版
    'todo':         lambda **kw: TODO.update(kw['items']),
    # 任务-持久化
    'task_create':  lambda **kw: TASKS.create(kw['subject'], kw.get('description')),
    'task_update':  lambda **kw: TASKS.update(kw['task_id'], kw.get('status'), kw.get('addBlockedBy'), kw.get('removeBlockedBy')),
    'task_list':    lambda **kw: TASKS.list_all(),
    'task_get':     lambda **kw: TASKS.get(kw['task_id']),
    # 任务-指令异步执行
    'background_run':   lambda **kw: BG.run(kw['command']),
    'check_background': lambda **kw: BG.check(kw.get('task_id'))
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
    'task_create': ToolParam(
        name='task_create',
        description='Create a new task.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={
                'subject': {'type': 'string'},
                'description': {'type': 'string'},
            },
            required=['subject']
        )
    ),
    'task_update': ToolParam(
        name='task_update',
        description='Update status or dependencies of a task.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={
                'task_id': {'type': 'integer'},
                'status': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed']},
                'addBlockedBy': {
                    'type': 'array',
                    'items': {'type': 'integer'}
                },
                'removeBlockedBy': {
                    'type': 'array',
                    'items': {'type': 'integer'}
                },
            },
            required=['task_id']
        )
    ),
    'task_list': ToolParam(
        name='task_list',
        description='List all tasks with status summary.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={}
        )
    ),
    'task_get': ToolParam(
        name='task_get',
        description='Get full details of a task by ID.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={'task_id': {'type': 'string'}},
            required=['task_id']
        )
    ),
    'background_run': ToolParam(
        name='background_run',
        description='Run command in background thread. Returns task_id immediately.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={'command': {'type': 'string'}},
            required=['command']
        )
    ),
    'check_background': ToolParam(
        name='check_background',
        description='Check background task status. Omit task_id to list all.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={'task_id': {'type': 'string'}}
        )
    ),
    'subtask': ToolParam(
        name='subtask',
        description='Spawn a subagent with fresh context. It shares the filesystem but not conversation history.',
        input_schema=InputSchemaTyped(
            type='object',
            properties={
                'prompt': {'type': 'string'},
                'description': {'type': 'string', 'description': 'Short description of the task'}
            },
            required=['prompt']
        )
    )



}


