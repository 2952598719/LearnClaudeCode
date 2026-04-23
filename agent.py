import os
import json
import time
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

from dispatcher import TOOL_PARAMS, TOOL_HANDLERS


# ==============================================================================
# 配置
# ==============================================================================
# --- 基础参数 ---
load_dotenv(override=True)
WORKDIR = Path.cwd()
MODEL = os.getenv('MODEL_ID')
client = Anthropic(base_url=os.getenv('ANTHROPIC_BASE_URL'))
RESPONSE_MAX_LEN = 8000
# --- 压缩参数 ---
COMPACT_THRESHOLD_LEN = 50000
COMPACT_TRUNCATE_LEN = 80000
COMPACT_RESULT_MAX_LEN = 2000
TRANSCRIPT_DIR = WORKDIR / '.transcripts'
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {'read_file'}
# 提示参数
NAG_INTERVAL = 3         # 超过此轮数未调用待办更新则提醒模型
# --- System prompt & 工具列表 ---
SYSTEM = f"""
你是一个编程agent，位于{WORKDIR}，使用提供的工具去解决任务。
使用todo工具规划多步骤任务。启动前和完成后分别标记。
""".strip()
TOOLS = [
    TOOL_PARAMS['bash'],
    TOOL_PARAMS['read_file'],
    TOOL_PARAMS['write_file'],
    TOOL_PARAMS['edit_file'],
    TOOL_PARAMS['compact'],
    TOOL_PARAMS['todo']
]

# ==============================================================================
# 压缩函数
# ==============================================================================
def estimate_tokens(messages: list) -> int:
    # 1汉字≈1token，4字母≈1token
    text = str(messages)
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - cn_chars
    return cn_chars + other_chars // 4

def micro_compact(messages: list) -> list:
    # messages: (1) 用户-输入，字符串 (2) 模型，列表-Text或Text+Tool×n (3) 用户-调用，列表-ToolResult×n
    # 则我们的目标是压缩(3)中，3条以外的非PRESERVE_RESULT_TOOLS工具调用结果，同时如果不是字符串或者比较短也会被保留

    # --- 得到(2)所有Tool ---
    tool_name_map = {}
    for msg in messages:
        if msg['role'] == 'assistant' and isinstance(msg['content'], list):
            for block in msg['content']:    # TextBlock或ToolUseBlock，不能作为dict
                if hasattr(block, 'type') and block.type == 'tool_use':
                    tool_name_map[block.id] = block.name

    # --- 收集(3)所有ToolResult ---
    # 用户-输入可以扩展成支持图片、引用的形式，则此时就变成list了。因此list本身并不起到核心判断作用，只不过是优化，part['type']才是核心约束
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg['role'] == 'user' and isinstance(msg['content'], list):
            for part in msg['content']:
                if isinstance(part, dict) and part['type'] == 'tool_result':  # 不是tool_name_map中的Block，而是dict
                    tool_results.append(part)

    # --- 遍历处理 ---
    if len(tool_results) <= KEEP_RECENT:
        return messages
    to_clear = tool_results[:-KEEP_RECENT]
    for part in to_clear:
        if not isinstance(part['content'], str) or len(part['content']) <= 100:
            continue
        tool_id = part['tool_use_id']
        tool_name = tool_name_map.get(tool_id, 'unknown')
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue
        part['content'] = f'[历史记录: 使用了 {tool_name} 工具]'
    return messages

# 作为工具是命令行下的权宜之计，按理来说应该在ui界面中点击压缩按钮后执行，而不是让agent调用
def auto_compact(messages: list) -> tuple[bool, list]:
    # --- 保存messages到磁盘文件 ---
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f'transcript_{int(time.time())}.jsonl'
    with open(transcript_path, 'w') as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + '\n')
    print(f'[transcript saved: {transcript_path}]')
    # --- 交由llm压缩 ---
    conversation_text = json.dumps(messages, default=str)[-COMPACT_TRUNCATE_LEN:]
    response = client.messages.create(
        model=MODEL,
        messages=[MessageParam(
            role='user',
            content='总结对话内容，包含：1) 已完成的工作, 2) 当前状态, 3) 做出的关键决策'
                    '简洁但保留关键细节。\n\n' + conversation_text
        )],
        max_tokens=COMPACT_RESULT_MAX_LEN
    )
    summary = next((block.text for block in response.content if hasattr(block, 'text')), '')

    if not summary:
        summary = '没有生成总结'
        return False, [summary]
    else:
        return True, [{
            'role': 'user',
            'content': f'[对话压缩成功. 文件: {transcript_path}]\n\n{summary}'
        }]

# ==============================================================================
# block处理
# ==============================================================================
def _process_tool(block):
    # --- 获取工具 ---
    print(f'\033[32m【agent】\033[0m\033[33m{block.name}: {block.input}\033[0m')
    handler = TOOL_HANDLERS.get(block.name)
    # --- 执行输出 ---
    try:
        output = handler(**block.input) if handler else f'Unknown tool: {block.name}'
    except Exception as e:
        output = f'Error: {e}'
    print(output if len(output) <= 200 else f'{output[:200]}... ({len(output) - 200} chars left)')
    print('-' * 64)
    return output

def _process_text(block):
    print(f'\033[32m【agent】\033[0m{block.text}')

# ==============================================================================
# 主 Agent 循环
# ==============================================================================
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        # --- 将消息上下文进行压缩 ---
        micro_compact(messages)
        if estimate_tokens(messages) > COMPACT_THRESHOLD_LEN:
            print('[auto compact triggered]')
            success, new_messages = auto_compact(messages)
            if success:
                messages[:] = new_messages  # 深替换，不能去掉[:]
        # --- 获取模型响应，主体是一系列block ---
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=RESPONSE_MAX_LEN,
        )
        messages.append({'role': 'assistant', 'content': response.content})
        # --- 处理block ---
        if response.stop_reason == 'tool_use':  # response.stop_reason = tool_use时，content = TextBlock × 0-1 + ToolBlock × 1-n
            manual_compact = False
            using_todo = False
            results = []
            for block in response.content:
                if block.type == 'tool_use':
                    if block.name == 'compact': manual_compact = True
                    elif block.name == 'todo': using_todo = True
                    output = _process_tool(block)
                    results.append({'type': 'tool_result', 'tool_use_id': block.id, 'content': str(output)})
                elif block.type == 'text':
                    _process_text(block)
            # --- 未更新待办时的提醒 ---
            rounds_since_todo = 0 if using_todo else rounds_since_todo + 1
            if rounds_since_todo >= NAG_INTERVAL:
                results.append({'type': 'text', 'text': '<reminder>如果已有todo列表，请更新列表</reminder>'})
            messages.append({'role': 'user', 'content': results})
            # --- 执行压缩 ---
            if manual_compact:
                print('[manual compact triggered]')
                success, new_messages = auto_compact(messages)
                if success:
                    messages[:] = new_messages
        elif response.stop_reason == 'end_turn':      # response.stop_reason = end_turn时，content = TextBlock × 1
            block = response.content[0]
            _process_text(block)
            break

if __name__ == '__main__':
    history: list = []
    while True:
        # --- 获取当前输入 ---
        try:
            query = input('\x01\033[36m【user】\033[0m\x02').encode('utf-8', 'replace').decode('utf-8')
            if query.strip().lower() in ('q', 'quit', 'e', 'exit', ''):
                break
            history.append({'role': 'user', 'content': query})
        except (EOFError, KeyboardInterrupt):
            break
        # --- 交给agent执行
        agent_loop(history)
