import os
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam, ToolResultBlockParam
from dotenv import load_dotenv

from dispatcher import TOOL_PARAMS, TOOL_HANDLERS

# ==============================================================================
# 配置
# ==============================================================================
load_dotenv(override=True)
WORKDIR = Path.cwd()
MODEL = os.getenv('MODEL_ID')
client = Anthropic(base_url=os.getenv('ANTHROPIC_BASE_URL'))

# ==============================================================================
# System prompt & 工具列表
# ==============================================================================
SYSTEM = f"""
You are a coding agent at {WORKDIR}. Use tools to solve tasks.
""".strip()
TOOLS = [
    TOOL_PARAMS['bash'],
    TOOL_PARAMS['read_file'],
    TOOL_PARAMS['write_file'],
    TOOL_PARAMS['edit_file']
]

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
    while True:
        # --- 获取模型响应，主体是一系列block ---
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append(MessageParam(role='assistant', content=response.content))

        # --- 处理block ---
        results = []
        for block in response.content:
            if block.type == 'text':
                _process_text(block)
            elif block.type == 'tool_use':
                output = _process_tool(block)
                results.append(ToolResultBlockParam(type='tool_result', tool_use_id=block.id, content=output))
        messages.append(MessageParam(role='user', content=results))

        # --- 收尾 ---
        if response.stop_reason != 'tool_use':  # 也就是end_turn，表明模型完成任务
            return


if __name__ == '__main__':
    history: list[MessageParam] = []
    while True:
        # --- 获取当前输入 ---
        try:
            query = input('\x01\033[36m【user】\033[0m\x02').encode('utf-8', 'replace').decode('utf-8')
            if query.strip().lower() in ('q', 'quit', 'e', 'exit', ''):
                break
            history.append(MessageParam(role='user', content=query))
        except (EOFError, KeyboardInterrupt):
            break
        # --- 交给agent执行
        agent_loop(history)

