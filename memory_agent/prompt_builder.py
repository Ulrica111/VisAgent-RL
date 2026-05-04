"""
Prompt Builder — 将检索到的历史记忆拼接进模型输入的 Prompt
参考 HiMem 的层次化记忆增强策略
"""
from __future__ import annotations

from typing import Optional


SYSTEM_PROMPT = """你是一个视觉推理 Agent，擅长处理需要局部放大观察的视觉问答任务。
你可以调用 zoom_in 工具对图片的某个区域进行放大，以便看清细节后再作答。

zoom_in 工具调用格式：
zoom_in(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
其中坐标为归一化值 [0,1] 范围内的 左(x1)、上(y1)、右(x2)、下(y2)。

推理策略：
1. 先观察完整图片，判断哪些区域可能包含答案
2. 对模糊或需要细看的区域调用 zoom_in
3. 每次 zoom 后更新推理，决定是否继续 zoom 或给出最终答案
4. 最终答案用 <answer>答案内容</answer> 格式输出

注意：每次只调用一个 zoom_in，给出坐标后等待裁剪结果再决策。
"""


def get_system_prompt() -> str:
    """返回系统提示词。"""
    return SYSTEM_PROMPT


def build_initial_prompt(
    question: str,
    retrieved_memories: list[dict],
    max_memories: int = 3,
) -> str:
    """
    构建初始 user prompt（含记忆上下文）。

    Args:
        question: 当前问题
        retrieved_memories: MemoryRetriever.retrieve() 返回的列表
                            每项为 dict，含 'record'(MemoryRecord) 和 'score'(float)
        max_memories: 最多使用几条记忆

    Returns:
        完整 user prompt 字符串
    """
    memories = retrieved_memories[:max_memories]

    if not memories:
        return f"请回答以下问题：\n\n{question}"

    # 构建记忆提示段
    memory_lines = ["[历史推理经验 — 仅供参考，请根据当前图片独立判断]"]
    for i, mem_item in enumerate(memories, 1):
        hint = _format_memory_item(i, mem_item)
        memory_lines.append(hint)
    memory_lines.append("[历史经验结束]\n")

    memory_hint = "\n".join(memory_lines)

    return f"{memory_hint}\n请回答以下问题：\n\n{question}"


def build_step_prompt(step: int, zoom_trace: list[dict]) -> str:
    """
    构建后续 zoom 步骤的 prompt。

    Args:
        step: 当前步骤编号（从1开始）
        zoom_trace: 已执行的 zoom 记录列表

    Returns:
        步骤 prompt 字符串
    """
    lines = [f"[已完成 {step} 次 zoom 操作]"]
    for record in zoom_trace:
        bbox = record.get("bbox", [])
        reason = record.get("reason", "")
        lines.append(f"  步骤{record.get('step', '?')}: bbox={bbox}，原因：{reason}")
    lines.append("\n基于当前放大图片，请继续推理：是否需要继续 zoom？还是可以给出最终答案？")
    lines.append("如需继续 zoom，请输出 zoom_in(x1=..., y1=..., x2=..., y2=...) 并说明原因。")
    lines.append("如可作答，请用 <answer>答案</answer> 格式输出。")
    return "\n".join(lines)


def _format_memory_item(idx: int, mem_item: dict) -> str:
    """将单条检索结果格式化为可读的提示文本。"""
    record = mem_item.get("record")
    score = mem_item.get("score", 0.0)
    match_type = mem_item.get("match_type", "semantic")

    match_label = "（同一图片）" if match_type == "exact_image" else "（相似问题）"
    lines = [f"\n经验 {idx} {match_label}（相关度: {score:.2f}）："]

    if record:
        q = record.question
        lines.append(f"  问题类型：{q[:80]}{'...' if len(q) > 80 else ''}")

        if record.zoom_trace:
            lines.append(f"  有效的 zoom 策略（共 {len(record.zoom_trace)} 步）：")
            for step in record.zoom_trace[:2]:
                bbox = step.get("bbox", [])
                reason = str(step.get("reason", ""))[:100]
                if len(str(step.get("reason", ""))) > 100:
                    reason += "..."
                lines.append(f"    步骤{step.get('step', '?')}: bbox={bbox}，原因：{reason}")

        if record.answer:
            ans = record.answer
            lines.append(f"  历史答案：{ans[:60]}{'...' if len(ans) > 60 else ''}")

        if record.is_correct is not None:
            correctness = "✓ 正确" if record.is_correct else "✗ 错误"
            lines.append(f"  验证结果：{correctness}")

    return "\n".join(lines)
