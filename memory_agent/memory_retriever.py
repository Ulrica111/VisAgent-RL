"""
HiMem-style Memory Retriever
负责从记忆库中检索与当前问题最相关的历史记录
策略：基于图片 hash 精确匹配 + 问题 TF-IDF 余弦相似度
"""

import math
import re
from collections import Counter
from typing import Optional

from memory_store import MemoryRecord, MemoryStore


def _tokenize(text: str) -> list[str]:
    """简单分词：转小写 + 按非字母数字切分（支持中文字符）"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text)
    return tokens


def _tf(tokens: list[str]) -> dict[str, float]:
    """计算词频 TF"""
    counter = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {word: count / total for word, count in counter.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """计算两个词频向量的余弦相似度"""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0

    dot = sum(vec_a[w] * vec_b[w] for w in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class MemoryRetriever:
    """
    记忆检索器 —— 实现 HiMem 风格的分层检索：
    1. 精确匹配：同一张图片（image_hash 相同）优先，基础分 +0.5
    2. 语义匹配：问题文本的 TF-IDF 余弦相似度
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def retrieve(
        self,
        image_hash: str,
        question: str,
        top_k: int = 3,
        min_similarity: float = 0.1,
    ) -> list[dict]:
        """
        检索最相关的历史记忆

        Args:
            image_hash: 当前图片的 hash
            question: 当前问题文本
            top_k: 返回最多几条记忆
            min_similarity: 最小相似度阈值（仅对非精确匹配生效）

        Returns:
            list of dict，每个元素包含：
                - record: MemoryRecord
                - score: float
                - match_type: "exact_image" | "semantic"
        """
        all_records = self.memory_store.get_all_records()
        if not all_records:
            return []

        query_tokens = _tokenize(question)
        query_tf = _tf(query_tokens)

        scored: list[tuple[MemoryRecord, float, str]] = []

        for record in all_records:
            record_tokens = _tokenize(record.question)
            record_tf = _tf(record_tokens)
            sim = _cosine_similarity(query_tf, record_tf)

            if record.image_hash == image_hash:
                # 精确图片匹配：同图片加基础分 0.5
                score = 0.5 + 0.5 * sim
                scored.append((record, score, "exact_image"))
            elif sim >= min_similarity:
                # 语义匹配：不同图片但问题类型相似
                scored.append((record, sim, "semantic"))

        # 按分数降序，只保留 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return [
            {
                "record": record,
                "score": score,
                "match_type": match_type,
            }
            for record, score, match_type in top
        ]

    def format_memory_prompt(self, retrieved: list[dict]) -> str:
        """
        将检索到的记忆格式化为 Prompt 增强片段

        Args:
            retrieved: retrieve() 的返回值列表

        Returns:
            str: 插入到 prompt 中的历史记忆提示文本，若无记忆则返回空字符串
        """
        if not retrieved:
            return ""

        lines = ["[历史记忆参考]"]
        lines.append("以下是我处理过的相似问题的推理经验，可作参考：\n")

        for i, item in enumerate(retrieved, 1):
            record: MemoryRecord = item["record"]
            score: float = item["score"]
            match_type: str = item["match_type"]

            match_label = "（同一图片）" if match_type == "exact_image" else "（相似问题）"
            lines.append(f"--- 参考记忆 {i} {match_label} 相似度={score:.3f} ---")
            lines.append(f"历史问题：{record.question}")

            if record.zoom_trace:
                lines.append("有效的 zoom 策略：")
                for step in record.zoom_trace:
                    bbox = step.get("bbox", [])
                    reason = step.get("reason", "")
                    lines.append(f"  步骤 {step.get('step', '?')}: bbox={bbox}，原因：{reason}")

            lines.append(f"最终答案：{record.answer}")

            if record.answer_correct is not None:
                correctness = "✓ 正确" if record.answer_correct else "✗ 错误"
                lines.append(f"验证结果：{correctness}")

            lines.append("")  # 空行分隔

        lines.append("[历史记忆参考结束]")
        return "\n".join(lines)
