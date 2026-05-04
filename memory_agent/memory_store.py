"""
HiMem 风格的记忆存储模块
分为短期记忆（当前 session）和长期记忆（跨 session 持久化）
"""
import json
import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class MemoryRecord:
    """一条完整的推理记忆记录"""
    record_id: str                  # 唯一ID
    image_hash: str                 # 图片内容哈希（MD5）
    image_path: str                 # 原始图片路径（方便调试）
    question: str                   # 原始问题
    zoom_trace: List[dict]          # zoom 操作链，每项含 step/bbox/reason
    answer: str                     # 最终答案
    session_id: str                 # session 标识
    timestamp: str                  # ISO 8601 时间戳
    is_correct: Optional[bool] = None   # 答案是否正确（有 ground truth 时填写）
    confidence: float = 1.0         # 置信度

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())),
            image_hash=d["image_hash"],
            image_path=d.get("image_path", ""),
            question=d["question"],
            zoom_trace=d.get("zoom_trace", []),
            answer=d["answer"],
            session_id=d.get("session_id", ""),
            timestamp=d.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            is_correct=d.get("is_correct", None),
            confidence=d.get("confidence", 1.0),
        )

    # 兼容性别名：部分模块用 final_answer / answer_correct
    @property
    def final_answer(self) -> str:
        return self.answer

    @property
    def answer_correct(self) -> Optional[bool]:
        return self.is_correct


def compute_image_hash(image_bytes: bytes) -> str:
    """计算图片的 MD5 哈希，用于精确匹配"""
    return hashlib.md5(image_bytes).hexdigest()


class MemoryStore:
    """
    统一的记忆管理接口
    - 运行时在内存中维护记录列表
    - 每次 add 后自动持久化到 JSON 文件
    """

    def __init__(self, db_path: str = "memory_agent/data/memory_db.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[MemoryRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # 静态工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def compute_image_hash(image_bytes: bytes) -> str:
        """计算图片的 MD5 哈希"""
        return compute_image_hash(image_bytes)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_record(
        self,
        image_hash: str,
        image_path: str,
        question: str,
        zoom_trace: List[dict],
        answer: str,
        session_id: str = "default",
        is_correct: Optional[bool] = None,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        """构造并保存一条新记忆记录，返回该记录"""
        record = MemoryRecord(
            record_id=str(uuid.uuid4()),
            image_hash=image_hash,
            image_path=image_path,
            question=question,
            zoom_trace=zoom_trace,
            answer=answer,
            session_id=session_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            is_correct=is_correct,
            confidence=confidence,
        )
        self._records.append(record)
        self._save()
        return record

    def get_all(self) -> List[MemoryRecord]:
        """返回所有记忆记录"""
        return list(self._records)

    # 兼容性别名
    def get_all_records(self) -> List[MemoryRecord]:
        return self.get_all()

    def get_by_image(self, image_hash: str) -> List[MemoryRecord]:
        """按图片哈希过滤"""
        return [r for r in self._records if r.image_hash == image_hash]

    def count(self) -> int:
        return len(self._records)

    def get_stats(self) -> dict:
        return {
            "total_records": self.count(),
            "db_path": str(self.db_path),
        }

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._records = [MemoryRecord.from_dict(d) for d in raw]
            except (json.JSONDecodeError, KeyError):
                self._records = []
        else:
            self._records = []

    def _save(self) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self._records], f, ensure_ascii=False, indent=2)
