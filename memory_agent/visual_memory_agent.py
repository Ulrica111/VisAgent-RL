"""
Visual Memory Agent - 核心 Agent 推理链路
整合记忆存储、检索、Prompt 增强和 VLLM 推理
采用 HiMem 风格的层次化记忆管理方案
"""

import re
import json
import base64
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import requests

from memory_store import MemoryStore, MemoryRecord, compute_image_hash
from memory_retriever import MemoryRetriever
from prompt_builder import build_initial_prompt, get_system_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Agent 配置"""
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    memory_db_path: str = "memory_agent/data/memory_db.json"
    max_zoom_steps: int = 3
    retrieval_top_k: int = 3
    temperature: float = 0.1
    max_tokens: int = 1024
    # 发送给 vLLM 前将图像最长边限制到此值（像素），显著降低 Qwen-VL 视觉 token，适配较小 max_model_len
    vllm_max_image_edge: int = 768
    # user 消息纯文本超过此长度时截断（不含图片），防止记忆/历史把 prompt 撑爆
    vllm_max_user_text_chars: int = 5500


class VLLMClient:
    """VLLM 推理客户端（对接 OpenAI 兼容接口）"""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def chat(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> str:
        """调用 VLLM 的 /v1/chat/completions 接口"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            finish = choice.get("finish_reason")
            if finish == "length":
                usage = data.get("usage") or {}
                logger.warning(
                    "vLLM 因长度上限停止生成（finish_reason=length），回答可能被截断。"
                    " usage=%s。若常出现：请提高服务端 --max-model-len，或缩小图像/提示词；"
                    "客户端 max_tokens=%s 不会超过「max_model_len - prompt 占用」。",
                    usage,
                    max_tokens,
                )
            return choice["message"]["content"]
        except requests.exceptions.HTTPError as e:
            detail = ""
            if e.response is not None:
                detail = (e.response.text or "")[:4000]
            logger.error("VLLM HTTP 错误 %s: %s", e.response.status_code if e.response else "?", detail)
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"VLLM 请求失败: {e}")
            raise


class ZoomToolExecutor:
    """Zoom 工具执行器"""

    # 匹配格式：zoom_in(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
    ZOOM_PATTERN = re.compile(
        r"zoom_in\s*\(\s*x1\s*=\s*([\d.]+)\s*,\s*y1\s*=\s*([\d.]+)\s*,"
        r"\s*x2\s*=\s*([\d.]+)\s*,\s*y2\s*=\s*([\d.]+)\s*\)"
    )
    # 也支持 bbox=[x1,y1,x2,y2] 格式
    BBOX_PATTERN = re.compile(
        r"bbox\s*=\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
    )

    def __init__(self, max_zoom_steps: int = 3):
        self.max_zoom_steps = max_zoom_steps

    def parse_tool_call(self, model_output: str) -> Optional[tuple]:
        """从模型输出中解析 zoom_in 调用的坐标"""
        for pattern in [self.ZOOM_PATTERN, self.BBOX_PATTERN]:
            match = pattern.search(model_output)
            if match:
                x1, y1, x2, y2 = [float(v) for v in match.groups()]
                if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                    return x1, y1, x2, y2
        return None

    def execute_zoom(self, image_bytes: bytes, bbox: tuple) -> bytes:
        """对图片执行 zoom_in 裁剪操作"""
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            x1, y1, x2, y2 = bbox
            crop_box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
            cropped = img.crop(crop_box)

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            logger.warning("Pillow 未安装，返回原始图片")
            return image_bytes

    def is_final_answer(self, model_output: str) -> bool:
        """判断模型是否已经给出最终答案"""
        has_zoom = (
            self.ZOOM_PATTERN.search(model_output) is not None
            or self.BBOX_PATTERN.search(model_output) is not None
        )
        has_answer_marker = any(
            marker in model_output
            for marker in ["最终答案", "Final Answer", "答案是", "The answer is", "<answer>"]
        )
        return has_answer_marker or not has_zoom


def downscale_image_for_vllm(
    image_bytes: bytes,
    max_edge: int,
) -> tuple[bytes, str]:
    """
    将图像最长边限制在 max_edge 以内并导出为 JPEG，降低 Qwen2.5-VL 等模型的视觉 token 数。
    max_edge <= 0 时不缩放（仍转为 JPEG 以减小体积）。
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        logger.warning("Pillow 未安装，跳过图像缩放")
        return image_bytes, "image/png"

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max_edge > 0 and max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((w, h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue(), "image/jpeg"


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def build_vision_message(prompt_text: str, image_bytes: bytes, image_mime: str = "image/png") -> dict:
    """构造包含图片的 user message（OpenAI vision 格式）"""
    image_b64 = image_to_base64(image_bytes)
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}
            },
            {"type": "text", "text": prompt_text}
        ]
    }


class VisualMemoryAgent:
    """
    视觉记忆 Agent
    实现 HiMem 风格的两层记忆管理：
    - 短期记忆：当前 session 内的推理轨迹（in-context）
    - 长期记忆：跨 session 的持久化 zoom 策略（external storage）
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        memory_db_path: str = "memory_agent/data/memory_db.json",
        max_zoom_steps: int = 3,
        retrieval_top_k: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        vllm_max_image_edge: int = 768,
        vllm_max_user_text_chars: int = 5500,
        config: Optional[AgentConfig] = None,
    ):
        if config is None:
            config = AgentConfig(
                vllm_base_url=vllm_base_url,
                model_name=model_name,
                memory_db_path=memory_db_path,
                max_zoom_steps=max_zoom_steps,
                retrieval_top_k=retrieval_top_k,
                temperature=temperature,
                max_tokens=max_tokens,
                vllm_max_image_edge=vllm_max_image_edge,
                vllm_max_user_text_chars=vllm_max_user_text_chars,
            )
        self.config = config

        self.memory_store = MemoryStore(db_path=config.memory_db_path)
        self.memory_retriever = MemoryRetriever(memory_store=self.memory_store)
        self.vllm_client = VLLMClient(
            base_url=config.vllm_base_url,
            model_name=config.model_name,
        )
        self.zoom_executor = ZoomToolExecutor(max_zoom_steps=config.max_zoom_steps)

        logger.info(
            f"VisualMemoryAgent 初始化完成，记忆库路径: {config.memory_db_path}"
        )

    def run(
        self,
        image_path: str,
        question: str,
        session_id: str = "default",
        ground_truth: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        与 reason() 相同，返回字段额外包含 demo / 脚本兼容键：
        zoom_steps、memory_hits。
        """
        out = self.reason(
            image_path=image_path,
            question=question,
            session_id=session_id,
            ground_truth=ground_truth,
            verbose=verbose,
        )
        merged = dict(out)
        merged["zoom_steps"] = out["zoom_count"]
        merged["memory_hits"] = out.get("retrieved_count", 0)
        return merged

    def reason(
        self,
        image_path: str,
        question: str,
        session_id: str = "default",
        ground_truth: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        执行一次完整的视觉推理（含记忆增强）

        Returns:
            dict: 包含 answer, zoom_count, memory_hit, retrieved_memory 等字段
        """
        # 1. 读取图片
        image_bytes = Path(image_path).read_bytes()
        image_hash = compute_image_hash(image_bytes)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[Agent] 开始推理")
            print(f"  图片: {image_path}")
            print(f"  问题: {question}")
            print(f"  图片 hash: {image_hash[:8]}...")

        # 2. 从长期记忆检索相关历史
        retrieved = self.memory_retriever.retrieve(
            image_hash=image_hash,
            question=question,
            top_k=self.config.retrieval_top_k,
        )
        memory_hit = len(retrieved) > 0

        if verbose and retrieved:
            print(f"\n[记忆] 检索到 {len(retrieved)} 条历史记忆:")
            for i, item in enumerate(retrieved):
                record = item["record"]
                score = item["score"]
                match_type = item["match_type"]
                print(f"  [{i+1}] {match_type} 相似度={score:.3f} | zoom步数={len(record.zoom_trace)}")

        # 3. 构建初始 Prompt（含记忆上下文），与 retrieve() 返回结构一致
        retrieved_memories = retrieved
        initial_prompt = build_initial_prompt(
            question=question,
            retrieved_memories=retrieved_memories,
        )

        # 4. 多步推理（每轮请求只带「当前」一张图，避免多轮里堆叠多图导致超过 max_model_len）
        system_msg = {"role": "system", "content": get_system_prompt()}
        current_image = image_bytes
        zoom_trace: list = []
        prior_assistant_outputs: list[str] = []
        final_answer = ""

        for step in range(self.config.max_zoom_steps + 1):
            if step == 0:
                step_prompt = initial_prompt
            else:
                step_prompt = (
                    f"[步骤 {step + 1}] 你刚才对区域 {zoom_trace[-1]['bbox']} 进行了放大。"
                    "请观察放大后的图片，决定是否继续 zoom 或给出最终答案。"
                )

            if step == 0:
                user_text = step_prompt
            else:
                history_chunks = []
                for i, prev in enumerate(prior_assistant_outputs):
                    chunk = prev.strip()
                    if len(chunk) > 8000:
                        chunk = chunk[:8000] + "\n…[输出过长已截断]"
                    history_chunks.append(f"[第 {i + 1} 步模型输出]\n{chunk}")
                history_block = "\n\n".join(history_chunks)
                user_text = (
                    "【任务与记忆背景（与首轮相同）】\n"
                    f"{initial_prompt}\n\n"
                    "【此前在同一原图上的推理输出】\n"
                    f"{history_block}\n\n"
                    "【当前轮次】\n"
                    f"{step_prompt}\n\n"
                    "说明：本消息所附图片为上一轮所选区域的放大结果，请仅基于该图继续推理。"
                )

            budget = self.config.vllm_max_user_text_chars
            if budget > 0 and len(user_text) > budget:
                user_text = user_text[:budget] + "\n…[文本过长已截断]"

            api_img, api_mime = downscale_image_for_vllm(
                current_image,
                self.config.vllm_max_image_edge,
            )
            messages = [
                system_msg,
                build_vision_message(user_text, api_img, image_mime=api_mime),
            ]

            if verbose:
                print(f"\n[步骤 {step+1}] 调用模型...")

            model_output = self.vllm_client.chat(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            if verbose:
                print(f"[模型输出]\n{model_output[:300]}{'...' if len(model_output) > 300 else ''}")

            prior_assistant_outputs.append(model_output)

            if self.zoom_executor.is_final_answer(model_output):
                final_answer = model_output
                if verbose:
                    print(f"\n[Agent] 已得出最终答案，停止推理")
                break

            bbox = self.zoom_executor.parse_tool_call(model_output)
            if bbox is None:
                if verbose:
                    print(f"\n[Agent] 未检测到有效 zoom 调用，停止推理")
                final_answer = model_output
                break

            zoom_record = {
                "step": step + 1,
                "bbox": list(bbox),
                "reason": self._extract_reason(model_output),
            }
            zoom_trace.append(zoom_record)

            if verbose:
                print(f"[Zoom] 执行裁剪: bbox={bbox}, reason={zoom_record['reason']}")

            current_image = self.zoom_executor.execute_zoom(current_image, bbox)

        # 达到 max_zoom_steps 上限仍只输出 zoom、未 break 时，final_answer 从未赋值，需兜底
        if not (final_answer or "").strip() and prior_assistant_outputs:
            final_answer = prior_assistant_outputs[-1]
            if verbose:
                print(
                    "\n[Agent] 已达最大 zoom 步数，未收到明确终答；使用最后一步模型输出作为兜底。"
                )

        # 5. 提取答案文本
        answer_text = self._extract_answer(final_answer)

        is_correct: Optional[bool] = None
        if ground_truth is not None and str(ground_truth).strip() != "":
            is_correct = self._answer_matches_ground_truth(answer_text, ground_truth)

        # 6. 写入长期记忆
        record = self.memory_store.add_record(
            image_hash=image_hash,
            image_path=image_path,
            question=question,
            zoom_trace=zoom_trace,
            answer=answer_text,
            session_id=session_id,
            is_correct=is_correct,
        )

        if verbose:
            print(f"\n[记忆] 已写入长期记忆 (ID={record.record_id[:8]}...)")
            print(f"  zoom 步数: {len(zoom_trace)}")
            print(f"  记忆库总记录数: {self.memory_store.count()}")

        return {
            "answer": answer_text,
            "zoom_count": len(zoom_trace),
            "memory_hit": memory_hit,
            "retrieved_count": len(retrieved),
            "retrieved_memory": {
                "zoom_trace": retrieved[0]["record"].zoom_trace if retrieved else [],
                "score": retrieved[0]["score"] if retrieved else 0.0,
            } if retrieved else None,
            "record_id": record.record_id,
            "full_output": final_answer,
            "is_correct": is_correct,
        }

    @staticmethod
    def _answer_matches_ground_truth(answer: str, ground_truth: str) -> bool:
        a = re.sub(r"\s+", " ", answer.strip().lower())
        b = re.sub(r"\s+", " ", str(ground_truth).strip().lower())
        return a == b

    def _extract_reason(self, model_output: str) -> str:
        lines = model_output.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and "zoom_in" not in line and "bbox" not in line:
                return line[:100]
        return "未说明原因"

    def _extract_answer(self, model_output: str) -> str:
        if not (model_output or "").strip():
            return ""
        match = re.search(r"<answer>(.*?)</answer>", model_output, re.DOTALL)
        if match:
            inner = match.group(1).strip()
            if inner:
                return inner

        patterns = [
            r"最终答案[：:]\s*(.+)",
            r"Final Answer[：:]\s*(.+)",
            r"答案[是为][：:]\s*(.+)",
            r"The answer is[：:]\s*(.+)",
            r"答案[：:]\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, model_output, re.IGNORECASE)
            if match:
                got = match.group(1).strip()
                if got:
                    return got

        lines = [l.strip() for l in model_output.strip().split("\n") if l.strip()]
        return lines[-1] if lines else model_output.strip()

    def clear_session_memory(self, session_id: str) -> None:
        """清空指定 session 的短期记忆（长期记忆不受影响）"""
        logger.info(f"清空 session {session_id} 的记忆（长期记忆保留）")

    def get_memory_stats(self) -> dict:
        return self.memory_store.get_stats()
