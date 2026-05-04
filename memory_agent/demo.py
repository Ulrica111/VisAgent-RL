"""
PyVision-RL Memory Agent Demo
基于 HiMem 方案的视觉推理记忆增强 Demo

用法（交互模式）:
    python demo.py --image_path /path/to/image.jpg

用法（批量测试模式）:
    python demo.py --batch_test /path/to/test_cases.json

依赖:
    pip install requests pillow
"""

import argparse
import json
import os
import sys
import time

from visual_memory_agent import VisualMemoryAgent, AgentConfig


def run_interactive_demo(agent: VisualMemoryAgent, image_path: str) -> None:
    """交互式多轮问答 Demo"""
    print("\n" + "=" * 60)
    print("PyVision-RL 记忆增强视觉推理 Demo")
    print("=" * 60)
    print(f"图片路径: {image_path}")
    print("输入 'quit' 退出，输入 'stats' 查看记忆统计")
    print("=" * 60 + "\n")

    session_id = f"demo_session_{os.getpid()}"
    question_count = 0

    while True:
        question = input("请输入问题: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            print("退出 Demo")
            break
        if question.lower() == "stats":
            stats = agent.get_memory_stats()
            print(f"[记忆统计]\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n")
            continue

        question_count += 1
        print(f"\n[推理中... 第 {question_count} 个问题]")
        start_time = time.time()

        result = agent.run(
            image_path=image_path,
            question=question,
            session_id=session_id,
            verbose=True,
        )

        elapsed = time.time() - start_time
        print(f"\n{'─' * 40}")
        print(f"最终答案: {result['answer']}")
        print(f"Zoom 次数: {result['zoom_steps']}")
        print(f"记忆命中: {result['memory_hits']} 条")
        print(f"推理耗时: {elapsed:.2f}s")
        print(f"{'─' * 40}\n")


def run_batch_demo(agent: VisualMemoryAgent, test_cases_path: str) -> None:
    """批量测试 Demo，用于验证记忆效果（对比有无记忆）"""
    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\n批量测试: 共 {len(test_cases)} 个样本\n")
    results = []

    for i, case in enumerate(test_cases):
        image_path = case["image_path"]
        question = case["question"]
        ground_truth = case.get("answer", "")

        print(f"[{i+1}/{len(test_cases)}] 图片: {image_path}")
        print(f"  问题: {question}")

        start_time = time.time()
        result = agent.run(
            image_path=image_path,
            question=question,
            session_id=f"batch_session_{i // 5}",  # 每5个问题一个session
            ground_truth=ground_truth,
            verbose=False,
        )
        elapsed = time.time() - start_time

        record = {
            "image_path": image_path,
            "question": question,
            "answer": result["answer"],
            "zoom_steps": result["zoom_steps"],
            "memory_hits": result["memory_hits"],
            "ground_truth": ground_truth,
            "is_correct": result.get("is_correct"),
            "elapsed_sec": round(elapsed, 2),
        }
        results.append(record)

        correct_mark = ""
        if result.get("is_correct") is True:
            correct_mark = " ✓"
        elif result.get("is_correct") is False:
            correct_mark = " ✗"

        print(f"  答案: {result['answer'][:80]}{correct_mark}")
        print(f"  Zoom次数: {result['zoom_steps']} | 记忆命中: {result['memory_hits']} | 耗时: {elapsed:.2f}s\n")

    # 统计结果
    total = len(results)
    memory_hit_questions = sum(1 for r in results if r["memory_hits"] > 0)
    avg_zoom = sum(r["zoom_steps"] for r in results) / total if total > 0 else 0
    correct_cases = [r for r in results if r["is_correct"] is not None]
    accuracy = sum(1 for r in correct_cases if r["is_correct"]) / len(correct_cases) if correct_cases else None

    print("=" * 60)
    print("批量测试完成:")
    print(f"  总样本数: {total}")
    print(f"  记忆命中率: {memory_hit_questions/total*100:.1f}% ({memory_hit_questions}/{total})")
    print(f"  平均 Zoom 次数: {avg_zoom:.2f}")
    if accuracy is not None:
        print(f"  答案准确率: {accuracy*100:.1f}% ({sum(1 for r in correct_cases if r['is_correct'])}/{len(correct_cases)})")
    print("=" * 60)

    # 保存结果
    output_path = "batch_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PyVision-RL 记忆增强视觉推理 Demo")
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000",
        help="vLLM 服务地址（默认: http://localhost:8000）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="模型名称（需与 vLLM 启动时一致）",
    )
    parser.add_argument(
        "--memory_db_path",
        type=str,
        default="memory_agent/memory_db.json",
        help="记忆数据库文件路径",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="图片路径（交互模式）",
    )
    parser.add_argument(
        "--batch_test",
        type=str,
        default=None,
        help="批量测试 JSON 文件路径（批量模式）",
    )
    parser.add_argument(
        "--max_zoom_steps",
        type=int,
        default=3,
        help="最大 zoom 步数（默认: 3）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="检索历史记忆条数（默认: 3）",
    )
    parser.add_argument(
        "--vllm_max_image_edge",
        type=int,
        default=768,
        help="发给 vLLM 前图像最长边上限（像素），降低视觉 token；若仍超长可改为 512 或提高服务端 max_model_len（默认: 768）",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="单次回复 max_tokens（上限仍受服务端 max_model_len - prompt 占用约束，默认: 1024）",
    )
    args = parser.parse_args()

    # 初始化 Agent
    config = AgentConfig(
        vllm_base_url=args.vllm_base_url,
        model_name=args.model_name,
        memory_db_path=args.memory_db_path,
        max_zoom_steps=args.max_zoom_steps,
        retrieval_top_k=args.top_k,
        vllm_max_image_edge=args.vllm_max_image_edge,
        max_tokens=args.max_tokens,
    )
    agent = VisualMemoryAgent(config=config)

    print(f"记忆数据库: {args.memory_db_path}")
    stats = agent.get_memory_stats()
    print(f"当前记忆条数: {stats.get('total_records', 0)}")

    if args.batch_test:
        run_batch_demo(agent, args.batch_test)
    elif args.image_path:
        if not os.path.exists(args.image_path):
            print(f"错误: 图片文件不存在: {args.image_path}")
            sys.exit(1)
        run_interactive_demo(agent, args.image_path)
    else:
        print("请指定 --image_path（交互模式）或 --batch_test（批量测试模式）")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
