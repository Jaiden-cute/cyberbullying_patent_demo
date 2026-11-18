"""Replay the prototype from Windows PowerShell or another Python 3.12 environment."""

import argparse
import os
from pathlib import Path
import sqlite3
import sys

from dotenv import dotenv_values
import yaml

from .config import load_settings
from .history_store import HistoryStore
from .input_records import DEMO_FEATURES, DEMO_MESSAGES, parse_features, parse_messages, parse_reviews
from .pipeline import run_pipeline
from .reporting import render_report
from .runtime import load_runtime

PROJECT_DIR = Path(__file__).resolve().parents[1]


def media_directory() -> Path | None:
    value = os.environ.get("PATENT_ALLOWED_MEDIA_DIR") or dotenv_values(PROJECT_DIR / ".env").get("PATENT_ALLOWED_MEDIA_DIR")
    return Path(value) if value else None


def _show_summary(report: dict) -> None:
    def number(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "未计算"

    print(f"{report['run_mode']} · 事件数 {len(report['events'])}")
    for row in report["events"]:
        event, severity, masking, recurrence, risk = (row[key] for key in ("event", "S1", "S4", "S5", "S6"))
        print(f"{event.start_time.isoformat()} [{event.victim_id}] {'/'.join(event.message_ids)}")
        for item in row["candidates"]:
            if item.S3_masking is None:
                print(f"  {item.S2.message.message_id}: A={number(item.S2.A)} ({item.S2.reason})")
            else:
                print(
                    f"  {item.S2.message.message_id}: A={number(item.S2.A)} "
                    f"M={number(item.S3_masking.M)} N={number(item.S3_nonverbal.N)} "
                    f"H={number(item.S4.H if item.S4 else None)} ({item.S4.branch if item.S4 else 'unavailable'})"
                )
        print(
            f"  B={number(severity.B)} H_i={number(masking['H'])} "
            f"P={number(recurrence.P)} ({recurrence.P_status}) F={number(recurrence.F)} "
            f"U={number(risk.U)} {risk.risk_level or '未分级'} "
            f"代表回应={masking['representative_reply_id'] or '无'}"
            + (" [人工复核]" if risk.human_review_required else "")
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行网络欺凌专利原型")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.yaml")
    parser.add_argument("--messages", type=Path, default=DEMO_MESSAGES)
    modalities = parser.add_mutually_exclusive_group()
    modalities.add_argument("--features", type=Path, help="结构化非语言测试观测 JSON")
    modalities.add_argument("--text-only", action="store_true", help="执行纯文本路径")
    parser.add_argument("--reviews", type=Path, help="人工事件核对输入 JSON")
    parser.add_argument("--mode", choices=("demo", "trained"), default="demo")
    parser.add_argument("--strict-model", action="store_true", help="模型缺失时文本模型项保留未计算")
    parser.add_argument("--no-qwen", action="store_true", help="忽略 Qwen 端点配置")
    parser.add_argument("--history", type=Path, help="使用指定 SQLite 文件读取和保存历史")
    parser.add_argument("--output", type=Path, help="完整计算报告 JSON 路径，默认打印到标准输出")
    args = parser.parse_args(argv)

    feature_path = args.features
    if feature_path is None and not args.text_only and args.messages.resolve() == DEMO_MESSAGES:
        feature_path = DEMO_FEATURES
    try:
        input_paths = (args.config, args.messages, feature_path, args.reviews, args.history)
        if args.output is not None and args.output.resolve() in {path.resolve() for path in input_paths if path}:
            raise ValueError("报告路径必须与输入文件路径不同")
        settings = load_settings(args.config)
        messages = parse_messages(args.messages.read_bytes(), args.messages.name)
        features = parse_features(feature_path.read_bytes()) if feature_path else []
        reviews = parse_reviews(args.reviews.read_bytes()) if args.reviews else []
        runtime = load_runtime(settings, requested=args.mode, strict=args.strict_model, use_qwen=not args.no_qwen)
        analyze = HistoryStore(args.history).analyze if args.history else run_pipeline
        report = analyze(messages, settings, features=features, reviews=reviews, runtime=runtime,
                         allowed_media_dir=media_directory())
        rendered = render_report(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            _show_summary(report)
            print(f"计算报告 {args.output.resolve()}")
        else:
            print(rendered, end="")
    except (OSError, ValueError, sqlite3.Error, yaml.YAMLError) as error:
        print(f"运行失败：{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
