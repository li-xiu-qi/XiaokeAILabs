"""起推理 HTTP 服务，加载训好的 checkpoint。

默认只监听 127.0.0.1；需要从别的机器访问时显式加 --host 0.0.0.0，
注意这会把决策接口暴露到局域网，确认网络环境可信再开。

用法：
  python scripts/serve.py --run-dir runs/agentjev_v1 \
      --backbone models/Qwen3-0.6B-Base --port 8149
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="训练产出的 run 目录")
    parser.add_argument("--backbone", required=True, help="本地底座目录")
    parser.add_argument("--port", type=int, default=8149)
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认本机")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = run_dir / "best.pt"
    temperatures = run_dir / "temperatures.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 {checkpoint}，先跑完训练")

    from http.server import ThreadingHTTPServer
    from jev_service.server import make_handler
    from jev_service.engine import DecisionEngine

    engine = DecisionEngine(
        str(checkpoint),
        args.backbone,
        args.device,
        args.max_tokens,
        temperatures=str(temperatures) if temperatures.exists() else None,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(json.dumps({"event": "ready", "host": args.host, "port": args.port, **engine.info()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
