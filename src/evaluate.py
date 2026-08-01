from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config, shared_stats_path
from engine import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MoEDDI checkpoint")
    parser.add_argument("--config", default="configs/moeddi.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", "--input-dir", dest="data_dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--stats-path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["data"]["root"] = str(Path(args.data_dir).expanduser().resolve())
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        config["training"]["run_dir"] = str(run_dir)
    if args.stats_path:
        config["data"]["stats_path"] = str(Path(args.stats_path).expanduser().resolve())
    elif args.data_dir:
        config["data"]["stats_path"] = str(shared_stats_path(config))
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.max_rows is not None:
        config["evaluation"]["max_rows"] = args.max_rows
    if args.device is not None:
        config["training"]["device"] = args.device
    print(json.dumps(evaluate_checkpoint(config, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
