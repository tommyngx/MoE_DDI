from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config, resolve_project_path, shared_stats_path
from engine import load_and_validate_schema, prepare_statistics, resolve_run_dir, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MoEDDI or a configured baseline")
    parser.add_argument("--config", default="configs/moeddi.yaml")
    parser.add_argument(
        "--data",
        "--input-dir",
        dest="data_dir",
        help="Folder containing the three split CSV files",
    )
    parser.add_argument("--train-file", help="Training CSV filename inside input-dir")
    parser.add_argument("--validation-file", help="Validation CSV filename inside input-dir")
    parser.add_argument("--test-file", help="Test CSV filename inside input-dir")
    parser.add_argument("--run-dir", help="Output folder for checkpoints, info, and plots")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--stats-path",
        help="Custom reusable preprocessing cache (default: <data>/train_stats.npz)",
    )
    parser.add_argument("--model", choices=["moeddi", "tddi_mlp", "mlp", "linear"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--pretrain",
        "--pretrained-checkpoint",
        dest="pretrained_checkpoint",
        help=(
            "Checkpoint whose model weights are loaded before training. "
            "Optimizer and scheduler start fresh."
        ),
    )
    parser.add_argument(
        "--tddi-pretrain",
        dest="tddi_pretrained_checkpoint",
        help="Initialize the hybrid MoE global path from one corrected T-DDI checkpoint",
    )
    parser.add_argument("--freeze-tddi-epochs", type=int)
    parser.add_argument("--tddi-backbone-lr-multiplier", type=float)
    parser.add_argument("--router-top-k", type=int, help="Top-K experts to select in MoE router")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--force-stats", action="store_true")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    configured_stats_path = resolve_project_path(config, config["data"]["stats_path"])
    if args.data_dir:
        config["data"]["root"] = str(Path(args.data_dir).expanduser().resolve())
    if args.train_file:
        config["data"]["train_files"] = [args.train_file]
    if args.validation_file:
        config["data"]["validation_files"] = [args.validation_file]
    if args.test_file:
        config["data"]["test_files"] = [args.test_file]
    if args.run_name:
        config["run_name"] = args.run_name
    if args.run_dir:
        config["training"]["run_dir"] = args.run_dir
    elif args.run_name:
        config["training"]["run_dir"] = f"runs/{args.run_name}"

    scalar_overrides = {
        ("data", "batch_size"): args.batch_size,
        ("model", "router_top_k"): args.router_top_k,
        ("training", "epochs"): args.epochs,
        ("training", "learning_rate"): args.learning_rate,
        ("training", "weight_decay"): args.weight_decay,
        ("training", "pretrained_checkpoint"): args.pretrained_checkpoint,
        ("training", "tddi_pretrained_checkpoint"): args.tddi_pretrained_checkpoint,
        ("training", "freeze_tddi_epochs"): args.freeze_tddi_epochs,
        (
            "training",
            "tddi_backbone_lr_multiplier",
        ): args.tddi_backbone_lr_multiplier,
        ("training", "device"): args.device,
        ("training", "max_rows"): args.max_train_rows,
        ("training", "max_steps_per_epoch"): args.max_steps_per_epoch,
        ("evaluation", "max_rows"): args.max_eval_rows,
    }
    for (section, key), value in scalar_overrides.items():
        if value is not None:
            config[section][key] = value
    if args.seed is not None:
        config["seed"] = args.seed
    if args.model is not None:
        config["model"]["name"] = args.model

    run_dir = resolve_run_dir(config)
    config["training"]["run_dir"] = str(run_dir)
    if args.stats_path:
        config["data"]["stats_path"] = str(Path(args.stats_path).expanduser().resolve())
    elif args.data_dir:
        config["data"]["stats_path"] = str(shared_stats_path(config))
    if not args.stats_path:
        config["data"]["stats_fallback_paths"] = list(
            dict.fromkeys(
                [
                    str(run_dir / "info" / "train_stats.npz"),
                    str(configured_stats_path),
                ]
            )
        )
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)
    if args.force_stats:
        schema = load_and_validate_schema(config)
        prepare_statistics(config, schema, force=True)
    best_checkpoint = train(config)
    run_dir = resolve_run_dir(config)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(run_dir / "last.pt"),
                "statistics_cache": config["data"]["stats_path"],
                "training_plot": str(run_dir / "plots" / "training_curves.png"),
                "paper_plots": str(run_dir / "plots" / "paper"),
                "summary": str(run_dir / "summary.txt"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
