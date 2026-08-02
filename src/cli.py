from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config, resolve_project_path, split_paths
from engine import (
    evaluate_checkpoint,
    load_and_validate_schema,
    prepare_statistics,
    train,
)
from schema import family_summary
from splits import generate_split_manifest


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def inspect_command(config: dict) -> None:
    schema = load_and_validate_schema(config)
    files = {}
    for role in ("train", "validation", "test"):
        files[role] = [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in split_paths(config, role)
        ]
    _print(
        {
            "num_columns": len(schema.all_columns),
            "num_features": schema.num_features,
            "metadata_columns": schema.metadata_columns,
            "target_column": schema.target_column,
            "schema_fingerprint": schema.fingerprint,
            "feature_family_membership_counts": family_summary(schema),
            "files": files,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="moeddi")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "train"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--force", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument(
        "--checkpoint",
        required=True,
        nargs="+",
        help="One checkpoint, or multiple checkpoints for deep-ensemble evaluation",
    )

    split_parser = subparsers.add_parser("make-split")
    split_parser.add_argument("--config", required=True)
    split_parser.add_argument(
        "--strategy",
        required=True,
        choices=["random", "unseen_drug", "scaffold"],
    )
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--ratios", nargs=3, type=float, default=(0.6, 0.2, 0.2))
    split_parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "inspect":
        inspect_command(config)
    elif args.command == "prepare":
        schema = load_and_validate_schema(config)
        path = prepare_statistics(config, schema, force=args.force)
        _print({"statistics_path": str(path)})
    elif args.command == "train":
        path = train(config)
        _print({"best_checkpoint": str(path)})
    elif args.command == "evaluate":
        _print(evaluate_checkpoint(config, args.checkpoint))
    elif args.command == "make-split":
        schema = load_and_validate_schema(config)
        paths = []
        for role in ("train", "validation", "test"):
            paths.extend(split_paths(config, role))
        unique_paths = list(dict.fromkeys(paths))
        output = Path(args.output)
        if not output.is_absolute():
            output = resolve_project_path(config, output)
        report = generate_split_manifest(
            unique_paths,
            schema,
            output,
            strategy=args.strategy,
            num_classes=config["data"]["num_classes"],
            ratios=tuple(args.ratios),
            seed=args.seed if args.seed is not None else config["seed"],
            block_size_mb=config["data"]["block_size_mb"],
        )
        _print(report)


if __name__ == "__main__":
    main()
