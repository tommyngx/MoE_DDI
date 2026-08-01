from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engine import _load_pretrained_weights
from train import build_parser


def _config(tmp_path, checkpoint):
    return {
        "_project_root": str(tmp_path),
        "training": {"pretrained_checkpoint": checkpoint},
    }


def test_train_parser_accepts_pretrain_alias():
    args = build_parser().parse_args(["--pretrain", "runs/old/best.pt"])
    assert args.pretrained_checkpoint == "runs/old/best.pt"


def test_load_pretrained_project_checkpoint(tmp_path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "schema_fingerprint": "schema-1",
            "label_values": np.array([1, 7]),
        },
        checkpoint_path,
    )

    loaded = _load_pretrained_weights(
        target,
        _config(tmp_path, "best.pt"),
        SimpleNamespace(fingerprint="schema-1"),
        SimpleNamespace(label_values=np.array([1, 7])),
        torch.device("cpu"),
    )

    assert loaded == checkpoint_path
    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_load_pretrained_rejects_different_schema(tmp_path):
    model = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "schema_fingerprint": "old-schema",
            "label_values": np.array([1, 7]),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="schema does not match"):
        _load_pretrained_weights(
            model,
            _config(tmp_path, "best.pt"),
            SimpleNamespace(fingerprint="new-schema"),
            SimpleNamespace(label_values=np.array([1, 7])),
            torch.device("cpu"),
        )
