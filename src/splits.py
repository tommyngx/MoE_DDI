from __future__ import annotations

from pathlib import Path

import numpy as np

from data import discover_label_values, encode_labels, iter_metadata
from schema import DatasetSchema
from utils import write_json

DROPPED = np.uint8(255)


def _assign_groups(
    groups: list[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    ordered = np.asarray(sorted(set(groups)), dtype=object)
    rng.shuffle(ordered)
    first = round(len(ordered) * ratios[0])
    second = first + round(len(ordered) * ratios[1])
    return {
        str(group): 0 if index < first else 1 if index < second else 2
        for index, group in enumerate(ordered)
    }


def _stratified_random_assignments(
    labels: np.ndarray,
    ratios: tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    assignments = np.full(len(labels), DROPPED, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for class_value in np.unique(labels):
        indices = np.flatnonzero(labels == class_value)
        rng.shuffle(indices)
        first = round(len(indices) * ratios[0])
        second = first + round(len(indices) * ratios[1])
        assignments[indices[:first]] = 0
        assignments[indices[first:second]] = 1
        assignments[indices[second:]] = 2
    return assignments


def _scaffold_for_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:
        raise RuntimeError(
            "Scaffold splitting requires the optional rdkit dependency: "
            "pip install -e '.[scaffold]'"
        ) from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return f"INVALID::{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return scaffold or f"ACYCLIC::{Chem.MolToSmiles(molecule, canonical=True)}"


def generate_split_manifest(
    paths: list[Path],
    schema: DatasetSchema,
    output_path: str | Path,
    *,
    strategy: str,
    num_classes: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 2026,
    block_size_mb: int = 64,
) -> dict:
    if strategy not in {"random", "unseen_drug", "scaffold"}:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to one")

    file_drug_a: list[list[np.ndarray]] = [[] for _ in paths]
    file_drug_b: list[list[np.ndarray]] = [[] for _ in paths]
    file_labels: list[list[np.ndarray]] = [[] for _ in paths]
    smiles_by_drug: dict[str, str] = {}
    label_values = discover_label_values(
        paths,
        schema.target_column,
        expected_num_classes=num_classes,
        block_size_mb=block_size_mb,
    )
    for file_index, batch in iter_metadata(paths, schema, block_size_mb=block_size_mb):
        drug_a = batch["drugid-drug_a"].astype(str)
        drug_b = batch["drugid-drug_b"].astype(str)
        smiles_a = batch["drugsmiles-drug_a"].astype(str)
        smiles_b = batch["drugsmiles-drug_b"].astype(str)
        labels = encode_labels(batch[schema.target_column], label_values)
        file_drug_a[file_index].append(drug_a)
        file_drug_b[file_index].append(drug_b)
        file_labels[file_index].append(labels)
        smiles_by_drug.update(zip(drug_a, smiles_a, strict=True))
        smiles_by_drug.update(zip(drug_b, smiles_b, strict=True))

    drug_a_arrays = [np.concatenate(parts) for parts in file_drug_a]
    drug_b_arrays = [np.concatenate(parts) for parts in file_drug_b]
    label_arrays = [np.concatenate(parts) for parts in file_labels]

    if strategy == "random":
        combined = np.concatenate(label_arrays)
        combined_assignments = _stratified_random_assignments(combined, ratios, seed)
        assignments = []
        offset = 0
        for labels in label_arrays:
            assignments.append(combined_assignments[offset : offset + len(labels)])
            offset += len(labels)
    else:
        if strategy == "unseen_drug":
            group_by_drug = {drug: drug for drug in smiles_by_drug}
        else:
            group_by_drug = {
                drug: _scaffold_for_smiles(smiles) for drug, smiles in smiles_by_drug.items()
            }
        group_assignment = _assign_groups(list(group_by_drug.values()), ratios, seed)
        drug_assignment = {
            drug: group_assignment[group] for drug, group in group_by_drug.items()
        }
        assignments = []
        for drug_a, drug_b in zip(drug_a_arrays, drug_b_arrays, strict=True):
            assigned_a = np.fromiter(
                (drug_assignment[value] for value in drug_a),
                dtype=np.uint8,
                count=len(drug_a),
            )
            assigned_b = np.fromiter(
                (drug_assignment[value] for value in drug_b),
                dtype=np.uint8,
                count=len(drug_b),
            )
            values = np.where(assigned_a == assigned_b, assigned_a, DROPPED).astype(np.uint8)
            assignments.append(values)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"file_names": np.asarray([path.name for path in paths])}
    payload.update(
        {f"assignments_{index}": values for index, values in enumerate(assignments)}
    )
    np.savez_compressed(output_path, **payload)

    class_counts: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    names = ("train", "validation", "test")
    for code, name in enumerate(names):
        selected_labels = np.concatenate(
            [
                labels[assignment == code]
                for labels, assignment in zip(label_arrays, assignments, strict=True)
            ]
        )
        class_counts[name] = np.bincount(selected_labels, minlength=num_classes).tolist()
        totals[name] = int(len(selected_labels))
    dropped = sum(int(np.sum(values == DROPPED)) for values in assignments)
    total = sum(len(values) for values in assignments)
    report = {
        "strategy": strategy,
        "seed": seed,
        "ratios": ratios,
        "files": [path.name for path in paths],
        "num_unique_drugs": len(smiles_by_drug),
        "totals": totals,
        "dropped_pairs": dropped,
        "dropped_fraction": dropped / total if total else 0.0,
        "class_counts": class_counts,
    }
    write_json(output_path.with_suffix(".json"), report)
    return report
