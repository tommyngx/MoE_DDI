import csv

from schema import infer_schema


def test_schema_uses_names_and_supports_interleaved_metadata(tmp_path):
    path = tmp_path / "sample.csv"
    header = [
        "PEOEVSA1",
        "drugid-drug_a",
        "PEOEVSA1*MRVSA2",
        "class",
        "drugid-drug_b",
        "LabuteASA",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, [1, "A", 2, 0, "B", 3]])

    schema = infer_schema(path, expected_num_features=3)
    assert schema.feature_columns == ("PEOEVSA1", "PEOEVSA1*MRVSA2", "LabuteASA")
    assert schema.family_indices["PEOE_VSA"] == (0, 1)
    assert schema.family_indices["MR_VSA"] == (1,)
    assert schema.family_indices["LabuteASA"] == (2,)

