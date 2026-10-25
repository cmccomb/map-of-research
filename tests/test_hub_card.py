"""The atomic Hub revision must contain a card describing its staged files."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml
from datasets import Dataset
from huggingface_hub import DatasetCard

from map_of_research.hub_card import (
    PUBLISHED_CONFIGS,
    _indexed_entries,
    _require_preserved_parent_card,
    build_updated_dataset_card,
)


def _parent_card() -> str:
    metadata = {
        "language": ["en"],
        "license": "mit",
        "pretty_name": "CMU Engineering Publications",
        "size_categories": ["10K<n<100K"],
        "configs": [
            {
                "config_name": name,
                "data_files": [
                    {
                        "split": "train",
                        "path": f"{'data' if name == 'default' else name}/train-*",
                    }
                ],
            }
            for name in sorted(PUBLISHED_CONFIGS | {"default"})
        ],
        "dataset_info": [
            {
                "config_name": name,
                "features": [{"name": "prior", "dtype": "string"}],
                "splits": [{"name": "train", "num_bytes": 100, "num_examples": 1}],
                "download_size": 60,
                "dataset_size": 100,
            }
            for name in sorted(PUBLISHED_CONFIGS | {"default"})
        ],
        "custom_field": {"keep": "this"},
    }
    return (
        f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n"
        "# Existing card\n\nUnrelated prose.\n"
    )


def _staged(tmp_path):
    staged = {}
    for name in PUBLISHED_CONFIGS:
        dataset = Dataset.from_dict({"value": [f"{name}-a", f"{name}-b"]})
        parquet_path = tmp_path / name / "train-00000-of-00001.parquet"
        parquet_path.parent.mkdir()
        dataset.to_parquet(parquet_path)
        staged[name] = dataset, parquet_path
    return staged


def _entries(metadata, field):
    return {row["config_name"]: row for row in metadata[field]}


def test_atomic_card_updates_all_staged_configs_and_preserves_parent_content(
    tmp_path,
) -> None:
    parent = _parent_card()
    staged = _staged(tmp_path)
    updated = build_updated_dataset_card(parent, staged)
    prior = DatasetCard(parent)
    card = DatasetCard(updated)
    prior_metadata = prior.data.to_dict()
    metadata = card.data.to_dict()

    assert card.text == prior.text
    assert metadata["configs"] == prior_metadata["configs"]
    assert (
        _entries(metadata, "dataset_info")["default"]
        == _entries(prior_metadata, "dataset_info")["default"]
    )
    assert {k: v for k, v in metadata.items() if k != "dataset_info"} == {
        k: v for k, v in prior_metadata.items() if k != "dataset_info"
    }
    for name, (dataset, path) in staged.items():
        info = _entries(metadata, "dataset_info")[name]
        assert info["config_name"] == name
        assert info["splits"] == [
            {
                "name": "train",
                "num_bytes": dataset.data.nbytes,
                "num_examples": len(dataset),
            }
        ]
        assert info["dataset_size"] == dataset.data.nbytes
        assert info["download_size"] == path.stat().st_size
        assert info["features"] == [{"name": "value", "dtype": "string"}]


def test_atomic_card_rejects_incomplete_stage(tmp_path) -> None:
    staged = _staged(tmp_path)
    staged.pop("people")
    with pytest.raises(ValueError, match="exactly the four"):
        build_updated_dataset_card(_parent_card(), staged)


def test_atomic_card_rejects_unmatched_parquet(tmp_path) -> None:
    staged = _staged(tmp_path)
    dataset, path = staged["people"]
    Dataset.from_dict({"value": ["only one"]}).to_parquet(path)
    with pytest.raises(ValueError, match="row count differs"):
        build_updated_dataset_card(_parent_card(), staged)
    Dataset.from_dict({"other": ["a", "b"]}).to_parquet(path)
    with pytest.raises(ValueError, match="schema differs"):
        build_updated_dataset_card(_parent_card(), staged)
    staged["people"] = dataset, tmp_path / "works" / path.name
    with pytest.raises(ValueError, match="Unexpected staged Parquet path"):
        build_updated_dataset_card(_parent_card(), staged)
    staged["people"] = dataset, path.with_name("wrong.parquet")
    with pytest.raises(ValueError, match="Unexpected staged Parquet path"):
        build_updated_dataset_card(_parent_card(), staged)


def test_atomic_card_rejects_changed_parent_structure(tmp_path) -> None:
    staged = _staged(tmp_path)
    with pytest.raises(ValueError, match="no dataset-card front matter"):
        build_updated_dataset_card("# body only", staged)

    card = DatasetCard(_parent_card())
    metadata = deepcopy(card.data.to_dict())
    metadata["configs"].append(deepcopy(metadata["configs"][0]))
    duplicate = f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n# body\n"
    with pytest.raises(ValueError, match="duplicate config names"):
        build_updated_dataset_card(duplicate, staged)

    metadata["configs"].pop()
    metadata["configs"] = [
        entry for entry in metadata["configs"] if entry["config_name"] != "default"
    ]
    missing_default = f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n# body\n"
    with pytest.raises(ValueError, match="missing a published or default"):
        build_updated_dataset_card(missing_default, staged)

    metadata = deepcopy(card.data.to_dict())
    metadata["dataset_info"] = [
        entry for entry in metadata["dataset_info"] if entry["config_name"] != "default"
    ]
    missing_default_info = (
        f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n# body\n"
    )
    with pytest.raises(ValueError, match="missing a published or default"):
        build_updated_dataset_card(missing_default_info, staged)

    metadata = deepcopy(card.data.to_dict())
    _entries(metadata, "configs")["people"]["data_files"][0]["path"] = "elsewhere/*"
    changed_path = f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n# body\n"
    with pytest.raises(ValueError, match="Unexpected parent Hub data_files pattern"):
        build_updated_dataset_card(changed_path, staged)


@pytest.mark.parametrize("entries", [None, ["wrong"]])
def test_card_config_index_rejects_non_mapping_entries(entries) -> None:
    with pytest.raises(ValueError, match="must be a list of mappings"):
        _indexed_entries(entries, "configs")


@pytest.mark.parametrize("name", ["", 2])
def test_card_config_index_rejects_unnamed_entries(name) -> None:
    with pytest.raises(ValueError, match="unnamed config"):
        _indexed_entries([{"config_name": name}], "configs")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("body", "body or config declarations"),
        ("configs", "body or config declarations"),
        ("inventory", "config inventory"),
        ("default", "unrelated config default"),
        ("license", "unrelated metadata license"),
    ],
)
def test_card_preservation_guard_detects_unrelated_edits(mutation, message) -> None:
    prior = DatasetCard(_parent_card())
    prior_data = deepcopy(prior.data.to_dict())
    changed = deepcopy(prior_data)
    body = prior.text
    if mutation == "body":
        body += "Changed.\n"
    elif mutation == "configs":
        _entries(changed, "configs")["people"]["data_files"][0]["path"] = "other/*"
    elif mutation == "inventory":
        changed["dataset_info"].pop()
    elif mutation == "default":
        _entries(changed, "dataset_info")["default"]["download_size"] = 99
    else:
        changed["license"] = "apache-2.0"
    altered = f"---\n{yaml.safe_dump(changed, sort_keys=False)}---\n{body}"
    with pytest.raises(ValueError, match=message):
        _require_preserved_parent_card(prior_data, prior.text, altered)
