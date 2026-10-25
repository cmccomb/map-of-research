"""Build the dataset card for a single-commit Hub publication.

The four staged Parquet configs and README must describe the same revision. The
legacy ``default`` config and any independent card text belong to the parent
revision and are retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

PUBLISHED_CONFIGS = frozenset(
    {"people", "works", "authorships", "profile_publications"}
)


def _indexed_entries(entries: Any, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list) or not all(
        isinstance(row, dict) for row in entries
    ):
        raise ValueError(f"Dataset card {field} must be a list of mappings")
    names = [row.get("config_name") for row in entries]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"Dataset card {field} has an unnamed config")
    if len(names) != len(set(names)):
        raise ValueError(f"Dataset card {field} has duplicate config names")
    return dict(zip(names, entries, strict=True))


def _require_preserved_parent_card(
    prior_data: dict[str, Any], prior_body: str, updated: str
) -> None:
    from huggingface_hub import DatasetCard

    rendered = DatasetCard(updated)
    after = rendered.data.to_dict()
    if rendered.text != prior_body or after.get("configs") != prior_data["configs"]:
        raise ValueError("Dataset card rewrite changed the body or config declarations")
    prior_infos = _indexed_entries(prior_data["dataset_info"], "dataset_info")
    after_infos = _indexed_entries(after.get("dataset_info"), "dataset_info")
    if after_infos.keys() != prior_infos.keys():
        raise ValueError("Dataset card rewrite changed the config inventory")
    for name in prior_infos.keys() - PUBLISHED_CONFIGS:
        if after_infos[name] != prior_infos[name]:
            raise ValueError(f"Dataset card rewrite changed unrelated config {name}")
    for field in prior_data.keys() - {"dataset_info"}:
        if after.get(field) != prior_data[field]:
            raise ValueError(f"Dataset card rewrite changed unrelated metadata {field}")


def build_updated_dataset_card(
    prior_readme: str, staged: Mapping[str, tuple[Any, Path]]
) -> str:
    """Update four config descriptions using the exact staged datasets/files.

    ``staged`` values are the ``datasets.Dataset`` instances used to serialize
    their respective Parquet files and those file paths. The bytes returned by
    this function must be added to the same Hub commit as the Parquet files.
    """

    import pyarrow.parquet as parquet
    from datasets import DatasetInfo
    from datasets.info import DatasetInfosDict
    from datasets.splits import SplitDict, SplitInfo
    from huggingface_hub import DatasetCard

    if set(staged) != PUBLISHED_CONFIGS:
        raise ValueError("Dataset card requires exactly the four published configs")
    if not prior_readme.startswith("---\n"):
        raise ValueError("Parent Hub README has no dataset-card front matter")
    card = DatasetCard(prior_readme)
    prior_body = card.text
    prior_data = deepcopy(card.data.to_dict())
    prior_configs = _indexed_entries(prior_data.get("configs"), "configs")
    prior_infos = _indexed_entries(prior_data.get("dataset_info"), "dataset_info")
    required = PUBLISHED_CONFIGS | {"default"}
    if not required <= prior_configs.keys() or not required <= prior_infos.keys():
        raise ValueError("Parent Hub README is missing a published or default config")

    updates: dict[str, DatasetInfo] = {}
    for name in sorted(PUBLISHED_CONFIGS):
        dataset, path = staged[name]
        path = Path(path)
        if path.name != "train-00000-of-00001.parquet" or path.parent.name != name:
            raise ValueError(f"Unexpected staged Parquet path for {name}: {path}")
        data_files = prior_configs[name].get("data_files")
        if data_files != [{"split": "train", "path": f"{name}/train-*"}]:
            raise ValueError(f"Unexpected parent Hub data_files pattern for {name}")
        candidate = parquet.ParquetFile(path)
        if candidate.metadata.num_rows != len(dataset):
            raise ValueError(f"Staged {name} Parquet row count differs from Dataset")
        if (
            candidate.schema_arrow.remove_metadata()
            != dataset.data.schema.remove_metadata()
        ):
            raise ValueError(f"Staged {name} Parquet schema differs from Dataset")
        arrow_bytes = dataset.data.nbytes
        splits = SplitDict()
        splits.add(
            SplitInfo(name="train", num_bytes=arrow_bytes, num_examples=len(dataset))
        )
        updates[name] = DatasetInfo(
            config_name=name,
            features=dataset.features,
            splits=splits,
            download_size=path.stat().st_size,
            dataset_size=arrow_bytes,
        )

    DatasetInfosDict(updates).to_dataset_card_data(card.data)
    updated = str(card)
    _require_preserved_parent_card(prior_data, prior_body, updated)
    return updated
