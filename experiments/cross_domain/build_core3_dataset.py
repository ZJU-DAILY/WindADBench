from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import pandas as pd

try:
    from .core3_contract import (
        CORE3_VERSION,
        FARMS,
        FEATURE_COLUMNS,
        META_COLUMNS,
        OUTPUT_COLUMNS,
        canonical_feature_description,
        canonicalize_chunk,
        derived_farm_root,
        ensure_exact_columns,
        event_sort_key,
        mapping_document,
        normalize_identifier,
        raw_farm_root,
        repository_root,
        source_columns_for_farm,
        update_frame_digest,
    )
except ImportError:  # pragma: no cover
    from core3_contract import (  # type: ignore
        CORE3_VERSION,
        FARMS,
        FEATURE_COLUMNS,
        META_COLUMNS,
        OUTPUT_COLUMNS,
        canonical_feature_description,
        canonicalize_chunk,
        derived_farm_root,
        ensure_exact_columns,
        event_sort_key,
        mapping_document,
        normalize_identifier,
        raw_farm_root,
        repository_root,
        source_columns_for_farm,
        update_frame_digest,
    )


STRING_DTYPES = {column: "string" for column in META_COLUMNS}


def _atomic_json(document: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, sep=sep, index=False, encoding="utf-8")
    os.replace(temporary, path)


def _event_info_by_id(source_root: Path, farm_id: str) -> pd.DataFrame:
    path = raw_farm_root(source_root, farm_id) / "event_info.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing event metadata: {path}")
    frame = pd.read_csv(path, sep=";")
    if "event_id" not in frame.columns:
        raise ValueError(f"{path} has no event_id column.")
    frame = frame.copy()
    frame["_event_key"] = frame["event_id"].map(normalize_identifier)
    if frame["_event_key"].duplicated().any():
        duplicates = frame.loc[frame["_event_key"].duplicated(), "_event_key"].tolist()
        raise ValueError(f"Duplicate event IDs in {path}: {duplicates}")
    return frame.set_index("_event_key", drop=False)


def _central_metadata(source_root: Path) -> pd.DataFrame:
    path = source_root / "WIND_AD_META.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"farm_id", "event_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing {sorted(required - set(frame.columns))}.")
    frame = frame.copy()
    frame["_farm_key"] = frame["farm_id"].astype(str).str.strip()
    frame["_event_key"] = frame["event_id"].map(normalize_identifier)
    if frame.duplicated(["_farm_key", "_event_key"]).any():
        raise ValueError(f"Duplicate farm/event rows in {path}.")
    return frame.set_index(["_farm_key", "_event_key"], drop=False)


def _source_event_files(source_root: Path, farm_id: str) -> Dict[str, Path]:
    datasets = raw_farm_root(source_root, farm_id) / "datasets"
    if not datasets.is_dir():
        raise FileNotFoundError(f"Missing datasets directory: {datasets}")
    result: Dict[str, Path] = {}
    for path in datasets.glob("*.csv"):
        key = normalize_identifier(path.stem)
        if key in result:
            raise ValueError(f"Duplicate event filename key {key!r} in {datasets}.")
        result[key] = path
    return result


def _read_raw_chunks(
    path: Path, farm_id: str, chunk_size: int
) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, sep=";", nrows=0)
    required = source_columns_for_farm(farm_id)
    missing = set(required) - set(header.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    yield from pd.read_csv(
        path,
        sep=";",
        usecols=required,
        dtype=STRING_DTYPES,
        chunksize=chunk_size,
        low_memory=False,
        float_precision="round_trip",
    )


def _read_derived_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, sep=";", nrows=0)
    ensure_exact_columns(header.columns, OUTPUT_COLUMNS)
    yield from pd.read_csv(
        path,
        sep=";",
        dtype=STRING_DTYPES,
        chunksize=chunk_size,
        low_memory=False,
        float_precision="round_trip",
    )


def _expected_count(meta_row: Optional[pd.Series], column: str) -> Optional[int]:
    if meta_row is None or column not in meta_row or pd.isna(meta_row[column]):
        return None
    return int(float(meta_row[column]))


def _transform_one_event(
    *,
    source_path: Path,
    output_path: Path,
    farm_id: str,
    event_id: str,
    event_row: pd.Series,
    meta_row: Optional[pd.Series],
    source_root: Path,
    output_root: Path,
    chunk_size: int,
) -> Dict[str, object]:
    temporary = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()

    source_digest = hashlib.sha256()
    total_rows = 0
    train_rows = 0
    prediction_rows = 0
    missing_counts = {column: 0 for column in FEATURE_COLUMNS}
    asset_ids: set[str] = set()
    wrote_header = False

    try:
        for raw_chunk in _read_raw_chunks(source_path, farm_id, chunk_size):
            canonical = canonicalize_chunk(raw_chunk, farm_id)
            if canonical.empty:
                continue

            update_frame_digest(source_digest, canonical)
            total_rows += len(canonical)

            split = canonical["train_test"].astype("string").str.strip()
            allowed = {"train", "prediction", "test"}
            invalid = sorted(set(split.dropna().unique()) - allowed)
            if invalid:
                raise ValueError(
                    f"{source_path} contains invalid train_test values: {invalid}"
                )
            train_rows += int(split.eq("train").sum())
            prediction_rows += int(split.isin(["prediction", "test"]).sum())

            assets = canonical["asset_id"].dropna().map(normalize_identifier)
            asset_ids.update(value for value in assets if value)
            for column in FEATURE_COLUMNS:
                missing_counts[column] += int(canonical[column].isna().sum())

            canonical.to_csv(
                temporary,
                sep=";",
                index=False,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                encoding="utf-8",
                float_format="%.17g",
            )
            wrote_header = True

        if not wrote_header:
            raise ValueError(f"Source event is empty: {source_path}")
        if len(asset_ids) != 1:
            raise ValueError(
                f"{source_path} must contain exactly one asset_id, got {sorted(asset_ids)}"
            )
        if train_rows + prediction_rows != total_rows:
            raise ValueError(
                f"{source_path} has rows outside train/prediction partitions."
            )

        for manifest_name, actual in (
            ("train_lens", train_rows),
            ("prediction_lens", prediction_rows),
            ("total_lens", total_rows),
        ):
            expected = _expected_count(meta_row, manifest_name)
            if expected is not None and expected != actual:
                raise ValueError(
                    f"{source_path}: {manifest_name}={actual}, metadata says {expected}."
                )

        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    output_digest = hashlib.sha256()
    output_rows = 0
    for chunk in _read_derived_chunks(output_path, chunk_size):
        update_frame_digest(output_digest, chunk)
        output_rows += len(chunk)
    if output_rows != total_rows or output_digest.hexdigest() != source_digest.hexdigest():
        raise RuntimeError(
            f"Round-trip validation failed for {farm_id}_{event_id}: "
            f"source_rows={total_rows}, output_rows={output_rows}."
        )

    event_label = str(event_row.get("event_label", "")).strip().lower()
    record: Dict[str, object] = {
        "file_name": f"{farm_id}_{event_id}.csv",
        "farm_id": farm_id,
        "event_id": event_id,
        "asset_id": next(iter(asset_ids)),
        "event_label": event_label,
        "event_start": event_row.get("event_start"),
        "event_end": event_row.get("event_end"),
        "event_start_id": event_row.get("event_start_id"),
        "event_end_id": event_row.get("event_end_id"),
        "event_description": event_row.get("event_description"),
        "source_path": source_path.relative_to(source_root).as_posix(),
        "derived_path": output_path.relative_to(output_root).as_posix(),
        "train_rows": train_rows,
        "prediction_rows": prediction_rows,
        "total_rows": total_rows,
        "missing_wind_speed": missing_counts["wind_speed"],
        "missing_active_power": missing_counts["active_power"],
        "missing_rotor_speed": missing_counts["rotor_speed"],
        "content_sha256": source_digest.hexdigest(),
    }
    return record


def _preflight(
    source_root: Path, output_root: Path, *, overwrite: bool
) -> Dict[str, Dict[str, Path]]:
    all_files: Dict[str, Dict[str, Path]] = {}
    total = 0
    for farm_id in FARMS:
        files = _source_event_files(source_root, farm_id)
        event_info = _event_info_by_id(source_root, farm_id)
        file_ids = set(files)
        metadata_ids = set(event_info.index)
        if file_ids != metadata_ids:
            raise ValueError(
                f"Wind Farm {farm_id} event/file mismatch. "
                f"files_only={sorted(file_ids - metadata_ids, key=event_sort_key)}, "
                f"metadata_only={sorted(metadata_ids - file_ids, key=event_sort_key)}"
            )
        all_files[farm_id] = files
        total += len(files)

    existing = []
    for farm_id, files in all_files.items():
        target_dir = derived_farm_root(output_root, farm_id) / "datasets"
        existing.extend(
            target_dir / f"{event_id}.csv"
            for event_id in files
            if (target_dir / f"{event_id}.csv").exists()
        )
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"Found {len(existing)} existing derived event files (for example {preview}). "
            "Use --overwrite to rebuild them atomically, or --validate-only to check them."
        )
    print(f"Preflight OK: {total} source events across farms A/B/C.", flush=True)
    return all_files


def _metadata_row_for(
    central: pd.DataFrame, farm_id: str, event_id: str
) -> Optional[pd.Series]:
    if central.empty or (farm_id, event_id) not in central.index:
        return None
    row = central.loc[(farm_id, event_id)]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Duplicate metadata rows for {farm_id}_{event_id}.")
    return row


def build_dataset(
    source_root: Path,
    output_root: Path,
    *,
    chunk_size: int,
    overwrite: bool,
) -> pd.DataFrame:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("Source and output roots must be different.")

    all_files = _preflight(source_root, output_root, overwrite=overwrite)
    central = _central_metadata(source_root)
    records: List[Dict[str, object]] = []

    for farm_id in FARMS:
        event_info = _event_info_by_id(source_root, farm_id)
        source_farm = raw_farm_root(source_root, farm_id)
        output_farm = derived_farm_root(output_root, farm_id)
        output_farm.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_farm / "event_info.csv", output_farm / "event_info.csv")
        _atomic_csv(
            canonical_feature_description(),
            output_farm / "feature_description.csv",
            sep=";",
        )

        ordered_ids = sorted(all_files[farm_id], key=event_sort_key)
        for position, event_id in enumerate(ordered_ids, start=1):
            source_path = all_files[farm_id][event_id]
            output_path = output_farm / "datasets" / f"{event_id}.csv"
            meta_row = _metadata_row_for(central, farm_id, event_id)
            record = _transform_one_event(
                source_path=source_path,
                output_path=output_path,
                farm_id=farm_id,
                event_id=event_id,
                event_row=event_info.loc[event_id],
                meta_row=meta_row,
                source_root=source_root,
                output_root=output_root,
                chunk_size=chunk_size,
            )
            records.append(record)
            print(
                f"[{farm_id} {position:02d}/{len(ordered_ids):02d}] "
                f"event={event_id} asset={record['asset_id']} rows={record['total_rows']}",
                flush=True,
            )

    manifest = pd.DataFrame(records)
    manifest.sort_values(
        ["farm_id", "event_id"],
        key=lambda col: col.map(event_sort_key) if col.name == "event_id" else col,
        inplace=True,
    )
    manifest.reset_index(drop=True, inplace=True)

    wind_meta = manifest[
        [
            "file_name",
            "farm_id",
            "event_id",
            "event_label",
            "event_start",
            "event_end",
            "event_start_id",
            "event_end_id",
            "event_description",
        ]
    ].copy()
    wind_meta["raw_path"] = manifest["derived_path"].map(
        lambda value: str((output_root / str(value)).resolve())
    )
    wind_meta["train_lens"] = manifest["train_rows"]
    wind_meta["prediction_lens"] = manifest["prediction_rows"]
    wind_meta["total_lens"] = manifest["total_rows"]

    _atomic_csv(manifest, output_root / "manifest.csv")
    _atomic_csv(wind_meta, output_root / "WIND_AD_META.csv")
    _atomic_json(mapping_document(), output_root / "feature_mapping.json")
    print(
        f"Built {CORE3_VERSION}: {len(manifest)} events, "
        f"{int(manifest['total_rows'].sum()):,} rows -> {output_root}",
        flush=True,
    )
    return manifest


def validate_dataset(output_root: Path, *, chunk_size: int) -> pd.DataFrame:
    output_root = output_root.resolve()
    manifest_path = output_root / "manifest.csv"
    mapping_path = output_root / "feature_mapping.json"
    meta_path = output_root / "WIND_AD_META.csv"
    for path in (manifest_path, mapping_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Core-3 artifact: {path}")

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if mapping.get("dataset_version") != CORE3_VERSION:
        raise ValueError(
            f"Expected dataset_version={CORE3_VERSION}, got {mapping.get('dataset_version')!r}."
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype={"farm_id": "string", "event_id": "string", "asset_id": "string"},
    )
    if len(manifest) != 95:
        raise ValueError(f"Expected 95 events, found {len(manifest)}.")

    farm_counts = manifest.groupby("farm_id").size().to_dict()
    expected_farm_counts = {"A": 22, "B": 15, "C": 58}
    if farm_counts != expected_farm_counts:
        raise ValueError(
            f"Unexpected event counts: {farm_counts}; expected {expected_farm_counts}."
        )

    for position, row in manifest.iterrows():
        path = output_root / str(row["derived_path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing derived event: {path}")
        digest = hashlib.sha256()
        rows = 0
        for chunk in _read_derived_chunks(path, chunk_size):
            update_frame_digest(digest, chunk)
            rows += len(chunk)
        if rows != int(row["total_rows"]):
            raise ValueError(
                f"{path}: rows={rows}, manifest total_rows={row['total_rows']}"
            )
        if digest.hexdigest() != str(row["content_sha256"]):
            raise ValueError(f"{path}: content digest does not match manifest.")
        if (position + 1) % 10 == 0 or position + 1 == len(manifest):
            print(f"Validated {position + 1}/{len(manifest)} derived events.", flush=True)

    print(
        f"Validation OK: 95 events, {int(manifest['total_rows'].sum()):,} rows, "
        f"schema={list(OUTPUT_COLUMNS)}",
        flush=True,
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root,
        help="Raw dataset repository root (default: repository root).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "derived_datasets" / CORE3_VERSION,
        help="Destination for the isolated derived dataset.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace expected Core-3 files that already exist.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check an existing Core-3 dataset without reading wide raw files.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.validate_only:
        validate_dataset(args.output_root, chunk_size=args.chunk_size)
    else:
        build_dataset(
            args.source_root,
            args.output_root,
            chunk_size=args.chunk_size,
            overwrite=bool(args.overwrite),
        )
        validate_dataset(args.output_root, chunk_size=args.chunk_size)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Interrupted; completed event files remain valid and can be rebuilt "
            "with --overwrite."
        )
        sys.exit(130)
