from collections import deque

import httpx
import polars as pol
from loguru import logger
from tqdm import tqdm

from jackiecheck24.data.entities import (
    Mapping,
    SpecialAlphabeticalToken,
    SpecialNumericalToken,
)


def download_file(url, output_path):
    logger.info(f"Starting download from {url} to {output_path}")
    with httpx.stream("GET", url) as response:
        response.raise_for_status()
        total_length = int(response.headers.get("content-length"))
        with open(output_path, "wb") as f:
            progress_bar = tqdm(
                total=total_length, unit="MB", unit_scale=True, desc=output_path
            )
            for chunk in response.iter_bytes():
                f.write(chunk)
                progress_bar.update(len(chunk))
            progress_bar.close()
    logger.info(f"Completed download from {url} to {output_path}")


class Registry:
    def __init__(self):
        self.registry: dict[str, Mapping] = {}

    def add(self, mapping: Mapping) -> None:
        self.registry[mapping.mapping_name] = mapping

    def get(self, key) -> Mapping:
        return self.registry.get(key)

    def list_mappings(self) -> None:
        print(list(self.registry.keys()))

    def apply_map(self, df: pol.DataFrame, column_name: str, mapping_name: str):
        source_mapping = self.get(mapping_name)
        old_indices = source_mapping.old_indices
        new_indices = source_mapping.new_indices
        padding_index = source_mapping.pad_token.index

        df = df.with_columns(
            pol.col(column_name).replace(
                old=old_indices, new=new_indices, default=padding_index
            )
        )

        return df


def generate_pad_token(
    column: pol.Series,
) -> [SpecialNumericalToken, SpecialAlphabeticalToken]:
    if column.dtype.is_numeric():
        value = column.max() + 1
        return SpecialNumericalToken(index=0, value=value)

    if column.dtype.is_(pol.String):
        value = "<PAD>"
        return SpecialAlphabeticalToken(index=0, value=value)

    raise TypeError(f"dtype {column.dtype} not supported")


def generate_cls_token(column: pol.Series):
    if column.dtype.is_numeric():
        value = column.max() + 2
        return SpecialNumericalToken(index=1, value=value)

    if column.dtype.is_(pol.String):
        value = "<MASK>"
        return SpecialAlphabeticalToken(index=1, value=value)

    raise TypeError(f"dtype {column.dtype} not supported")


def generate_map(
    df: pol.DataFrame, column_name, use_pad=True, use_cls=False
) -> Mapping:
    column: pol.Series = (
        df.select(pol.col(column_name)).to_series().unique(maintain_order=True)
    )
    mapping = dict(enumerate(column, 2))

    pad_token = None
    cls_token = None

    if use_pad:
        pad_token = generate_pad_token(column)
        mapping[pad_token.index] = pad_token.value

    if use_cls:
        cls_token = generate_cls_token(column)
        mapping[cls_token.index] = cls_token.value

    mapping = Mapping(
        mapping_name=column_name,
        pad_token=pad_token,
        cls_token=cls_token,
        new_indices=pol.Series(mapping.keys(), strict=False),
        old_indices=pol.Series(list(mapping.values())),
        len=max(mapping.keys()) + 2,
    )
    return mapping


def register_map(mapping: Mapping, registry: Registry) -> None:
    registry.add(mapping)


def apply_map(df: pol.DataFrame, column_name, mapping: Mapping) -> pol.DataFrame:
    new_indices = mapping.new_indices
    old_indices = mapping.old_indices
    df = df.with_columns(
        pol.col(column_name).replace(old=old_indices, new=new_indices).cast(pol.UInt32)
    )
    return df


def chunker(
    seq: list, chunk_size, step_size, min_chunk_size=2, apply_padding=False
) -> list:
    seq_len = len(seq)
    seq_chunked = deque([])

    for i in range(seq_len, 0, -step_size):
        lower_boundary = max(0, i - chunk_size)
        upper_boundary = i

        if upper_boundary - lower_boundary < min_chunk_size:
            break

        chunk = seq[lower_boundary:upper_boundary]

        if lower_boundary == 0:
            if apply_padding and (n_missing := chunk_size - len(chunk)) > 0:
                chunk = pol.zeros(n_missing, chunk.dtype, eager=True).append(chunk)

            seq_chunked.appendleft(chunk)
            break

        seq_chunked.appendleft(chunk)

    return list(seq_chunked)
