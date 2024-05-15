from collections.abc import Callable

import polars as pol

from jackiecheck24.data.utils import Registry, apply_map, generate_map, register_map


def extract_year(df_source: pol.DataFrame) -> pol.DataFrame:
    df_source = df_source.with_columns(
        df_source["title"].str.extract(r"\((\d{4})\)").cast(int).alias("year")
    )
    return df_source


def remove_year_from_title(df_source):
    df_source = df_source.with_columns(
        df_source["title"].str.replace(r"\((\d{4})\)", "").str.strip_chars()
    )
    return df_source


def split_column(df_source, column_name, by="|", alias_name="genre"):
    df_source = df_source.with_columns(
        pol.col(column_name).str.split(by).alias(alias_name)
    )
    return df_source


def explode_column(df_source, column_name):
    df_source = df_source.explode(column_name)
    return df_source


def implode_column(df_source, column_name_id, column_name_target):
    df_target = (
        df_source.select([column_name_id, column_name_target])
        .group_by(column_name_id, maintain_order=True)
        .all()
    )
    df_source = df_source.select(
        [
            column_name
            for column_name in df_source.columns
            if column_name != column_name_target
        ]
    ).unique(maintain_order=True)
    df_source = df_target.join(df_source, on=column_name_id, how="left")
    df_source = df_source.drop(
        col_name for col_name in df_source.columns if col_name.endswith("_right")
    )
    return df_source


def merge_column(
    df_source: pol.DataFrame, df_target: pol.DataFrame, id_column_name: str
):
    df_source = df_source.join(df_target, on=id_column_name, how="inner")
    return df_source


def generate_register_apply_map(df, column_name, registry: Registry):
    mapping = generate_map(df, column_name)
    register_map(mapping, registry)
    df = apply_map(df, column_name, mapping)
    return df


def prepare_movieId(df, registry):
    df = df.pipe(generate_register_apply_map, column_name="movieId", registry=registry)
    return df


def prepare_title(df, registry):
    df = df.pipe(extract_year).pipe(
        generate_register_apply_map, "year", registry=registry
    )
    return df


def prepare_genres(df, registry):
    df = (
        df.pipe(split_column, column_name="genres", alias_name="genre")
        .pipe(explode_column, column_name="genre")
        .pipe(generate_register_apply_map, "genre", registry=registry)
        .pipe(implode_column, "movieId", "genre")
        .drop("genres")
        .sort("movieId")
    )

    return df


def prepare_movies(df, registry):
    df = (
        df.pipe(prepare_movieId, registry=registry)
        .pipe(prepare_title, registry=registry)
        .pipe(prepare_genres, registry=registry)
    )
    return df


def apply_map_to_column(df, column_name, registry):
    mapping = registry.get(column_name)
    df = apply_map(df, column_name, mapping)
    return df


def chunk_list(df, column_name, chunking_strategy: Callable[[list], list[list]]):
    return_dtype = pol.List(df.schema.get(column_name))
    df = df.with_columns(
        pol.col(column_name).map_elements(
            lambda seq: chunking_strategy(seq), return_dtype=return_dtype
        )
    )
    return df


def prepare_ratings(df: pol.DataFrame, chunking_strategy, registry):
    df = (
        (
            df.pipe(lambda df: df.sort("timestamp").sort("userId")).pipe(
                generate_register_apply_map, registry=registry, column_name="movieId"
            )
        )
        .pipe(generate_register_apply_map, registry=registry, column_name="userId")
        .pipe(generate_register_apply_map, registry=registry, column_name="rating")
        .pipe(lambda df: df.group_by("userId", maintain_order=True).all())
        .pipe(chunk_list, column_name="rating", chunking_strategy=chunking_strategy)
        .pipe(chunk_list, column_name="movieId", chunking_strategy=chunking_strategy)
        .explode(["movieId", "rating"])
        .drop("timestamp", "index")
    )
    return df
