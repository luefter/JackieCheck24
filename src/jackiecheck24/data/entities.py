from dataclasses import dataclass

import polars as pol


@dataclass
class DataParameter:
    num_rating: int
    num_movie_id: int
    num_genre: int
    num_year: int
    num_user: int


@dataclass
class SpecialToken:
    index: int = 0


@dataclass
class SpecialNumericalToken(SpecialToken):
    value: int = 0


@dataclass
class SpecialAlphabeticalToken(SpecialToken):
    value: str = "<PAD>"


@dataclass
class Mapping:
    mapping_name: str
    pad_token: SpecialNumericalToken | SpecialAlphabeticalToken
    cls_token: SpecialNumericalToken | SpecialAlphabeticalToken
    new_indices: pol.Series
    old_indices: pol.Series
    len: int
