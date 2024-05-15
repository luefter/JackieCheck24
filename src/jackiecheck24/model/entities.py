from dataclasses import dataclass

import torch


@dataclass
class TrainingData:
    user_id: torch.tensor
    movie_ids: torch.tensor
    genres_matrix: torch.tensor
    years: torch.tensor
    src_padding_mask: torch.tensor
    ratings: torch.tensor


@dataclass
class HyperparameterTraining:
    lr: float


@dataclass
class HyperparameterTransformer:
    nhead: int
    num_layers: int
    d_genre: int
    d_year: int
    d_movie_id: int
    d_user: int
    d_rating: int


@dataclass
class HyperparameterClassifier:
    strategy: str


@dataclass
class Hyperparameter:
    transformer: HyperparameterTransformer
    training: HyperparameterTraining
    classifier: HyperparameterClassifier
