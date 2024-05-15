import zipfile
from functools import partial
from pathlib import Path

import polars as pol
import torch
import torch.nn.functional as func
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from jackiecheck24.model.entities import TrainingData
from jackiecheck24.data.entities import DataParameter
from jackiecheck24.data.preprocessing import prepare_movies, prepare_ratings
from jackiecheck24.data.utils import Registry, chunker, download_file


def label_last_row(df, column_name):
    df = df.with_row_index()
    df = df.join(
        df.group_by(column_name, maintain_order=True).last(), on="index", how="left"
    ).fill_nan(False)
    df = df.with_columns(last=~df.select(f"{column_name}_right").to_series().is_null())
    df = df.drop([col for col in df.columns if col.endswith("_right")])
    return df


def train_val_split(df: pol.DataFrame) -> (pol.DataFrame, pol.DataFrame):
    df = label_last_row(df, "userId")

    df_train = df.filter(pol.col("last").not_())
    df_val = df.filter(pol.col("last"))

    df_train = df_train.drop("last")
    df_val = df_val.drop("last")

    df_train = df_train.drop("index")
    df_val = df_val.drop("index")

    return df_train, df_val


class MovieDataset(Dataset):
    def __init__(
        self,
        df_movies,
        df_ratings,
        registry: Registry,
        chunk_size=3,
    ):
        self.df_movies = df_movies
        self.df_ratings = df_ratings
        self.chunk_size = chunk_size
        self.registry = registry

    def __len__(self):
        return len(self.df_ratings)

    def __getitem__(self, index):
        data = self.df_ratings.row(index, named=True)

        #
        movie_ids = data.get("movieId")
        user_id = data.get("userId")
        torch.tensor(movie_ids)
        # src_padding_mask = (torch.tensor(movie_ids) == PADDING_INDEX)
        src_padding_mask = (
            torch.tensor(movie_ids) == self.registry.get("movieId").pad_token.index
        )
        n_movie_ids = len(movie_ids)

        user_id = torch.tensor([user_id]).repeat((n_movie_ids,))
        ratings = data.get("rating")

        # get movie data
        movies = self.df_movies.filter(pol.col("movieId").is_in(movie_ids))
        years = movies.select("year").to_series().to_list()
        genres = movies.select("genre").to_series().to_list()
        n_genres = len(genres)

        # pad genres
        genres_matrix = torch.zeros(self.chunk_size, 21, dtype=torch.int64)
        genre_pad_token_index = self.registry.get("genre").pad_token.index
        for index, genre in enumerate(genres, n_movie_ids - n_genres):
            genre = torch.tensor(genre)  # noqa: PLW2901
            genre = func.pad(  # noqa: PLW2901
                genre, (21 - genre.size(0), 0), "constant", genre_pad_token_index
            )
            genres_matrix[index, :] = genre

            # pad years
        years = torch.tensor(years)
        years_pad_token_index = self.registry.get("year").pad_token.index

        years = func.pad(
            years,
            (self.chunk_size - years.size(0), 0),
            "constant",
            years_pad_token_index,
        )

        # cast
        movie_ids = torch.tensor(movie_ids)
        ratings = torch.tensor(ratings)

        data = {
            "user_id": user_id,
            "movie_ids": movie_ids,
            "genres_matrix": genres_matrix,
            "years": years,
            "src_padding_mask": src_padding_mask,
            "ratings": ratings,
        }

        return data


class MovieLensDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./movielens",
        batch_size=64,
        chunk_size=3,
        step_size=2,
    ):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.registry: Registry = Registry()

    def get_parameter(self) -> DataParameter:
        return DataParameter(
            num_rating=self.registry.get("rating").len,
            num_movie_id=self.registry.get("movieId").len,
            num_genre=self.registry.get("genre").len,
            num_year=self.registry.get("year").len,
            num_user=self.registry.get("userId").len,
        )

    def prepare_data(self, force=False) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        url: str = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        file_name: str = url.split("/")[-1]
        file_path: Path = Path(self.data_dir, file_name)

        if file_path.exists() and not force:
            logger.info("file aleady exists, preparations completed")
            return None

        download_file(url, str(file_path))
        logger.info(f"Starting extraction of {file_path}")
        zip_ref = zipfile.ZipFile(file_path, "r")
        zip_ref.extractall(self.data_dir)
        logger.info(f"Completed extraction of {file_path}")

    def setup(self, stage: str):
        if stage == "fit":
            chunking_strategy = partial(
                chunker,
                chunk_size=self.chunk_size,
                step_size=self.step_size,
                apply_padding=True,
            )

            ratings_dir: Path = Path(self.data_dir, "ml-latest-small/ratings.csv")
            movies_dir: Path = Path(self.data_dir, "ml-latest-small/movies.csv")

            raw_ratings: pol.DataFrame = pol.read_csv(ratings_dir)
            raw_movies: pol.DataFrame = pol.read_csv(movies_dir)

            df_ratings = prepare_ratings(
                raw_ratings, chunking_strategy, registry=self.registry
            )
            df_movies = prepare_movies(raw_movies, registry=self.registry)

            df_ratings_train, df_ratings_validation = train_val_split(df_ratings)

            self.movie_train = MovieDataset(
                df_movies,
                df_ratings_train,
                chunk_size=self.chunk_size,
                registry=self.registry,
            )
            self.movie_val: Dataset = MovieDataset(
                df_movies,
                df_ratings_validation,
                chunk_size=self.chunk_size,
                registry=self.registry,
            )

    def train_dataloader(self):
        return DataLoader(
            self.movie_train,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.movie_val,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            shuffle=False,
        )

    def predict_dataloader(self):
        raise NotImplementedError
