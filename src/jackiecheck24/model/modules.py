import math
from dataclasses import asdict

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy


from jackiecheck24.data.entities import DataParameter
from jackiecheck24.model.entities import Hyperparameter, HyperparameterTransformer
from jackiecheck24.model.utils import output_processor


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class Transformer(nn.Module):
    def __init__(
        self, data_parameter: DataParameter, hyper_parameter: HyperparameterTransformer
    ):
        super().__init__()

        self.d_model = (
            hyper_parameter.d_genre
            + hyper_parameter.d_movie_id
            + hyper_parameter.d_year
        )

        # positional encoding
        self.pos_encoder = PositionalEncoding(emb_size=16, dropout=0)

        # user specific embeddings
        self.user_embedding = nn.Embedding(
            data_parameter.num_user, hyper_parameter.d_user, padding_idx=0
        )
        self.rating_embedding = nn.Embedding(
            data_parameter.num_rating, hyper_parameter.d_rating, padding_idx=0
        )

        # movie specific embeddings
        self.movie_id_embedding = nn.Embedding(
            data_parameter.num_movie_id, hyper_parameter.d_movie_id, padding_idx=0
        )
        self.movie_genre_embedding = nn.Embedding(
            data_parameter.num_genre, hyper_parameter.d_genre, padding_idx=0
        )
        self.movie_year_embedding = nn.Embedding(
            data_parameter.num_year, hyper_parameter.d_year, padding_idx=0
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=hyper_parameter.nhead,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=hyper_parameter.num_layers,
            enable_nested_tensor=True,
        )

    def forward(self, data):
        user_id = data["user_id"]
        movie_ids = data["movie_ids"]
        genres_matrix = data["genres_matrix"]
        years = data["years"]
        ratings = data["ratings"]
        src_padding_mask = data["src_padding_mask"]

        source_ratings = torch.ones_like(ratings)
        source_ratings[..., :-1] += ratings[..., :-1]

        # compute embedding for each component of the movie embedding
        movie_id_embedding = self.movie_id_embedding(movie_ids)
        movie_year_embedding = self.movie_year_embedding(years)
        movie_genre_embedding = self.movie_genre_embedding(genres_matrix)
        movie_genre_embedding = torch.sum(movie_genre_embedding, dim=-2)
        # concat embeddings related with movie
        movie_embedding = torch.concat(
            (movie_id_embedding, movie_genre_embedding, movie_year_embedding), -1
        )
        # add rating embedding to movie embeddings
        rating_id_embeddings = self.rating_embedding(source_ratings)
        user_behavior = torch.add(movie_embedding, rating_id_embeddings)
        # add position encoding
        user_behavior = self.pos_encoder(user_behavior)
        # add context to embeddings
        contextual_user_behavior = self.transformer_encoder(
            src=user_behavior, src_key_padding_mask=src_padding_mask
        )
        #  concat user embedding with contextualized movie embeddings
        user_id_embeddings = self.user_embedding(user_id)
        contextual_user_behavior = torch.concat(
            (user_id_embeddings, contextual_user_behavior), dim=-1
        )

        return contextual_user_behavior


class Classifier(nn.Module):
    def __init__(self, n_classes, input_preprocessor):
        super().__init__()
        self.input_preprocessor = input_preprocessor
        self.model = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.input_preprocessor(x)
        x = self.model(x)
        return x


class MovieRecommender(LightningModule):
    def __init__(
        self,
        hyper_parameter: Hyperparameter,
        data_parameter: DataParameter,
        n_classes: int = 12,
    ):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.lr = hyper_parameter.training.lr
        self.acc = Accuracy(task="multiclass", num_classes=n_classes)
        input_preprocessor = output_processor(
            strategy=hyper_parameter.classifier.strategy
        )
        self.model = nn.Sequential(
            Transformer(
                hyper_parameter=hyper_parameter.transformer,
                data_parameter=data_parameter,
            ),
            Classifier(n_classes=n_classes, input_preprocessor=input_preprocessor),
        )
        self.hyper_parameter = hyper_parameter

    def forward(self, batch) -> torch.tensor:
        return self.model(batch)

    def on_train_start(self):
        self.logger.log_hyperparams(asdict(self.hyper_parameter))

    def training_step(self, batch, batch_index):
        ratings = batch.get("ratings")
        true_rating = ratings[..., -1]
        pred_logits = self(batch)

        loss = self.loss(pred_logits, true_rating)
        self.log(
            name="train_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_index):
        ratings = batch.get("ratings")
        true_rating = ratings[..., -1]
        pred_logits = self(batch)
        loss = self.loss(pred_logits, true_rating)
        self.log(
            name="val_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        predicted_rating = torch.argmax(pred_logits, dim=1).to(torch.int64)
        acc = self.acc(predicted_rating, true_rating)
        self.log(
            name="val_acc",
            value=acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
