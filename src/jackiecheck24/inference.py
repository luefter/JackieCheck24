import mlflow.pyfunc
from mlflow import MlflowClient
from pprint import pprint
import torch

from jackiecheck24.data.dataset import MovieDataset, MovieLensDataModule

tracking_uri = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient(tracking_uri=tracking_uri)


for rm in client.search_registered_models():
    pprint(dict(rm), indent=4)

model_name = "some_model"
model_version = 1

model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
md = MovieLensDataModule(batch_size=5, chunk_size=10)
md.prepare_data()
md.setup(stage="fit")


data = md.movie_val[0]
single_batch = {k: torch.unsqueeze(v, 0) for k, v in data.items()}

batch = next(iter(md.val_dataloader()))


out = model(batch)

print(out.shape)
pred_rating = torch.argmax(out, 1)
true_rating = batch["ratings"][..., -1]

print(pred_rating, true_rating)
acc = torch.sum(pred_rating == true_rating) / len(pred_rating)
print(acc)


for batch in md.val_dataloader():
    out = model(batch)

    pred_rating = torch.argmax(out, 1)
    true_rating = batch["ratings"][..., -1]
    acc = torch.sum(pred_rating == true_rating) / len(pred_rating)
    print(acc)
