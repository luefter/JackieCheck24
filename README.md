# JackieCheck24

A Transformer-based model for predicting movie ratings on movielens.
The following input features are use for predictions

 - user_id
 - movie_id
 - genre of movie
 - year of movie (extracted from title)
 - ratings from users of the movies

### Installation Steps

1.Clone repository
```sh
git clone https://github.com/luefter/JackieCheck24.git 
cd JackieCheck24 
```
2. Install package 
 ```sh
pip install -e .
```


3. Setup local tracking server
```
mkdir mlflow-server
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri mlflow-server --default-artifact-root mlflow-server
```

### Basic usage
To train the model with default configs:
```sh
 python -m jackiecheck24 fit --config config.yaml 
 ``` 

### Advanced usage
The default configs can be changed in config.yaml. Those values can also be passed as arguments in the cli. 
For instance one can save the trained model and set the number of epochs to 3: 

```sh 
 python -m jackiecheck24 fit --config config.yaml --after_fit.save_model False --trainer.max_epochs 3
```

