# JackieCheck24

Transformer model for predicting the rating of a film by a user.

### Data Sources: 
 - user_id
 - movie_id
 - genre of movie
 - year of movie
 - ratings from users of the movies

### How to run 
Within root directory

1. Install package 
 ```pip install -e .```


2. Setup local tracking server  
```mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri mlflow-server --default-artifact-root mlflow-server```


3. Start train using cli

``` python -m jackiecheck24 fit --config config.yaml --after_fit.save_model False``` 



### Stack:
- Lightning
- Polars
- Mlflow