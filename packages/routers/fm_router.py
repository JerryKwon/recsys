from typing import List

from fastapi import APIRouter

import os
import pandas as pd

import torch
import torch.nn.functional as F

from packages.config import DataInput, DataInputCS, PredictOutput
from packages.config import ProjectConfig

project_config = ProjectConfig("fm")

movies = pd.read_csv(os.path.join(project_config.data_path, 'ml-1m/movies.dat'), engine='python', header=None,
                     sep='::', names=['movie_id', 'title', 'genre'], encoding_errors="ignore")

dict_title_to_mid = dict(zip(movies["title"], movies["movie_id"]))
dict_mid_to_title = {v:k for k,v in dict_title_to_mid.items()}

model = project_config.load_model()
model.eval()

dict_col_offset = {'user_id': 0,
                   'movie_id': 6040,
                   'gender': 9992,
                   'age': 9994,
                   'occupation': 10001}

dict_age_to_idx = {1: 0, 18: 1, 25: 2, 35: 4, 45: 5, 50: 5, 56: 6}
dict_gender_to_idx = {"f":0, "m":1}

emb_movies = model.embedding.weight[dict_col_offset["movie_id"]: dict_col_offset["gender"], :].detach().cpu()

fm = APIRouter(prefix='/models/fm')

@fm.get('/',tags=['fm'])
async def start_ncf():
    return {'msg': 'Here is FM'}

@fm.post('/prediction', tags=['fm'], response_model=List[PredictOutput])
async def predict_fm(data_request: DataInput):
    requested_movie = data_request.movie_id - 1

    requested_movie = requested_movie + dict_col_offset["movie_id"]
    emb_requested_movie = model.embedding(torch.tensor(requested_movie, dtype=torch.long, device="cpu"))

    cos = torch.tensor(
        [F.cosine_similarity(emb_requested_movie, emb, dim=0) for emb in emb_movies]
    )

    k = 20

    cos_movies = sorted(cos.detach().numpy(), reverse=True)[:k]
    rec_movies = [mid+1 for mid in cos.argsort(descending=True)[:k].detach().numpy()]
    dict_rec_cos = dict(zip(rec_movies, cos_movies))

    df_rec_movies = movies.loc[movies["movie_id"].isin(rec_movies)]

    df_rec_movies["cos"] = df_rec_movies.movie_id.apply(lambda x:dict_rec_cos[x])
    df_rec_movies = df_rec_movies.sort_values(by="cos",ascending=False).copy()

    list_pred_output = []

    for idx, row in df_rec_movies.iterrows():
        pred_output = {
            "movie_id":row["movie_id"],
            "title":row["title"],
            "genre": row["genre"],
            "score": row["cos"]
        }

        list_pred_output.append(pred_output)

    return list_pred_output

@fm.post('/prediction-cs', tags=["fm"], response_model=List[PredictOutput])
async def predict_fm_cs(data_request: DataInputCS):
    gender = data_request.gender
    age = data_request.age

    idx_gender = dict_gender_to_idx[gender]
    idx_age = dict_age_to_idx[age]

    gender_emb = model.embedding(torch.tensor(idx_gender + dict_col_offset["gender"], dtype=torch.long, device="cpu"))
    age_emb = model.embedding(torch.tensor(idx_age + dict_col_offset["age"], dtype=torch.long, device="cpu"))
    metadata_emb = gender_emb + age_emb

    rankings = model.weight[dict_col_offset["movie_id"]:dict_col_offset["gender"]] + torch.matmul(
        model.embedding.weight[dict_col_offset["movie_id"]:dict_col_offset["gender"]], metadata_emb)

    k = 20

    score_movies = sorted(rankings.detach().cpu().numpy(), reverse=True)[:k]
    rec_movies = rankings.detach().cpu().numpy().argsort()[::-1][:k]

    dict_rec_score = dict(zip(rec_movies, score_movies))

    df_rec_movies = movies.loc[movies["movie_id"].isin(rec_movies)]

    df_rec_movies["score"] = df_rec_movies.movie_id.apply(lambda x:dict_rec_score[x-1])
    df_rec_movies = df_rec_movies.sort_values(by="score",ascending=False).copy()

    list_pred_output = []

    for idx, row in df_rec_movies.iterrows():
        pred_output = {
            "movie_id":row["movie_id"],
            "title":row["title"],
            "genre": row["genre"],
            "score": row["score"]
        }

        list_pred_output.append(pred_output)

    return list_pred_output