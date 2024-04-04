from typing import List

from fastapi import APIRouter
import torch

from packages.config import DataInput, PredictOutput
from packages.config import ProjectConfig

project_config = ProjectConfig("fm")

model = project_config.load_model()
model.eval()

dict_col_offset = {'user_id': 0,
                   'movie_id': 6040,
                   'gender': 9992,
                   'age': 9994,
                   'occupation': 10001}


fm = APIRouter(prefix='/models/fm')

@fm.get('/',tags=['fm'])
async def start_ncf():
    return {'msg': 'Here is FM'}

@fm.post('/prediction', tags=['fm'], response_model=List[PredictOutput])
async def predict_fm(data_request: DataInput):
    movie_id = data_request.movie_id




# @fm.post('/coldstart-prediction')