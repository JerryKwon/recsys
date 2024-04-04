import os

from pydantic import BaseModel
from pydantic import Field

from packages.handler import ModelHandler


class ProjectConfig(ModelHandler):
    """
    개별 모델을 위한 Router 클래스에서 사용되는 Config 클래스
    """
    def __init__(self, model_type):
        super().__init__(ProjectConfig, self)

        self.model_type = model_type
        self.project_path = os.path.abspath(os.getcwd())
        self.model_path = os.path.join(self.project_path, "models")
        self.target_path = os.path.join(self.model_path, f"{model_type}.pth")

class DataInput(BaseModel):
    movie_id:int=Field(ge=1, le=3953)


class PredictOutput(BaseModel):
    movie_id:int=Field(ge=1, le=3953)
    title:str
    genre:str
    cos_sim:float=Field(ge=-1, le=1)
