import os

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pydantic import Field
from pydantic import field_validator

from packages.handler import ModelHandler


class ProjectConfig(ModelHandler):
    """
    개별 모델을 위한 Router 클래스에서 사용되는 Config 클래스
    """
    def __init__(self, model_type):
        super(ProjectConfig, self).__init__()

        self.model_type = model_type
        self.project_path = os.path.abspath(os.getcwd())
        self.input_path = os.path.join(self.project_path, "input")
        self.data_path = os.path.join(self.project_path, "data")
        self.model_path = os.path.join(self.project_path, "models")
        self.target_path = os.path.join(self.model_path, f"{model_type}.pth")


class VariableConfig:
    def __init__(self):
        self.host_list = ['127.0.0.1', '0.0.0.0']
        self.port_list = ['8000', '8088']


class APIEnvConfig(BaseSettings):
    host: str = Field(default='0.0.0.0', env='api host')
    port: int = Field(default='8000', env='api server port')

    # host 점검 V1->V2
    @field_validator("host", mode="before")
    @classmethod
    def check_host(cls, host_input):
        if host_input == 'localhost':
            host_input = "127.0.0.1"
        if host_input not in VariableConfig().host_list:
            raise ValueError("host error")
        return host_input

    # port 점검 V1->V2
    @field_validator("port", mode="before")
    @classmethod
    def check_port(cls, port_input):
        if port_input not in VariableConfig().port_list:
            raise ValueError("port error")
        return port_input


class APIConfig(BaseModel):
    api_name: str = 'main:app'
    api_info: APIEnvConfig = APIEnvConfig()

class DataInput(BaseModel):
    movie_id:int=Field(ge=1, le=3953)

class DataInputCS(BaseModel):
    gender:str
    age:int
    @field_validator('gender')
    @classmethod
    def check_gender(cls, gender):
        g = gender.lower()
        possible_input = ['female', 'f', 'male', 'm']
        possible_input.extend([pi.upper() for pi in possible_input])
        print(possible_input)
        if g not in possible_input:
            raise ValueError(f'Gender must be in {possible_input}')
        return g

    @field_validator('age')
    @classmethod
    def check_age(cls, age):
        possible_input = [1, 18, 25, 35, 45, 50, 56]
        if age not in possible_input:
            raise ValueError(f'Age must be in {possible_input}')
        return age

class PredictOutput(BaseModel):
    movie_id:int=Field(ge=1, le=3953)
    title:str
    genre:str
    score:float
