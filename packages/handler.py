import torch
from models import FM

class ModelHandler:
    """
    config.py에서 정의될 저장된 모델을 load할 ProjectConfig class에 상속할 클래스
    """
    def load_model(self):
            if self.model_type == "fm":
                model = FM(num_feats=10022, emb_dim=10)
                model.load_state_dict(torch.load(self.fm_path))

            return model

class DataHandler:
    def check_type(self, check_class, data):
        data = check_class(**data)

        return data