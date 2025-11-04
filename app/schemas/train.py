# 자가학습 관련 스키마
from pydantic import BaseModel

class TrainRequest(BaseModel):
    start_training: bool
    training_data: list

class TrainResponse(BaseModel):
    is_completed: bool