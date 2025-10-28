from pydantic import BaseModel
from typing import Optional

# 학습 완료 유무
class TrainResponse(BaseModel):
    is_completed: bool
    message: str[Optional]

# 학습 시작 및 데이터 ID
class TrainRequest(BaseModel):
    start_training: bool = True
    # 파인튜닝에 필요한 데이터 시작 id (DB에 저장되어 있는)
    start_id: int