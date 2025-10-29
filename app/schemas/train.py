from pydantic import BaseModel

# 학습 완료 유무
class TrainResponse(BaseModel):
    is_completed: bool
    message: str | None

# 학습 시작 및 데이터 ID
class TrainRequest(BaseModel):
    start_training: bool = True
    # 파인튜닝에 필요한 데이터
    # 사용자 질문, 마스킹된 선호 답변, 마스킹된 비선호 답변
    training_data: list