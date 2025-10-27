from pydantic import BaseModel

# 학습 완료 유무를 기본 필드로 설정함.
class TrainResponse(BaseModel):
    is_completed: bool

# 학습 관련 파라미터
class TrainStart(BaseModel):
    pass