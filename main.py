from fastapi import FastAPI

# 1. FastAPI 인스턴스 생성
app = FastAPI()

# 2. 경로 작동 함수 (Route Operation) 정의
@app.get("/")
def read_root():
    """
    HTTP GET 요청이 루트 경로('/')로 들어왔을 때 실행되는 함수
    """
    return {"message": "Hello, FastAPI"}
