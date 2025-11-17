import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import status

from main import app  # 실제 FastAPI app import
from app.models.llm_loader import load_tokenizers

load_tokenizers()

@pytest.mark.asyncio
async def test_train_success(monkeypatch):
    """✅ 정상 학습 완료 시 응답 테스트"""

    # mock dependencies
    async def mock_train_model(*args, **kwargs):
        return True

    def mock_process_training_data(tokenizer, data):
        return ["processed"]

    def mock_evaluate_model(model_path):
        return {"accuracy": 0.95}

    monkeypatch.setattr("app.services.dpo_train.train_model", mock_train_model)
    monkeypatch.setattr("app.services.load_dpo_datasets.process_training_data", mock_process_training_data)
    monkeypatch.setattr("app.services.evaluate.evaluate_model", mock_evaluate_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "start_training": True,
            "training_data": [
                {"input": "dummy input", "output": "dummy output"}
            ]
        }
        response = await ac.post("/api/v1/ai/train", json=payload)

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["is_completed"] is True
    assert "score" in response.json()


@pytest.mark.asyncio
async def test_train_no_data():
    """⚠️ training_data가 비었을 때"""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"start_training": True, "training_data": []}
        response = await ac.post("/api/v1/ai/train", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "training_data" in response.json()["detail"]


@pytest.mark.asyncio
async def test_train_data_processing_error(monkeypatch):
    """⚠️ 데이터 처리 중 에러"""

    def mock_process_training_data(tokenizer, data):
        raise ValueError("Invalid format")

    monkeypatch.setattr("app.services.load_dpo_datasets.process_training_data", mock_process_training_data)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "start_training": True,
            "training_data": ["bad data"]
        }
        response = await ac.post("/api/v1/ai/train", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "데이터 처리 실패" in response.json()["detail"]


@pytest.mark.asyncio
async def test_train_gpu_oom(monkeypatch):
    """⚠️ GPU 메모리 부족(RuntimeError) 시"""
    async def mock_train_model(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    def mock_process_training_data(tokenizer, data):
        return ["data"]

    monkeypatch.setattr("app.services.dpo_train.train_model", mock_train_model)
    monkeypatch.setattr("app.services.load_dpo_datasets.process_training_data", mock_process_training_data)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"start_training": True, "training_data": ["dummy"]}
        response = await ac.post("/api/v1/ai/train", json=payload)

    assert response.status_code == status.HTTP_507_INSUFFICIENT_STORAGE
    assert "GPU 메모리 부족" in response.json()["detail"]


@pytest.mark.asyncio
async def test_train_tokenizer_error(monkeypatch):
    """⚠️ Tokenizer 관련 오류(RuntimeError)"""
    async def mock_train_model(*args, **kwargs):
        raise RuntimeError("Tokenizer Error: pad_token missing")

    def mock_process_training_data(tokenizer, data):
        return ["data"]

    monkeypatch.setattr("app.services.dpo_train.train_model", mock_train_model)
    monkeypatch.setattr("app.services.load_dpo_datasets.process_training_data", mock_process_training_data)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"start_training": True, "training_data": ["dummy"]}
        response = await ac.post("/api/v1/ai/train", json=payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Tokenizer" in response.json()["detail"]
