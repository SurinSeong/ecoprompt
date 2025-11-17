import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import status

from main import app
from app.models.llm_loader import load_llm_engines


@pytest.mark.asyncio
async def test_chat_vllm_success(monkeypatch):
    """✅ 정상 스트림 동작 테스트"""
    # --- LLM 엔진 및 토크나이저 Mock ---
    monkeypatch.setattr("app.models.llm_loader.get_llm_engine_1", lambda: "dummy_engine_1")
    monkeypatch.setattr("app.models.llm_loader.get_llm_engine_2", lambda: "dummy_engine_2")
    monkeypatch.setattr("app.models.llm_loader.get_tokenizer_1", lambda: "dummy_tokenizer_1")
    monkeypatch.setattr("app.models.llm_loader.get_tokenizer_2", lambda: "dummy_tokenizer_2")
    monkeypatch.setattr("app.models.mongodb_loader.get_mongodb", lambda: "dummy_mongo")

    # --- Mock 서비스 로직 ---
    async def mock_find_question_type(**kwargs):
        class Dummy:
            async def ainvoke(self, payload):
                return {"route": "general"}
        return Dummy()

    async def mock_stream_chosen_response_vllm(**kwargs):
        class Dummy:
            async def astream(self, payload):
                for token in ["hello", "world"]:
                    yield token
        return Dummy()

    async def mock_generate_rejected_response_vllm(**kwargs):
        class Dummy:
            async def ainvoke(self, payload):
                return "REJECTED RESPONSE"
        return Dummy()

    monkeypatch.setattr("app.services.chat.find_question_type", mock_find_question_type)
    monkeypatch.setattr("app.services.chat.stream_chosen_response_vllm", mock_stream_chosen_response_vllm)
    monkeypatch.setattr("app.services.chat.generate_rejected_response_vllm", mock_generate_rejected_response_vllm)

    # --- 실제 요청 ---
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": "안녕!",
            "personal_prompt": "",
            "message_uuid": "test-uuid"
        }
        response = await ac.post("/api/v1/ai/prompt-response/vllm", json=payload)

    assert response.status_code == status.HTTP_201_CREATED
    body = response.text
    assert "START" in body
    assert "DONE" in body
    assert "REJECTED" in body


@pytest.mark.asyncio
async def test_chat_vllm_internal_error(monkeypatch):
    """❌ 내부 예외 발생 시 500 응답"""
    async def mock_find_question_type(**kwargs):
        raise RuntimeError("mock internal error")

    monkeypatch.setattr("app.services.chat.find_question_type", mock_find_question_type)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": "테스트",
            "personal_prompt": "",
            "message_uuid": "error-test"
        }
        response = await ac.post("/api/v1/ai/prompt-response/vllm", json=payload)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Internal error" in response.text


@pytest.mark.asyncio
async def test_chat_vllm_timeout(monkeypatch):
    """⏰ 상위 타임아웃 발생 시 503 응답"""

    async def mock_find_question_type(**kwargs):
        await asyncio.sleep(2)
        raise asyncio.TimeoutError()

    monkeypatch.setattr("app.services.chat.find_question_type", mock_find_question_type)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": "timeout",
            "personal_prompt": "",
            "message_uuid": "timeout-test"
        }
        response = await ac.post("/api/v1/ai/prompt-response/vllm", json=payload)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "server timeout" in response.text


@pytest.mark.asyncio
async def test_chat_vllm_stream_error(monkeypatch):
    """⚠️ 스트림 내부 오류 발생 (chunk 중 에러)"""

    async def mock_find_question_type(**kwargs):
        class Dummy:
            async def ainvoke(self, payload):
                return {"route": "general"}
        return Dummy()

    async def mock_stream_chosen_response_vllm(**kwargs):
        class Dummy:
            async def astream(self, payload):
                yield "ok"
                raise RuntimeError("stream failure")
        return Dummy()

    async def mock_generate_rejected_response_vllm(**kwargs):
        class Dummy:
            async def ainvoke(self, payload):
                return "rejected"
        return Dummy()

    monkeypatch.setattr("app.services.chat.find_question_type", mock_find_question_type)
    monkeypatch.setattr("app.services.chat.stream_chosen_response_vllm", mock_stream_chosen_response_vllm)
    monkeypatch.setattr("app.services.chat.generate_rejected_response_vllm", mock_generate_rejected_response_vllm)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_input": "에러 발생",
            "personal_prompt": "",
            "message_uuid": "stream-error"
        }
        response = await ac.post("/api/v1/ai/prompt-response/vllm", json=payload)

    # 스트리밍은 항상 201로 응답 시작하므로 상태는 201
    assert response.status_code == status.HTTP_201_CREATED
    assert "[ERROR]" in response.text
