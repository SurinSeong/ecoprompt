from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse, FileResponse
import pymongo

from app.services.use_tools import create_pdf_from_conversation
from app.models.mongodb_loader import get_mongodb

router = APIRouter()

@router.post("/export-pdf", status_code=status.HTTP_201_CREATED)
async def export_conversation_to_pdf(
    chatting_id: int,
    title: str = "대화 기록",
    mongo_client=Depends(get_mongodb)
):
    """
    특정 대화방의 전체 대화 내용을 PDF로 내보내기
    """
    try:
        database = mongo_client["eco_prompt"]
        collection = database["message"]

        # pdf용 메시지 형식을 위한 리스트
        pdf_messages = []

        # 채팅 기록을 최신에서 과거순으로 불러온다.
        # 답변 성공한 AI 메시지 불러오기
        ai_messages = collection.find({"chatting_id": chatting_id, "sender_type": "AI", "status": "COMPLETED"}).sort("created_at", pymongo.DESCENDING)
        ai_messages = await ai_messages.to_list()

        # 최신 메시지가 가장 아래에 나올 수 있도록 수정함.
        for ai_message in ai_messages[::-1]:
            pdf_messages.append({
                "role": ai_message["sender_type"],
                "content": ai_message["content"],
                "timestamp": ai_message.get("created_at")
            })

            message_uuid = ai_message.get("messageUUID", None)
            if message_uuid:
                user_message = await collection.find_one({"chatting_id": chatting_id, "sender_type": "USER", "status": "COMPLETED", "messageUUID": str(message_uuid)})
                if user_message:
                    pdf_messages.append({
                        "role": user_message["sender_type"],
                        "content": user_message["content"],
                        "timestamp": user_message.get("created_at")
                    })

        # PDF 생성
        output_path = create_pdf_from_conversation(
            title=f"{title}_{chatting_id}",
            messages=pdf_messages,
            metadata={
                "chatting ID": str(chatting_id),
                "messages": str(len(pdf_messages))
            }
        )

        return FileResponse(
            path=output_path,
            filename=f"conversation_{chatting_id}.pdf",
            media_type="application/pdf"
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": str(e)}
        )
    
