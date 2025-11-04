import os
from dotenv import load_dotenv

from langchain_mongodb import MongoDBChatMessageHistory

load_dotenv()

# 메시지 UUID 기준으로 조회하는 커스텀 클래스 설정
class CustomMongoDBChatMessageHistorywithUUID(MongoDBChatMessageHistory):

    def messages(self):
        """UUID 기준으로 조회"""
        cursor = self.collection.find(
            {"messageUUID": self.session_id}
        ).sort("createdAt", 1)
        return cursor

    
# chattingId 기준으로 조회하는 커스텀 클래스 설정
class CustomMongoDBChatMessageHistorywithChattingId(MongoDBChatMessageHistory):
    def add_message(self, message):
        """chatting_id를 사용하여 저장한다."""
        record = {
            "chattingId": self.session_id,
            "messageUUID": message.message_uuid,
            "content": message.content,
            "senderType": message.sender_type,
        }
        self.collection.insert_one(record)

    def messages(self):
        """UUID 기준으로 조회"""
        cursor = self.collection.find(
            {"chattingId": self.session_id}
        ).sort("createdAt", 1)
        return cursor

# MongoDB에서 채팅 내역 가져오기
def find_chatting_id(message_uuid: str):
    history = CustomMongoDBChatMessageHistorywithUUID(
        connection_string=os.getenv("MONGO_URL"),
        database_name="eco_prompt",
        collection_name="message",
        session_id=message_uuid
    )
    messages = history.messages()
    for message in messages:
        if message["chattingId"]:
            return message["chattingId"]
    return None

# Chat History 가져오기
def get_chat_history(chatting_id: int):
    chat_history = CustomMongoDBChatMessageHistorywithChattingId(
        connection_string=os.getenv("MONGO_URL"),
        database_name="eco_prompt",
        collection_name="message",
        session_id=chatting_id
    )
    return chat_history
