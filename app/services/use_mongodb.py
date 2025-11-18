from langchain_core.runnables import RunnableLambda
import pymongo

# MongoDB에서 사용자 대화 내용 가져와서 요약하는 로직
def summarize_history(chat_history, tokenizer, llm):
    stored_messages = ""
    
    # 저장된 내용 가져오기 - AI, USER
    for message in chat_history.messages():
        sender_type = message["senderType"]
        if sender_type in ["AI", "USER"]:
            content = message["content"]
            stored_messages += f"[{sender_type}] {content}\n"
        
    if stored_messages:
        print(f"[이전 대화 기록] {stored_messages}")
    else:
        print("[이전 대화 기록] 없음")
    
    def build_prompt_with_template(chat_history: str) -> str:

        summarize_prompt = [
        {"role": "system", "content": "당신은 대화 기록을 요약하는 전문가입니다. 다음 대화 기록을 읽고 내용을 3~5줄로 요약해주세요."},
        {"role": "user", "content": chat_history}
    ]

        return tokenizer.apply_chat_template(
            summarize_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
    
    prompt = RunnableLambda(build_prompt_with_template)

    summarize_chain = prompt | llm

    summarized_messages = summarize_chain.invoke(stored_messages)

    return summarized_messages


# 현재 chatting_id 찾기
async def find_chatting_id(mongo_client, message_uuid: str):
    try:
        
        database = mongo_client["eco_prompt"]
        collection = database["message"]

        # message_uuid를 통해서 chatting_id를 찾는다.
        result = await collection.find_one({"messageUUID": message_uuid})
        # 채팅방 ID 받기
        chatting_id = result.get("chatting_id")
        if not chatting_id:
            raise ValueError(f"채팅 ID가 존재하지 않습니다. message_uuid: {message_uuid}")

        return chatting_id
        
    except Exception as e:
        raise Exception(f"❌ Failed to load MongoDB: {e}")


# 채팅 기록 가져오기
async def get_chat_history(mongo_client, chatting_id: int):
    try:
        database = mongo_client["eco_prompt"]
        collection = database["message"]

        # 채팅 기록 저장용
        chat_history = ""

        # 채팅 기록을 최신에서 과거순으로 불러온다.
        # 답변 성공한 AI 메시지 불러오기
        ai_messages = collection.find({"chatting_id": chatting_id, "sender_type": "AI", "status": "COMPLETED"}).sort("created_at", pymongo.DESCENDING)
        ai_messages = await ai_messages.to_list(3)

        # 최신 메시지가 가장 아래에 나올 수 있도록 수정함.
        for ai_message in ai_messages[::-1]:

            message_uuid = ai_message.get("messageUUID", None)

            if message_uuid:
                user_message = await collection.find_one({"chatting_id": chatting_id, "sender_type": "USER", "status": "COMPLETED", "messageUUID": str(message_uuid)})
                if user_message:
                    chat_history += f"[USER] ({user_message["created_at"]}) {user_message["content"]}\n[AI] ({ai_message["created_at"]}) {ai_message["content"]}\n"
                else:
                    chat_history += f"[AI] ({ai_message["created_at"]}) {ai_message["content"]}\n"

        return chat_history
    
    except Exception as e:
        raise Exception(f"❌ Failed to load chat history: {e}")
                
            
# 비선호 응답 저장하기
async def save_rejected_response(mongo_client, chatting_id: int, message_uuid: str, rejected_response: str):
    try:
        database = mongo_client["eco_prompt"]
        collection = database["message"]
        await collection.insert_one(
            {
                "chatting_id": chatting_id,
                "messageUUID": message_uuid,
                "content": rejected_response,
                "sender_type": "TRAINING",
            }
        )
        return True
    
    except Exception as e:
        raise Exception(f"❌ Failed to save rejected response: {e}")
        

# import os
# from dotenv import load_dotenv

# from pymongo import MongoClient

# load_dotenv()

# client = MongoClient(
#     os.getenv("MONGO_URL")
# )
# db = client["eco_prompt"]
# collection = db["message"]
# messages = collection.find({"chatting_id": 50, "sender_type": "AI"})
# # print(messa)
# for message in messages:
#     print(message)