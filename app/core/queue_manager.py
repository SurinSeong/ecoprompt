# app/core/queue_manager.py
import asyncio
from collections import deque
from datetime import datetime

class RequestQueue:
    def __init__(self, max_queue_size: int = 50):
        self.queue = deque(maxlen=max_queue_size)
        self.processing = set()
        
    async def enqueue(self, request_id: str):
        """요청을 큐에 추가"""
        if len(self.queue) >= self.queue.maxlen:
            raise Exception("Queue is full")
        
        self.queue.append({
            "id": request_id,
            "timestamp": datetime.now()
        })
        
    def get_position(self, request_id: str) -> int:
        """큐에서의 위치 반환"""
        for i, req in enumerate(self.queue):
            if req["id"] == request_id:
                return i + 1
        return -1