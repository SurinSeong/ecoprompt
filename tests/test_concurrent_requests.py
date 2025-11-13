# test_concurrent_requests.py
import asyncio
import aiohttp
import time
from datetime import datetime
import uuid

async def send_request(session, request_id):
    """단일 요청 전송"""
    url = "http://localhost:8001/api/v1/ai/prompt-response/vllm"
    
    payload = {
        "userInput": f"파이썬에서 리스트를 정렬하는 방법 알려줘 (요청 #{request_id})",
        "personalPrompt": "친절하게 답변해줘",
        "messageUUID": str(uuid.uuid4())
    }
    
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 요청 #{request_id} 시작")
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 요청 #{request_id} 실패 (상태: {response.status})")
                print(f"   에러 내용: {error_text}")
                return

            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 요청 #{request_id} 응답 시작")
            
            token_count = 0
            
            # SSE 스트림 읽기
            async for line in response.content:
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith('data:'):
                        token_count += 1
                        # 처음과 마지막 몇 개만 출력
                        if token_count <= 3 or 'DONE' in decoded:
                            print(f"   [{request_id}] {decoded[:80]}...")
            
            elapsed = time.time() - start_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 요청 #{request_id} 완료 (소요: {elapsed:.2f}초, 토큰: {token_count}개)")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 요청 #{request_id} 예외 발생: {e}")

async def test_concurrent_requests(num_requests=5):
    """여러 요청을 동시에 전송"""
    print(f"\n{'='*70}")
    print(f"🚀 동시 요청 테스트 시작: {num_requests}개의 요청")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # 타임아웃 설정
    timeout = aiohttp.ClientTimeout(total=300)  # 10분
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 모든 요청을 동시에 실행
        tasks = [send_request(session, i+1) for i in range(num_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✨ 전체 테스트 완료")
    print(f"   📊 총 요청 수: {num_requests}")
    print(f"   ⏱️  총 소요 시간: {total_time:.2f}초")
    print(f"   📈 평균 처리 시간: {total_time/num_requests:.2f}초/요청")
    print(f"{'='*70}\n")

async def test_sequential():
    """순차 테스트 - 서버가 정상 작동하는지 먼저 확인"""
    print("\n🔍 순차 테스트 (서버 정상 작동 확인)\n")
    
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await send_request(session, 1)

if __name__ == "__main__":
    import sys
    
    # 명령행 인자로 테스트 모드 선택
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "seq":
            # 순차 테스트
            asyncio.run(test_sequential())
        elif mode.isdigit():
            # 지정된 개수만큼 동시 테스트
            asyncio.run(test_concurrent_requests(int(mode)))
    else:
        # 기본: 3개 동시 요청 테스트
        asyncio.run(test_concurrent_requests(3))