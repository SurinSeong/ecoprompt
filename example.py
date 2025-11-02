# import os

# print(os.listdir('./models/Llama-SSAFY-8B'))

# # 제네레이터 연습
# def simple_generator():
#     print("함수 시작")
#     yield 1
#     print("중간 처리")
#     yield 2
#     print("끝")
#     return

# gen = simple_generator()

# print(next(gen))
# print(next(gen))
# print(next(gen))

import asyncio

async def event_stream():
    # Simulate AI processing steps
    steps = [
        "Step 1: Data loading complete...",
        "Step 2: Preprocessing data...",
        "Step 3: Model inference started...",
        "Step 4: Calculating results...",
        "Step 5: Postprocessing results...",
        "Step 6: Analysis complete."
    ]
    for step in steps:
        # Simulate work being done
        await asyncio.sleep(1)
        # Send data in SSE format: data: [message]\n\n
        yield f"data: {step}\n\n"
        print(f"FastAPI sent: {step}")  # 서버 측 로그

    # Optional: Send a final message or signal completion
    await asyncio.sleep(1)
    yield "data: FINISHED\n\n"
    print("FastAPI stream finished.")
