from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
import torch
import os

from app.core.config import base_settings
from app.models.llm_loader import get_tokenizer_2
from app.schemas.train import TrainRequest, TrainResponse
from app.services.dpo_train import train_model
from app.services.load_dpo_datasets import process_training_data
from app.services.evaluate import evaluate_model

router = APIRouter()

@router.post("", response_model=TrainResponse)
async def train(request: TrainRequest, tokenizer=Depends(get_tokenizer_2)):
    """모델 학습하기"""

    if not request.start_training:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "start_training 플래그가 비활성화되어 있습니다."}
        )
    
    # 마스킹 처리한 데이터 받기
    training_dataset = request.training_data

    if not training_dataset:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": "training_data가 비어있습니다."}
        )

    print("[START] 데이터 처리 시작")
    try:
        final_dataset = process_training_data(tokenizer, training_dataset)
        print("[COMPLETED] 데이터 처리 완료")

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": f"데이터 처리 실패: {str(e)}"},
        )

    print("[START] 모델 학습 시작")
    try:
        train_model(final_dataset, tokenizer)
        print("[COMPLETED] 모델 학습 완료")
    
    except RuntimeError as e:

        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            print("⚠️ GPU 메모리 부족 → 배치/길이 줄이기")
            return JSONResponse(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                content={"detail": "GPU 메모리 부족. 배치 크기나 시퀸스 길이를 줄이세요."}
            )
        
        elif "device-side assert triggered" in str(e):
            print("⚠️ CUDA Assert 발생 → 데이터셋 인덱스나 라벨 확인하기")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "CUDA Assert 발생. 데이터셋 확인 필요."}
            )

        elif "Tokenizer" in str(e):
            print("⚠️ Tokenizer 문제 → pad_token 설정 확인.")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": "Tokenizer 설정 오류. pad_token 또는 vocab 확인 필요."}
            )

        else:
            print(f"🚨 Unknown RuntimeError: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"RuntimeError: {str(e)}"},
            )
        
    except ValueError as e:
        print(f"⚠️ ValueError: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": f"잘못된 값: {str(e)}"},
        )
    
    except Exception as e:
        print(f"🚨 Unexpected error: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Unexpected error: {type(e).__name__}: {str(e)}"},
        )
    
    # 모델 성능평가 => HAERAE Benchmark 사용하기 + RAG 성능평가
    print("[START] 성능 평가")
    try:
        result = evaluate_model(base_settings.base_model + "/midm")
        print("[COMPLETED] 성능 평가 완료")
        
        os.rename(base_settings.base_model + "/midm", base_settings.base_model + "/midm_pre")
        print("기존 모델 /midm을 /midm_pre로 변경 완료")

        # 새로운 모델을 v_latest로 변경하기
        os.rename(base_settings.base_model + "/dpo_model", base_settings.base_model + "/midm")
        print("새로운 모델 /dpo_model을 /midm으로 변경")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"is_completed": True, "detail": "모델 학습 및 교체 완료"}
        )
    
    except Exception as e:
        print(f"[ERROR] 평가/교체 단계 실패: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"모델 평가 또는 교체 중 오류 발생: {str(e)}"},
        )
    
        
    