from fastapi import APIRouter, Depends
import torch
import os

from app.core.config import base_settings
from app.models.llm_loader import get_tokenizer
from app.schemas.train import TrainRequest, TrainResponse
from app.services.dpo_train import train_model
from app.services.load_dpo_datasets import process_training_data
from app.services.evaluate import evaluate_model

router = APIRouter()

@router.post("", response_model=TrainResponse, status_code=200)
async def train(request: TrainRequest, tokenizer=Depends(get_tokenizer)):
    """모델 학습하기"""

    if request.start_training:
        # 마스킹 처리한 데이터 받기
        training_dataset = request.training_data

    if training_dataset:

        print("[START] 데이터 처리 시작")
        # 데이터 처리
        final_dataset = process_training_data(tokenizer, training_dataset)

        print("[COMPLETED] 데이터 처리 완료")

        try:
            print("[START] 모델 학습 시작")
            train_model(final_dataset, tokenizer)
            print("[COMPLETED] 모델 학습 완료")
            
            try:
                # 모델 성능평가 => HAERAE Benchmark 사용하기 + RAG 성능평가
                result = evaluate_model(base_settings.base_model)

                # v_latest 모델명 변경하기
                total_version_number = len([name for name in os.listdir(base_settings.base_model) if name.startswith("v_")])
                
                os.rename(base_settings.base_model + "/v_latest", base_settings.base_model + f"/v_{total_version_number:03d}")
                # 새로운 모델을 v_latest로 변경하기
                os.rename(base_settings.base_model + "/dpo_model", base_settings.base_model + "/v_latest")

                return {"is_completed": result}
            
            except Exception as e:
                print(f"[ERROR] {e}")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print("⚠️ GPU 메모리 부족 → 배치/길이 줄이기")
                return {"is_completed": False}
            
            elif "device-side assert triggered" in str(e):
                print("⚠️ CUDA Assert 발생 → 데이터셋 인덱스나 라벨 확인하기")
                return {"is_completed": False}

            elif "Tokenizer" in str(e):
                print("⚠️ Tokenizer 문제 → pad_token 설정 확인.")
                return {"is_completed": False}

            else:
                print(f"🚨 Unknown RuntimeError: {e}")
                return {"is_completed": False}
            
        except ValueError as e:
            print(f"⚠️ ValueError: {e}")
            return {"is_completed": False}
        
        except Exception as e:
            print(f"🚨 Unexpected error: {type(e).__name__}: {e}")
            return {"is_completed": False}
    
        except Exception as e:
            print(f"🚨 데이터 처리 로직 없음: {type(e).__name__}: {e}")
            return {"is_completed": False}