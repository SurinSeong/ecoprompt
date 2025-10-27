from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# 학습된 모델 성능평가 클래스
@register_model("")
class LlamaKoreanLM(LM):

    def __init__(self, **kwargs):
        return
    
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return
    
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return
    
    def generate_until(self, requests: list[Instance]) -> list[str]:
        return