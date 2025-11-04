# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path="./local-models/Llama-SSAFY-8B/v_2",
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     pretrained_model_name_or_path="./local-models/Llama-SSAFY-8B/v_2",
# )

# model.save_pretrained("../ecoprompt")
# tokenizer.save_pretrained("../ecoprompt")

from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGING_FACE_API_KEY"))

api = HfApi()

api.upload_folder(
    folder_path="./local-models/Llama-SSAFY-8B/v_2",
    repo_id="tnfls/ecoprompt",
    repo_type="model"
)