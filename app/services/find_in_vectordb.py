import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url="https://c5c72aa1-571c-4980-8091-f3fe5f10b794.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key=QDRANT_API_KEY,
)

print(qdrant_client.get_collections())