import os

from dotenv import load_dotenv

load_dotenv(override=True)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"
API_VERSION = "2024-12-01-preview"


llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY_USE2"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_USE2"],
    api_version=API_VERSION,
    azure_deployment="gpt-4o-gs",
    model="gpt-4o",
    temperature=0.0,
    streaming=False,
    max_retries=5,
    timeout=180.0,
    max_tokens=2000,
)

embedding = AzureOpenAIEmbeddings(
    api_key=os.environ["AZURE_OPENAI_API_KEY_USE2"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_USE2"],
    api_version=API_VERSION,
    model=EMBEDDING_MODEL,
    max_retries=5,
    timeout=180.0,
)
