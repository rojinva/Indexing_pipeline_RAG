import logging
import azure.functions as func
import os
import json
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import asyncio

from dotenv import load_dotenv
load_dotenv()


# Enable this when want use in dev
# Set up environment variables
OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY_USE2"]
OPENAI_API_BASE = os.environ["AZURE_OPENAI_ENDPOINT_USE2"]
OPENAI_API_VERSION = "2024-12-01-preview"
EMBEDDING_MODEL = "text-embedding-3-small"

JSON_MIME_TYPE = "application/json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize semaphore with a limit of 5 concurrent requests
semaphore = asyncio.Semaphore(5)

# Define retry decorator with tenacity
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def get_embedding(chunk):
    """Fetch embedding from Azure OpenAI service."""
    url = f"{OPENAI_API_BASE}/openai/deployments/{EMBEDDING_MODEL}/embeddings?api-version={OPENAI_API_VERSION}"
    headers = {"Content-Type": JSON_MIME_TYPE, "api-key": OPENAI_API_KEY}
    data = {
        "input": chunk,
    }
    async with semaphore:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


async def process_value(value):
    record_id = value.get("recordId")
    data = value.get("data", {})
    chunk = data.get("chunk", "")
    if record_id is None:
        return None
    if not chunk:
        return {
            "recordId": record_id,
            "errors": [{"message": "'chunk' field is empty."}],
        }
    try:
        embedding = await get_embedding(chunk)
        return {"recordId": record_id, "data": {"embedding": embedding}}
    except Exception as e:
        logging.error(f"Error generating embedding for record {record_id}: {e}")
        return {"recordId": record_id, "errors": [{"message": str(e)}]}


async def process_request(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing request to generate embeddings.")
    try:
        req_body = req.get_json()
        values = req_body.get("values", [])
        results = await asyncio.gather(*[process_value(value) for value in values])
        results = [result for result in results if result is not None]
        return func.HttpResponse(
            json.dumps({"values": results}), mimetype=JSON_MIME_TYPE
        )
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}), status_code=500, mimetype=JSON_MIME_TYPE
        )


# Entry point for Azure Functions
def main(req: func.HttpRequest) -> func.HttpResponse:
    return asyncio.run(process_request(req))
