from .utils import split_text_batch
import azure.functions as func
import asyncio

def main(req: func.HttpRequest) -> func.HttpResponse:
    size = int(req.headers.get("size"))
    overlap = int(req.headers.get("overlap"))

    data = req.get_json()

    # Business logic invocation
    response = split_text_batch(data.get("values"), size, overlap)

    return func.HttpResponse(body=response, mimetype="application/json")
