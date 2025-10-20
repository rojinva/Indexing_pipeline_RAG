import json
from urllib.parse import unquote
import hashlib
from .text_splitter import split_text_into_chunks
from .excel_splitter import split_excel_into_chunks, generate_hash
from .document_intelligence_splitter import (
    split_powerpoint_pdf_into_chunks,
    split_word_documents_into_chunks,
)
import asyncio
import time

async def split_text_single(value, size, overlap, semaphore):
    # This is used in split_text_batch

    # Validate the input
    try:
        assert "recordId" in value
        record_id = value["recordId"]
    except AssertionError:
        return None

    # Validate the input
    try:
        assert "data" in value, "Field 'data' is required."
        data = value["data"]
        for field in ["content", "parent_filename", "blob_uri"]:
            assert field in data, f"Field '{field}' is required."
    except AssertionError as error:
        return {
            "recordId": record_id,
            "data": {},
            "errors": [{"message": f"Error: {error.args[0]}"}],
        }

    # Perform the operations
    parent_filename = data["parent_filename"]
    blob_uri = data["blob_uri"]
    content = data["content"]
    is_processed_using_custom_processing = False
    warning = []
    try:
        if (
            parent_filename.lower().endswith((".xls", ".xlsx"))
            and "customerSurvey" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="customerSurvey",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".xls", ".xlsx"))
            and "commonSpec" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="commonSpec",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".xlsx", ".xlsm"))
            and "engineeringElt" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="engineeringElt",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".doc", ".docx"))
            and "engineeringElt" in blob_uri
        ):
            
            chunks_with_metadata = await split_word_documents_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".pdf", ".ppt", ".pptx"))
            and "engineeringElt" in blob_uri
        ):
            chunks_with_metadata = await split_powerpoint_pdf_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".pdf", ".ppt", ".pptx"))
            and "iPLM" in blob_uri
        ):
            chunks_with_metadata = await split_powerpoint_pdf_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".doc", ".docx"))
            and "LightCAEBot" in blob_uri
        ):
            chunks_with_metadata = await split_word_documents_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".pdf", ".ppt", ".pptx"))
            and "LightCAEBot" in blob_uri
        ):
            chunks_with_metadata = await split_powerpoint_pdf_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
    except Exception as e:
        warning = [
            {
                "message": f"Could not complete operation for record using Custom Processing. Reason: {str(e)}. File Path: {blob_uri}"
            }
        ]

    if not is_processed_using_custom_processing:
        chunks = split_text_into_chunks(
            content=content, chunk_size=size, chunk_overlap=overlap
        )
        unquoted_blob_uri = unquote(blob_uri)
        parent_path_with_container_name = unquoted_blob_uri.split(
            ".blob.core.windows.net/"
        )[1]
        use_case = parent_path_with_container_name.split("/")[1]
        chunks_with_metadata = []
        for text in chunks:
            data = {
                "content": text,
                "chunk_hash": generate_hash(text),
                "use_case": use_case,
                "sheet_name": "",
                "row": -1,
                "parent_path": parent_path_with_container_name,
            }
            chunks_with_metadata.append(data)

    return {
        "recordId": record_id,
        "data": {"chunks": chunks_with_metadata},
        "errors": [],
        "warnings": warning,
    }


async def split_text_batch(values, size, overlap):
    """
    Custom split text function
    """
    response = {}
    # setting the concurrency to 5
    semaphore = asyncio.Semaphore(5)
    response["values"] = []
    print("Length: ", len(values))
    start_time = time.time()
    tasks = [asyncio.create_task(split_text_single(value, size, overlap, semaphore)) for value in values] 
    results = await asyncio.gather(*tasks)
    print("Time taken: ", time.time() - start_time)
    response["values"] = results
    return json.dumps(response)
