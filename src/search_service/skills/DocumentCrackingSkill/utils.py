import json
from urllib.parse import unquote
import hashlib
from .text_splitter import split_text_into_chunks

def generate_hash(text):
    hash_object = hashlib.sha256(text.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex

def split_text_single(value, size, overlap):
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
    blob_uri = data["blob_uri"]
    content = data["content"]
    warning = []
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


def split_text_batch(values, size, overlap):
    """
    Custom split text function
    """
    response = {}
    response["values"] = []

    for value in values:
        output_value = split_text_single(value, size, overlap)
        if output_value is not None:
            response["values"].append(output_value)

    return json.dumps(response)
