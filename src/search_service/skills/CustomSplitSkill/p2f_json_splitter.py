import json
from urllib.parse import unquote
from .excel_splitter import generate_hash
from .document_image_processor import get_file_from_adls
import tiktoken # Import tiktoken

try:
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
except Exception as e:
    # Fallback or handle error if encoding is not found
    # For example, you might want to log this and default to character-based trimming or raise an error.
    print(f"Warning: Could not load tiktoken encoding. Error: {e}")
    encoding = None

def load_json_from_blob(blob_uri):
    blob_data = get_file_from_adls(blob_uri)
    if blob_data is None:
        raise ValueError(f"Blob data for {blob_uri} is None")
    try:
        return json.loads(blob_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {blob_uri}: {e}")

def extract_parent_path(blob_uri):
    unquoted_blob_uri = unquote(blob_uri)
    if ".blob.core.windows.net/" in unquoted_blob_uri:
        return unquoted_blob_uri.split(".blob.core.windows.net/", 1)[1]
    return blob_uri

def get_records_to_process(json_data, blob_uri):
    if isinstance(json_data, dict):
        return [json_data]
    elif isinstance(json_data, list):
        return json_data
    else:
        raise ValueError(f"Expected JSON to be an object or a list of objects, but got {type(json_data)} for blob {blob_uri}")

def trim_content_for_chunking(image_summary_text):
    MAX_TOKENS_FOR_EMBEDDING = 7500
    MAX_CHARS_FALLBACK = 20000
    if image_summary_text and encoding:
        tokens = encoding.encode(image_summary_text)
        if len(tokens) > MAX_TOKENS_FOR_EMBEDDING:
            print(f"Warning: Content exceeds {MAX_TOKENS_FOR_EMBEDDING} tokens, trimming to fit.")
            trimmed_tokens = tokens[:MAX_TOKENS_FOR_EMBEDDING]
            print("Trimmed tokens length:", len(trimmed_tokens))
            return encoding.decode(trimmed_tokens)
        return image_summary_text
    elif image_summary_text and not encoding:
        if len(image_summary_text) > MAX_CHARS_FALLBACK:
            return image_summary_text[:MAX_CHARS_FALLBACK]
        return image_summary_text
    return image_summary_text

def extract_extraction_result(extraction_result_value):
    if isinstance(extraction_result_value, dict):
        sequence_data = extraction_result_value.get('sequence')
        instructions_data = extraction_result_value.get('instructions')
        bom_items_data = extraction_result_value.get('bom_items')
    else:
        sequence_data = None
        instructions_data = None
        bom_items_data = None
    return sequence_data, instructions_data, bom_items_data

def build_record(item_dict, usecase, parent_path_with_container_name):
    page_number = item_dict.get('page_number')
    image_summary_text = item_dict.get('image_summary')
    content_for_chunking = trim_content_for_chunking(image_summary_text)
    content_for_hash = image_summary_text if image_summary_text is not None else ""
    extraction_result_value = item_dict.get('extraction_result')
    sequence_data, instructions_data, bom_items_data = extract_extraction_result(extraction_result_value)
    return {
        "content": image_summary_text if image_summary_text is not None else " ",
        "content_for_chunking": content_for_chunking if content_for_chunking is not None else " ",
        "chunk_hash": generate_hash(content_for_hash),
        "use_case": usecase,
        "sheet_name": " ",
        "row": page_number,
        "parent_path": parent_path_with_container_name,
        "sequence": sequence_data if sequence_data is not None else [],
        "instructions": instructions_data if instructions_data is not None else [],
        "bom_items": bom_items_data if bom_items_data is not None else [],
    }

def process_json_file(usecase: str, blob_uri: str):
    """
    Process a JSON file and return a DataFrame.
    """
    try:
        json_data = load_json_from_blob(blob_uri)
        parent_path_with_container_name = extract_parent_path(blob_uri)
        records_to_process = get_records_to_process(json_data, blob_uri)
        output_records = []
        for item_dict in records_to_process:
            if not isinstance(item_dict, dict):
                continue
            record = build_record(item_dict, usecase, parent_path_with_container_name)
            output_records.append(record)
        return output_records
    except Exception as e:
        raise ValueError(f"Failed to process and transform JSON data from {blob_uri} due to: {e}")

