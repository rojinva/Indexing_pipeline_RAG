import os
import io
import os
import io
import openpyxl
import pandas as pd
import hashlib
from PyPDF2 import PdfReader
from urllib.parse import unquote
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from .text_splitter import split_text_into_chunks


def get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name):
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=credential
    )
    return blob_service_client


def generate_hash(text):
    hash_object = hashlib.sha256(text.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex

def get_blob_bytes_from_uri(blob_uri: str):
    """
    Retrieve the blob data and metadata from Azure Blob Storage based on the provided URI.
    """
    # Retrieve environment variables
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Initialize the BlobServiceClient
    blob_service_client = get_blob_service_client(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        storage_account_name=storage_account_name,
    )

    # Clean the blob URI
    unquoted_blob_uri = unquote(blob_uri)
    parent_path = unquoted_blob_uri.split(container_name)[1]
    parent_path_with_container_name = unquoted_blob_uri.split(".blob.core.windows.net/")[1]

    # Retrieve the blob data
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=parent_path)
        blob_data = blob_client.download_blob().content_as_bytes()
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve blob data: {e}")

    # Return the relevant paths and blob data
    return parent_path_with_container_name, parent_path, blob_data

def split_pdf_file_into_chunks(blob_uri, chunk_size, chunk_overlap):
    parent_path_with_container_name, parent_path, blob_data = get_blob_bytes_from_uri(blob_uri)
    use_case = parent_path.split("/")[1]

    # Prepare the file buffer
    file_buffer = io.BytesIO(blob_data)
    chunks_with_metadata = []

    # Read the PDF file
    reader = PdfReader(file_buffer)

    # Extract text from each page and split into chunks
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        chunked_text = split_text_into_chunks(
            content=page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for text in chunked_text:
            data = {
                "content": text,
                "chunk_hash": generate_hash(text),
                "use_case": use_case,
                "sheet_name": "",  # Not applicable for PDF
                "row": page_number,  # Page number
                "parent_path": parent_path_with_container_name,
            }
            chunks_with_metadata.append(data)

    return chunks_with_metadata