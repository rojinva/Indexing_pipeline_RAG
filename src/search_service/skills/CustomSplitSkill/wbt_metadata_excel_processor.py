import pandas as pd
import ast
from datetime import datetime
import io
import os
import json
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from urllib.parse import unquote
import openpyxl
from .constants import blob_uri_prefix_constant

def load_excel_and_get_parent_path(blob_service_client, container_name, blob_uri, sheet_name):
    
    unquoted_blob_uri = unquote(blob_uri)
    parent_path_with_container_name = unquoted_blob_uri.split(blob_uri_prefix_constant)[1]

    # Create BlobClient and download blob data
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=unquoted_blob_uri.split(container_name)[1])
    blob_data = blob_client.download_blob().content_as_bytes()

    # Read Excel file into DataFrame
    file_buffer = io.BytesIO(blob_data)
    df = pd.read_excel(file_buffer, sheet_name=sheet_name)
    
    return df, parent_path_with_container_name
    

# Function to initialize BlobServiceClient
def get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name):
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    return blob_service_client


def extract_metadata_chunks(df, metadata_fields, chunk_fields):
    result = []
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Create a dictionary for the current row
        row_dict = {}
        chunk_dict = {}
        
        # Add metadata fields to the dictionary
        for field in metadata_fields:
            if field in df.columns:
                row_dict[field.lower().replace(" ", "_")] = row[field]
        
        # Add chunk fields to the dictionary
        for field in chunk_fields:
            if field in df.columns:
                chunk_dict[field] = row[field]
                
        
        row_dict["chunk"] = str(chunk_dict)
        row_dict["chunk_hash"] = str(chunk_dict)
        row_dict["chunk_id"] = row.get("Training Object ID", None)
        row_dict["parent_filename"] = row.get('Training Title', None)
        row_dict["parent_url"] = f"""https://lamresearch.csod.com/samldefault.aspx?ouid=1&returnUrl=%252fDeepLink%252fProcessRedirect.aspx%253fmodule%253dlodetails%2526lo%253d{row.get("Training Object ID", None)}"""
        
        # Append the dictionary to the result list
        result.append(row_dict)
    
    return result

# Read_excel_metadata to read from Azure Blob to create text2sql metadata index
def wbt_metadata_splitter_from_blob(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)

    df, parent_path_with_container_name = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="Leap + NERDS Trainings")
    
    
    chunk_fields = ['Course Owner Group', 'Training Item ID', 'Training Type', 'Assignment Type', 'Export Control/ Trade Restricted', 'Training Subject', 'Training Provider', 'Training Title', 'Target Audience', 'Training Active', 'Training Provider Active', 'Training Description', 'Training Hours', 'Keyword']
    metadata_fields = ['Training Subject', 'Training Provider', 'Target Audience', 'Item ID', 'Training ID', "Training Object ID"]


    chunks_with_metadata = extract_metadata_chunks(df, metadata_fields, chunk_fields)
    
    
    return chunks_with_metadata
    
    
