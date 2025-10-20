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
from azure.storage.blob import BlobClient
from .excel_splitter import generate_hash, get_blob_size
import zipfile
from .constants import ColumnNames
# Function to initialize BlobServiceClient
def get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name):
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    return blob_service_client

def extract_fields_from_blob_fmea_agent(blob_uri):
    """
    Extract specified fields from an Excel file in Azure Blob Storage and add a new field "chunk".

    Args:
        blob_uri (str): URI of the blob containing the Excel file.

    Returns:
        list: A list of dictionaries containing "file_name", "chunk", and "updated_date".
    """
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    fields = [
        "Item Function",
        "Potential Failure Mode",
        "Potential Effect(s) of Failure",
        "Severity (S)",
        "Potential Cause(s) / Mechanism(s) of Failure",
        "Failure Category",
        "Current Design Prevention Control & Explanation of Occurence rating",
        "Occurrence (O)",
        "Class",
        "Current Detection Design Controls & Explanation of detection rating",
        "Detection (D)",
        "RPN (S*O*D)"
    ]

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)

    # Unquote the blob URI to handle any special characters like %20
    unquoted_blob_uri = unquote(blob_uri)
    parent_path = unquoted_blob_uri.split(container_name)[1]

    # Create BlobClient to download blob data
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=parent_path)
    blob_data = blob_client.download_blob().content_as_bytes()

    # Load Excel file into a BytesIO object
    file_buffer = io.BytesIO(blob_data)

    # Read the first few rows to determine the header row
    preview_df = pd.read_excel(file_buffer, sheet_name='FMEA', nrows=5)
    
    # Determine the header row based on the presence of "Item Function" or "Intended Function"
    header_row = 3 if "Item Function" in preview_df.iloc[2].values or "Intended Function" in preview_df.iloc[2].values else 4

    # Read the Excel file, skipping the first 3 rows and using rows 4 and 5 as headers
    df = pd.read_excel(file_buffer, sheet_name='FMEA', skiprows=header_row).iloc[:, 2:15]

    # Replace NaN values with empty strings
    df.fillna('', inplace=True)

    # Create the "chunk" field
    df['chunk'] = df.apply(lambda row: {field: row[field] for field in fields if field in df.columns}, axis=1)

    # Prepare the result list
    result_list = []
    for _, row in df.iterrows():
        result = {
            'file_name': parent_path.split('/')[-1],
            'chunk': str(row['chunk']),
            'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        result_list.append(result)

    return result_list

def extract_fields_from_blob_fmea_agent_updated(blob_uri):
    """
    Extract specified fields from an Excel file in Azure Blob Storage and add a new field "chunk".

    Args:
        blob_uri (str): URI of the blob containing the Excel file.

    Returns:
        list: A list of dictionaries containing "file_name", "chunk", and "updated_date".
    """
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)

    # Unquote the blob URI to handle any special characters like %20
    unquoted_blob_uri = unquote(blob_uri)
    parent_path = unquoted_blob_uri.split(container_name)[1]

    # Create BlobClient to download blob data
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=parent_path)
    blob_data = blob_client.download_blob().content_as_bytes()

    # Load Excel file into a BytesIO object
    file_buffer = io.BytesIO(blob_data)

    df = pd.read_excel(file_buffer)

    # List to hold multiple metadata dictionaries
    result_list = []

    # Iterate through the DataFrame row by row
    for i in range(len(df)):
        # Create a dictionary for the current row
        fmea_metadata = {
            'file_name': str(df['File_Name'].iloc[i]),
            'metadata': str(df['Metadata'].iloc[i]),
            'failurecategory': str(df['Failure Category'].iloc[i]),
            'rpnscore': int(df['RPN (S*O*D)'].iloc[i]),
            'chunk': str(df['chunk'].iloc[i]),
            'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        result_list.append(fmea_metadata)

    return result_list

def get_blob_data(blob_uri):
    """Download blob data from Azure."""
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]

    cred = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )

    blob = BlobClient.from_blob_url(blob_uri, credential=cred)
    
    try:
        blob_data = blob.download_blob().content_as_bytes()
        return blob_data
    except Exception as e:
        print("An error occurred while accessing the blob client.")
        print(f"Error details: {e}")
        raise

def parse_blob_uri(blob_uri, container_name):
    """Parse the blob URI to extract metadata."""
    unquoted_blob_uri = unquote(blob_uri)
    parent_path = unquoted_blob_uri.split(container_name)[1]
    parent_path_with_container_name = unquoted_blob_uri.split(".blob.core.windows.net/")[1]
    file_name = parent_path_with_container_name.split("/")[-1]
    use_case = parent_path_with_container_name.split("/")[1]
    return parent_path, parent_path_with_container_name, file_name, use_case

def load_excel_file(blob_data):
    """Load the Excel file from blob data."""
    file_buffer = io.BytesIO(blob_data)
    file_buffer.seek(0)
    magic_bytes = file_buffer.read(8)
    file_buffer.seek(0)

    if magic_bytes == b'':
        print("EMPTY")
        return None

    if zipfile.is_zipfile(file_buffer):
        return pd.ExcelFile(file_buffer, engine="openpyxl")
    else:
        raise ValueError("Not a zip-based xlsx file.")

def find_target_sheet(xls):
    """Find the target sheet starting with 'FMEA'."""
    for name in xls.sheet_names:
        if name.strip()=="FMEA":
            return name
    return None

def find_header_row(preview_df, target_keywords):
    """
    Find the header row in the preview DataFrame.

    Args:
        preview_df (pd.DataFrame): DataFrame containing the preview of the Excel sheet.
        target_keywords (set): Set of keywords to identify the header row.

    Returns:
        tuple: (header_row, skip_columns) where header_row is the row index of the header,
               and skip_columns is the number of columns to skip.
    """
    max_rows = min(10, len(preview_df))
    max_cols = min(10, preview_df.shape[1])
    header_row = None
    skip_columns = 0

    # Check if the first row contains any of the target keywords in column names
    if any(keyword in preview_df.columns for keyword in target_keywords):
        header_row = 0
        skip_columns = 0
        return header_row, skip_columns

    # Loop through the first few rows to find the header row
    for i in range(max_rows):
        for j in range(max_cols):
            cell_value = str(preview_df.iat[i, j]).strip()
            if cell_value in target_keywords:
                header_row = i + 1  # Adjust for zero-based indexing
                skip_columns = j
                return header_row, skip_columns

    return None, None

def clean_and_rename_columns(df):
    """Clean and rename columns in the DataFrame."""
    df.columns = df.columns.str.replace("\n", "", regex=False).str.strip()

    if ColumnNames.RECOMMENDED_ACTION in df.columns:
        df.rename(columns={ColumnNames.RECOMMENDED_ACTION: ColumnNames.RECOMMENDED_CORRECTIVE_ACTIONS}, inplace=True)

    if ColumnNames.CORRECTIVE_ACTIONS in df.columns:
        df.rename(columns={ColumnNames.CORRECTIVE_ACTIONS: ColumnNames.RECOMMENDED_CORRECTIVE_ACTIONS}, inplace=True)
        df = df.iloc[1:]

    if ColumnNames.INTENDED_FUNCTION in df.columns:
        df.rename(columns={ColumnNames.INTENDED_FUNCTION: ColumnNames.ITEM_FUNCTION}, inplace=True)

    return df
def find_target_sheet_poc(xls):
    """Find the target sheet starting with 'FMEA'."""
    target_sheet = None
    # metadata_sheet=None
    for name in xls.sheet_names:
        if name.strip()=="FMEA":
            target_sheet = name
    return target_sheet

def extract_fields_from_blob_fmea_poc(blob_uri):
    container_name = "knowledge-mining"
    fields = [
        ColumnNames.ITEM_FUNCTION,
        ColumnNames.POTENTIAL_FAILURE_MODE,
        ColumnNames.POTENTIAL_EFFECTS_OF_FAILURE,
        ColumnNames.SEVERITY,
        ColumnNames.POTENTIAL_CAUSES,
        ColumnNames.FAILURE_CATEGORY,
        ColumnNames.CURRENT_DESIGN_PREVENTION_CONTROL,
        ColumnNames.OCCURRENCE,
        ColumnNames.CLASS,
        ColumnNames.CURRENT_DETECTION_DESIGN_CONTROLS,
        ColumnNames.DETECTION,
        ColumnNames.RPN,
        ColumnNames.RECOMMENDED_CORRECTIVE_ACTIONS
    ]

    # Download blob data
    blob_data = get_blob_data(blob_uri)

    # Parse blob URI
    parent_path, parent_path_with_container_name, file_name, use_case = parse_blob_uri(blob_uri, container_name)

    #Load Excel file
    xls = load_excel_file(blob_data)
    if not xls:
        return

    # Find target sheet
    target_sheet = find_target_sheet_poc(xls)
    if not target_sheet:
        print(f"Error: No sheet starting with 'FMEA' found in {parent_path}")
        return

    # Find header row
    preview_df = pd.read_excel(xls, sheet_name=target_sheet, nrows=10, engine="openpyxl")
    # metadata_df=pd.read_excel(xls, sheet_name=metadata_sheet, engine="openpyxl")
    target_keywords = {
    ColumnNames.ITEM_FUNCTION.value,
    ColumnNames.INTENDED_FUNCTION.value,
    "\n" + ColumnNames.ITEM_FUNCTION.value,
    "\n" + ColumnNames.INTENDED_FUNCTION.value
    }
    # metadata_dict = pd.Series(metadata_df.Details.values, index=metadata_df.Items).to_dict()
    # metadata_dict = {key.split('(')[0].strip(): value for key, value in metadata_dict.items()}

    
    header_row, skip_columns = find_header_row(preview_df, target_keywords)
    if header_row is None:
        print("Header not found")
        return

    #  Load DataFrame and clean columns
    df = pd.read_excel(xls, sheet_name=target_sheet, skiprows=header_row, engine='openpyxl').iloc[:, skip_columns:]
    df = clean_and_rename_columns(df)

    # : Drop rows with null values in 'Item Function'
    df = df.dropna(subset=[ColumnNames.ITEM_FUNCTION])

    #  Create the "chunk" field
    df['chunk'] = df.apply(lambda row: {field.value: row[field.value] for field in fields if field.value in df.columns}, axis=1)
    
    # Prepare the result list
    # component_key=next((key for key in metadata_dict if key.startswith("Component")))
    chunks_with_metadata = []
    for _, row in df.iterrows():
        try:
            rpn = int(row['chunk'].get('RPN (S*O*D)', 0))
        except (ValueError, TypeError):
            rpn = 0  

        try:
            detection = int(row['chunk'].get('Detection (D)', 0))  
        except (ValueError, TypeError):
            detection = 0  

        try:
            occurrence = int(row['chunk'].get('Occurrence (O)', 0))  
        except (ValueError, TypeError):
            occurrence = 0  

        try:
            severity = int(row['chunk'].get('Severity (S)', 0))
        except (ValueError, TypeError):
            severity = 0
        Class = row['chunk'].get('Class', 'X')  # Default to 'X' if not present
        if pd.isna(Class):  # Check if Class is NaN
            Class = 'X'
        roman_to_int = {
            "I": 1,
            "II": 2,
            "III": 3,
            "X": 10,
            "x":10
        }    
        Class = roman_to_int.get(Class, Class)
        failure_category = row['chunk'].get('Failure Category', "N/A")  # Default to "N/A" if not present
        if pd.isna(failure_category):  # Check if Failure Category is NaN
            failure_category = ""
        
        result = {
        'content': str(row['chunk']),
        "chunk_hash": generate_hash(str(row['chunk']).strip()),
        "use_case": use_case,
        "sheet_name": "FMEA",
        # "row":-1 ,
        "parent_path":parent_path_with_container_name,
        # "PG":metadata_dict.get("PG"),
        # "BU":metadata_dict.get("BU"),
        # "tool":metadata_dict.get("Tool"),
        # "component":metadata_dict.get(component_key,"N/A"),
        # "part_no":metadata_dict.get("Part#"),
        "RPN": rpn,
        "Detection": detection,
        "Occurrence": occurrence,
        "Severity": severity,
        "Class": Class,
        "failure_category": failure_category,
        }
        chunks_with_metadata.append(result)

    return chunks_with_metadata

def extract_fields_from_blob_fmea_agent_conversational(blob_uri):
    container_name = "knowledge-mining"
    fields = [
        ColumnNames.ITEM_FUNCTION,
        ColumnNames.POTENTIAL_FAILURE_MODE,
        ColumnNames.POTENTIAL_EFFECTS_OF_FAILURE,
        ColumnNames.SEVERITY,
        ColumnNames.POTENTIAL_CAUSES,
        ColumnNames.FAILURE_CATEGORY,
        ColumnNames.CURRENT_DESIGN_PREVENTION_CONTROL,
        ColumnNames.OCCURRENCE,
        ColumnNames.CLASS,
        ColumnNames.CURRENT_DETECTION_DESIGN_CONTROLS,
        ColumnNames.DETECTION,
        ColumnNames.RPN,
        ColumnNames.RECOMMENDED_CORRECTIVE_ACTIONS
    ]

    # Download blob data
    blob_data = get_blob_data(blob_uri)

    # Parse blob URI
    parent_path, parent_path_with_container_name, file_name, use_case = parse_blob_uri(blob_uri, container_name)

    #Load Excel file
    xls = load_excel_file(blob_data)
    if not xls:
        return

    # Find target sheet
    target_sheet = find_target_sheet_poc(xls)
    if not target_sheet:
        print(f"Error: No sheet starting with 'FMEA' found in {parent_path}")
        return

    # Find header row
    preview_df = pd.read_excel(xls, sheet_name=target_sheet, nrows=10, engine="openpyxl")
    target_keywords = {
    ColumnNames.ITEM_FUNCTION.value,
    ColumnNames.INTENDED_FUNCTION.value,
    "\n" + ColumnNames.ITEM_FUNCTION.value,
    "\n" + ColumnNames.INTENDED_FUNCTION.value
    }
    
    header_row, skip_columns = find_header_row(preview_df, target_keywords)
    if header_row is None:
        print("Header not found")
        return

    df = pd.read_excel(xls, sheet_name=target_sheet, skiprows=header_row, engine='openpyxl').iloc[:, skip_columns:]
    df = clean_and_rename_columns(df)

    df = df.dropna(subset=[ColumnNames.ITEM_FUNCTION])

    df['chunk'] = df.apply(lambda row: {field.value: row[field.value] for field in fields if field.value in df.columns}, axis=1)
    
    result_list = []
    for _, row in df.iterrows():
        
        result = {
            'file_name': file_name,
            'chunk': str(row['chunk']),
            'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        }
        result_list.append(result)
    return result_list