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

# Read_excel_metadata to read from Azure Blob to create text2sql metadata index
def metadata_splitter_from_blob(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)

    df, parent_path_with_container_name = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name=0)

    # List to hold all metadata chunks
    metadata_list = []

    # Initialize chunking variables
    current_table_name = None
    current_table_description = None
    current_chunk_columns = []
    column_details = {}
    categorical_column_details = {}
    sample_rows = {}

    def finalize_chunk():
        """Helper function to finalize and append a chunk to metadata_list."""
        if current_chunk_columns:
            current_chunk_categorical_details = {key: categorical_column_details.get(key, {}) for key in current_chunk_columns}
            current_chunk_sample_rows = {key: sample_rows.get(key, []) for key in current_chunk_columns}
            metadata_chunk = {
                'table_name': current_table_name,
                'table_description': current_table_description,
                'chunk': json.dumps({key: column_details[key] for key in current_chunk_columns}),
                'categorical_column_details': current_chunk_categorical_details,
                'sample_rows': current_chunk_sample_rows,
                'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'parent_path': parent_path_with_container_name
            }
            metadata_list.append(metadata_chunk)

    # Process rows to detect table changes and create chunks
    for _, row in df.iterrows():
        table_name = row['TableName']
        column_name = row['ColumnName']
        column_description = row['ColumnDetails']
        sample_data = row['sample_data']
        categorical_fields_value = row['categorical_fields_value']

        # Detect table change
        if table_name != current_table_name:
            # Finalize current chunk before starting a new table
            finalize_chunk()
            current_chunk_columns = []
            current_table_name = table_name
            current_table_description = f"{table_name} {row['TableDescription']}" if pd.notna(row['TableDescription']) else table_name

        # Update column details
        column_details[column_name] = str(column_description) if pd.notna(column_description) else ""

        # Parse sample data
        if isinstance(sample_data, str) and sample_data.strip():
            try:
                sample_data_dict = ast.literal_eval(sample_data)
                sample_rows[column_name] = sample_data_dict.get(column_name, [])
            except (SyntaxError, ValueError):
                sample_rows[column_name] = []
        else:
            sample_rows[column_name] = []

        # Parse categorical details
        if pd.notna(categorical_fields_value):
            try:
                categorical_column_details[column_name] = ast.literal_eval(categorical_fields_value)
            except (SyntaxError, ValueError):
                categorical_column_details[column_name] = {}

        # Add column to current chunk
        current_chunk_columns.append(column_name)

        # Finalize chunk if it reaches 10 columns
        if len(current_chunk_columns) >= 10:
            finalize_chunk()
            current_chunk_columns = []

    # Finalize any remaining chunk after the loop
    finalize_chunk()

    return metadata_list

# Read_excel_metadata to read from Azure Blob to create text2sql metadata index - New
def metadata_splitter_from_blob_new(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    df, parent_path_with_container_name = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name=0)

    # List to hold all metadata chunks
    metadata_list = []

    # Initialize chunking variables
    current_table_name = None
    current_table_description = None
    current_chunk_columns = []
    column_details = {}
    categorical_column_details_v1 = {}
    categorical_column_details_v2 = {}
    categorical_column_details_v3 = {}
    sample_rows = {}

    def finalize_chunk():
        """Helper function to finalize and append a chunk to metadata_list."""
        if current_chunk_columns:
            metadata_chunk = {
                'table_name': current_table_name,
                'table_description': current_table_description,
                'chunk': json.dumps({key: str(column_details[key]) for key in current_chunk_columns}),
                'categorical_column_details_v1': {key: categorical_column_details_v1.get(key, {}) for key in current_chunk_columns} or {},
                'categorical_column_details_v2': {key: categorical_column_details_v2.get(key, {}) for key in current_chunk_columns} or {},
                'categorical_column_details_v3': {key: categorical_column_details_v3.get(key, {}) for key in current_chunk_columns} or {},
                'sample_rows': {key: sample_rows.get(key, []) for key in current_chunk_columns} or {},
                'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'parent_path': parent_path_with_container_name
            }
            metadata_list.append(metadata_chunk)

    # Process rows to detect table changes and create chunks
    for _, row in df.iterrows():
        table_name = row['TableName']
        column_name = row['ColumnName']
        column_description = row['ColumnDetails']
        sample_data = row['sample_data']
        
        # Ensure consistency in field names
        categorical_fields_value_v1 = row.get('categorical_fields_value_v1', None)
        categorical_fields_value_v2 = row.get('categorical_fields_value_v2', None)
        categorical_fields_value_v3 = row.get('categorical_fields_value_v3', None)

        # Detect table change
        if table_name != current_table_name:
            finalize_chunk()
            current_chunk_columns = []
            current_table_name = table_name
            current_table_description = f"{table_name} {row['TableDescription']}" if pd.notna(row['TableDescription']) else table_name

        # Update column details
        column_details[column_name] = str(column_description) if pd.notna(column_description) else ""

        # Parse sample data
        if isinstance(sample_data, str) and sample_data.strip():
            try:
                sample_data_dict = ast.literal_eval(sample_data)
                sample_rows[column_name] = sample_data_dict if isinstance(sample_data_dict, dict) else []
            except (SyntaxError, ValueError):
                sample_rows[column_name] = []
        else:
            sample_rows[column_name] = []

        # Parse categorical details for all three fields
        categorical_column_details_v1[column_name] = ast.literal_eval(categorical_fields_value_v1) if isinstance(categorical_fields_value_v1, str) and categorical_fields_value_v1.strip() else {}
        categorical_column_details_v2[column_name] = ast.literal_eval(categorical_fields_value_v2) if isinstance(categorical_fields_value_v2, str) and categorical_fields_value_v2.strip() else {}
        categorical_column_details_v3[column_name] = ast.literal_eval(categorical_fields_value_v3) if isinstance(categorical_fields_value_v3, str) and categorical_fields_value_v3.strip() else {}

        # Add column to current chunk
        current_chunk_columns.append(column_name)

        # Finalize chunk if it reaches 10 columns
        if len(current_chunk_columns) >= 10:
            finalize_chunk()
            current_chunk_columns = []

    # Finalize any remaining chunk after the loop
    finalize_chunk()

    return metadata_list

# Sample Queries Metadata indexing - New (change in chunk_size from 5 to 1)
def read_sample_queries_from_blob(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="Sample_Queries")

    # List to hold multiple metadata dictionaries
    sample_queries_metadata_list = []

    # Initialize variables to keep track of the previous table_name and chunk
    previous_table_name = None
    question_sql_queries = []
    chunk_size = 1 

    # Iterate through the DataFrame row by row
    for i in range(len(df)):
        # Get the current row's table_name
        table_name = df['Table_Name'].iloc[i]

        # If the table_name has changed or it's the first row, start a new chunk
        if table_name != previous_table_name:
            # If it's not the first chunk, finalize the previous chunk
            if previous_table_name is not None:
                sample_queries_metadata = {
                    'table_name': previous_table_name,
                    'chunk': str(question_sql_queries),
                    'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                sample_queries_metadata_list.append(sample_queries_metadata)

            # Start a new chunk for the new table_name
            previous_table_name = table_name
            question_sql_queries = []

        # Process the current record
        user_query = df['User_Query'].iloc[i]
        sql_query = df['SQL_Query'].iloc[i].strip('\"')

        # Append the combined user query and SQL query to the list
        question_sql_queries.append(f"{user_query} {sql_query}")

        # If the chunk size is reached, finalize the chunk and start a new one
        if len(question_sql_queries) >= chunk_size:
            sample_queries_metadata = {
                'table_name': table_name,
                'chunk': str(question_sql_queries),
                'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            sample_queries_metadata_list.append(sample_queries_metadata)
            question_sql_queries = [] 

    # If there are remaining records in the last chunk, add them
    if question_sql_queries:
        sample_queries_metadata = {
            'table_name': previous_table_name,
            'chunk': str(question_sql_queries),
            'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        sample_queries_metadata_list.append(sample_queries_metadata)

    return sample_queries_metadata_list

# NCE USE Case Sample Query Metadata Processing
def read_sample_queries_from_blob_updated(blob_uri, chunk_size=1):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="Sample_Queries")


    # List to hold multiple metadata dictionaries
    sample_queries_metadata_list = []

    # Get TableName
    table_name = df['Table_Name'].iloc[0]

    # Iterate through the DataFrame in chunks of 5 records
    for i in range(0, len(df), chunk_size):
        # Create a new dictionary for the current chunk
        sample_queries_metadata = {}
        sample_queries_metadata['table_name'] = table_name

        # Initialize the list for Question_SQL_Query
        question_sql_queries = []

        # Process the chunk of records
        for j in range(i, min(i + chunk_size, len(df))):
            user_query = df['User_Query'].iloc[j]
            sql_query = df['SQL_Query'].iloc[j].strip('\"')

            # Append the combined user query and SQL query to the list
            question_sql_queries.append(f"{user_query}: {sql_query}")

        # Assign the list to Question_SQL_Query
        sample_queries_metadata['chunk'] = str(question_sql_queries)
        sample_queries_metadata['updated_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Append the dictionary to the list
        sample_queries_metadata_list.append(sample_queries_metadata)

    return sample_queries_metadata_list

def read_metadata_from_blob_updated(blob_uri, chunk_size=1):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="Tables_Metadata")


    # List to hold multiple metadata dictionaries
    table_metadata_list = []

    # Get TableName
    table_name = df['TableName'].iloc[0]

    # Iterate through the DataFrame in chunks 
    for i in range(0, len(df), chunk_size):
        # Create a new dictionary for the current chunk
        table_metadata = {}
        table_metadata['table_name'] = table_name

        # Initialize the list for Table_Metadata
        table_metadata_chunk = []

        # Process the chunk of records
        for j in range(i, min(i + chunk_size, len(df))):
            table_name = df['TableName'].iloc[j]
            column_name = df['ColumnName'].iloc[j]
            column_detail = df['ColumnDetails'].iloc[j]


            # Append the combined user query and SQL query to the list
            table_metadata_chunk.append(f"Table {table_name} - Column {column_name}: {column_detail}")

        # Assign the list to table_metadata
        table_metadata['chunk'] = str(table_metadata_chunk)
        table_metadata['updated_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Append the dictionary to the list
        table_metadata_list.append(table_metadata)

    return table_metadata_list


def read_ddl_metadata_from_blob(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="DDL")

    # List of schemas
    ddl_metadata_list = []

    # Iterate over rows in the DataFrame
    for _, row in df.iterrows():
        # Create a metadata dictionary for each row
        ddl_metadata = {
            'table_name': row['Table_Name'],
            'chunk': row['Schema'],
            'Updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # Append the dictionary to the list
        ddl_metadata_list.append(ddl_metadata)

    return ddl_metadata_list

def read_new_text2sql_metadata_from_blob(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    queries_df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="queries")
    tableinfo_df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="tablesInfo")
    sampledata_df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="sampledata")

    # Select only the 'Table_Name' and 'TABLEINFO_SAMPLE_QUERIES' columns from tableinfo_df
    tableinfo_df = tableinfo_df[['Table_Name', 'TABLEINFO_SAMPLE_QUERIES']]

    # Merge the dataframes on 'Table_Name'
    merged_df = pd.merge(queries_df, tableinfo_df, on='Table_Name', how='left', validate="1:m")

    # Function to map sample_data
    def map_sample_data(row):
        table_name = row['Table_Name']
        table_columns = row['TABLE_COLUMNS']
        
        # Split the TABLE_COLUMNS field into individual column names
        column_names = table_columns.split(',')
        
        # List to store the sample data strings
        sample_data_list = []
        
        # Iterate over each column name
        for column_name in column_names:
            column_name = column_name.strip()
            
            # Find matching rows in the sampledata_df
            matching_rows = sampledata_df[
                (sampledata_df['Table_Name'] == table_name) &
                (sampledata_df['Column_Name'] == column_name)
            ]
            
            # Add the sample_data to the list as a string
            for _, match_row in matching_rows.iterrows():
                sample_data_list.append(match_row['sample_data'])
        
        return sample_data_list

    # Apply the function to each row in merged_df
    merged_df['sample_data_mapped'] = merged_df.apply(map_sample_data, axis=1)

    # List to hold the final output
    final_output = []

    # Iterate through the merged DataFrame row by row
    for _, row in merged_df.iterrows():
        # Create the chunk string
        chunk = f"TABLEINFO_SAMPLE_QUERIES: {row['TABLEINFO_SAMPLE_QUERIES']}"
        
        sample_data_str = str(row['sample_data_mapped'])

        output = {
            'table_name': row['Table_Name'],
            'user_query': row['User_Query'],
            'sql_query': row['SQL_Query'],
            'table_columns': row['TABLE_COLUMNS'],
            'tableinfo_sample_queries': row['TABLEINFO_SAMPLE_QUERIES'],
            'sample_data': sample_data_str,
            'chunk': chunk
        }
        final_output.append(output)

    return final_output

def read_new_text2sql_metadata_v2_from_blob(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    queries_df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="queries")
    tables_info_df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="tablesInfo")
    sample_data_df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name="sampledata")

    
    def build_metadata(sample_data_df, tables_info_df, queries_df):
        metadata_list = []
        
        for _, query_row in queries_df.iterrows():
            user_question = query_row['User_Query']
            user_query = query_row['SQL_Query']
            tables_used = query_row['Table_Name'].split(',')
            columns_used = [column.strip() for column in query_row['TABLE_COLUMNS'].split(',')]
            
            table_info_dict = {}
            columns_info_dict = {}
            sample_queries_dict = {}
            
            for table in tables_used:
                table_info = tables_info_df[tables_info_df['Table_Name'] == table]
                if not table_info.empty:
                    table_info = table_info.iloc[0]
                    table_metadata = {
                        "TABLE DESCRIPTION": table_info['Description'],
                        "ALL COLUMNS IN TABLE": table_info['Columns'],
                        "TABLE PRIMARY KEY": table_info['Primary Key']
                    }
                    table_info_dict[table] = table_metadata
                
                sample_queries = tables_info_df[tables_info_df['Table_Name'] == table]
                if not sample_queries.empty:
                    sample_queries_list = [
                        {"Question": table_info['Question1'], "Query": table_info['Query1']},
                        {"Question": table_info['Question2'], "Query": table_info['Query2']},
                        {"Question": table_info['Question3'], "Query": table_info['Query3']}
                    ]
                    sample_queries_dict[table] = sample_queries_list
            
            for column in columns_used:
                column_info = sample_data_df[sample_data_df['Column_Name'] == column]
                if not column_info.empty:
                    column_info = column_info.iloc[0]
                    column_metadata = {
                        "from_table": column_info['Table_Name'],
                        "column_type": column_info['DataType'],
                        "column_description": column_info['ColumnDetails'],
                        "column_sample_values": column_info['sample_data'],
                        "filter_operation_on_column": column_info['Filter Operation']
                    }
                    # Add column_distinct_values_if_any only if it is not NaN
                    if pd.notna(column_info['categorical_fields_value']):
                        column_metadata["column_distinct_values_if_any"] = column_info['categorical_fields_value']
                    
                    # Add column_also_known_as only if it is not NaN
                    if pd.notna(column_info['Column Also Known As']):
                        column_metadata["column_also_known_as"] = column_info['Column Also Known As']
                    
                    columns_info_dict[column] = column_metadata
            
            each_metadata = {
                "user_query": user_question,
                "sql_query": user_query,
                "table_name": tables_used,
                "columns_used_in_sample_queries": columns_used,
                "tableinfo": table_info_dict,
                "sample_queries": sample_queries_dict,
                "info_on_columns_used": columns_info_dict
            }
            
            # Append the metadata dictionary directly without converting to string
            metadata_list.append(each_metadata)
        
        return metadata_list
    
    def _extract_chunk_from_each_metadata_dict(metadata_dict):
        chunk = {
            "user_query": metadata_dict["user_query"],
            "sql_query": metadata_dict["sql_query"],
            "info_on_columns_used": str([col_info["column_sample_values"] for col_info in metadata_dict["info_on_columns_used"].values()]),
            "tableinfo": metadata_dict["tableinfo"],
            "sample_queries": metadata_dict["sample_queries"]
        }
        
        return chunk
    
    def add_chunks_to_metadata(metadata_list):
        for metadata_dict in metadata_list:
            chunk = _extract_chunk_from_each_metadata_dict(metadata_dict)
            metadata_dict["chunk"] = str(chunk)
        
        return metadata_list    
    
    metadata_list = build_metadata(sample_data_df, tables_info_df, queries_df)
    metadata_with_chunks = add_chunks_to_metadata(metadata_list)
    
    return metadata_with_chunks

def read_material_drawing_data_from_blob(blob_uri):
    # Environment variables for authentication
    tenant_id = os.environ["TENET_ID"]
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = "knowledge-mining"

    # Authenticate and get BlobServiceClient
    blob_service_client = get_blob_service_client(tenant_id, client_id, client_secret, storage_account_name)
    
    df, _ = load_excel_and_get_parent_path(blob_service_client=blob_service_client, container_name=container_name, blob_uri=blob_uri, sheet_name=0)

    # List to hold multiple metadata dictionaries
    material_drawing_metadata_list = []

    # Iterate through the DataFrame row by row
    for i in range(len(df)):
        # Create a dictionary for the current row
        material_drawing_metadata = {
            'parent_filename': str(df['parent_filename'].iloc[i]),
            'parent_url': str(df['parent_url'].iloc[i]),
            'Material': str(df['Material'].iloc[i]),
            'FileName': str(df['FileName'].iloc[i]),
            'Revision': str(df['Revision'].iloc[i]),
            'FileText': str(df['FileText'].iloc[i]),
            'SingleSourcePart': str(df['SingleSourcePart'].iloc[i]),
            'DocType': str(df['DocType'].iloc[i]),
            'DocClassification': str(df['DocClassification'].iloc[i]),
            'MetalName': str(df['MetalName'].iloc[i]),
            'CommercialPartClass': str(df['CommercialPartClass'].iloc[i]),
            'FileLastModified': str(df['FileLastModified'].iloc[i]),
            'BuildToPrint': str(df['BuildToPrint'].iloc[i]),
            'RevisionExtn': str(df['RevisionExtn'].iloc[i]),
            'ObsoletePartsIdentified': str(df['ObsoletePartsIdentified'].iloc[i]),
            'BatchID': str(df['BatchID'].iloc[i]),
            'DWH_Insert_Date': str(df['DWH_Insert_Date'].iloc[i]),
            'DWH_Update_Date': str(df['DWH_Update_Date'].iloc[i]),
            'ID': str(df['ID'].iloc[i]),
            'SingleSourcePart_Flag': str(df['SingleSourcePart_Flag'].iloc[i]),
            'chunk': str(df['chunk'].iloc[i]),
            'chunk_id': str(df['chunk_id'].iloc[i])
        }
        material_drawing_metadata_list.append(material_drawing_metadata)

    return material_drawing_metadata_list