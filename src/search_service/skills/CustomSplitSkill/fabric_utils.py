import os
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import ClientSecretCredential

def get_file_content_from_lakehouse(account_name: str,
                                    workspace_name: str,
                                    lakehouse_name: str,
                                    remote_path: str,
                                    filename: str,
                                    tenant_id: str,
                                    client_id: str,
                                    client_secret: str) -> bytes:
   """
   Download a file from Azure Fabric Lakehouse (OneLake) using service principal auth.
   Parameters:
   - account_name: OneLake account prefix (e.g., "onelake")
   - workspace_name: Fabric workspace file system name
   - lakehouse_name: Name of your lakehouse
   - remote_path: Path under lakehouse (e.g., "MyLakehouse.Lakehouse/Files/folder")
   - filename: Name of the file to download
   - tenant_id: Azure AD tenant (directory) ID
   - client_id: Service principal (app) client ID
   - client_secret: Service principal client secret
   """
   # 1. Build the account URL and authenticate using the service principal
   account_url = f"https://{account_name}.dfs.fabric.microsoft.com"
   credential  = ClientSecretCredential(
       tenant_id=tenant_id,
       client_id=client_id,
       client_secret=client_secret
   )  # Auth via SPN  [oai_citation_attribution:0‡Microsoft Learn](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.clientsecretcredential?view=azure-python)
   # 2. Create the DataLake service client
   service_client = DataLakeServiceClient(account_url=account_url, credential=credential)  # OneLake endpoint  [oai_citation_attribution:1‡Microsoft Learn](https://learn.microsoft.com/en-us/python/api/azure-storage-file-datalake/azure.storage.filedatalake.datalakeserviceclient?view=azure-python)
   # 3. Get the file system (workspace)
   fs_client = service_client.get_file_system_client(workspace_name)
   # 4. Build the directory client and get the file client
   directory  = f"{lakehouse_name}.Lakehouse/Files/{remote_path}"
   dir_client = fs_client.get_directory_client(directory)
   file_client = dir_client.get_file_client(filename)
   # 5. Download file data
   downloader = file_client.download_file()
   data       = downloader.readall()  # read all data into memory
   return data  # Return the downloaded data for further processing or testing