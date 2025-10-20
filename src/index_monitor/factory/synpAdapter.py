import pyodbc
import configparser
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from sqlalchemy import create_engine, text
import urllib




class SynapseConnection:
    config = None
    driver = None
    server = None
    database = None
    username = None
    secretname = None
    password = None
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

    def getSynpPassword(self,secret_name):
        # Azure AD credentials
        tenant_id = self.config.get('AzureADcredentials', 'tenant_id')
        client_id = self.config.get('AzureADcredentials', 'client_id')
        client_secret = self.config.get('AzureADcredentials', 'client_secret')
        # Key Vault details
        key_vault_url = self.config.get('Synapse', 'synpKeyvaultPswdURL')
        # Authenticate with Azure AD
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        # Create a SecretClient
        secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
        secret = secret_client.get_secret(secret_name)
        return secret.value
    

        

    def getConnection(self):
        self.driver = self.config.get('Synapse','synpDriver')
        self.server = self.config.get('Synapse','synpServer')
        self.database = self.config.get('Synapse','synpDatabase')
        self.username = self.config.get('Synapse','synpUsername')
        self.secretname = self.config.get('Synapse','synpSecretname')
        self.password = self.getSynpPassword(self.secretname)
        connection_string = "DRIVER={};SERVER={};DATABASE={};Authentication=ActiveDirectoryPassword;UID={};PWD={}".format(
            str(self.driver),self.server, self.database, self.username, self.password)
        
        try:
            return pyodbc.connect(connection_string)
        except pyodbc.Error as e:
            print(f"Error: {e}")
    
    
    def select(self, query):
        response = {"isSuccess":True, "errorCode":200, "result":[], "message":"success"}
        try:
            sessionConn = self.getConnection()
            sessionCursor = sessionConn.cursor()
            sessionCursor.execute(query)
            result = sessionCursor.fetchall()
            response['result'] = result
            sessionCursor.close()
            sessionConn.close()
            return response
        except Exception as e:
            print("Query exception is : ", e)
            response = {"isSuccess":True, "errorCode":500, "result":[], "message":e}
            return response
    

    def insertRow(self,df):
        insert_count = 0
        try:
            table_name = self.config.get('Synapse', 'synpTableName')
            sessionConn = self.getConnection()
            sessionCursor = sessionConn.cursor()
            for index, row in df.iterrows():
                hash_value = row['hash']
                sessionCursor.execute(f""" SELECT 1 FROM {table_name} WHERE hash = ? """,  hash_value)
                if not sessionCursor.fetchone():
                    sessionCursor.execute(f"""
                        INSERT INTO {table_name} (Filename, Error, Warning, Status, Job_start_time, Job_end_time, File_Retriggered, Attempt_cnt, File_Reprocessed_Flg, Data_source_name, Folder,Indexer,UUID,Captured,hash)
                        VALUES (?, ?, ?,?, ?, ?,?, ?, ?,?, ?, ?,?,?,?)
                        """, row['Filename'], row['Error'], row['Warning'],row['Status'], row['Job_start_time'], row['Job_end_time'],row['File_Retriggered'], row['Attempt_cnt'], row['File_Reprocessed_Flg'],row['Data_source_name'], row['Folder'], row['Indexer'], row['File_UUID'], row['CapturedOn'], hash_value)

                    insert_count = insert_count + 1
                
            # # Commit the transaction
            commitRes = sessionConn.commit()
            # Close the cursor and connection
            sessionCursor.close()
            sessionConn.close()
            print("Total inserted records count is : ", insert_count)
            return commitRes
        except Exception as e:
            print("Insert Row exception is : ",e)
            print("error row is : ",row)