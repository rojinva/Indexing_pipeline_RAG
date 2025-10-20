from datetime import datetime
import urllib.parse
#from factory.synpAdapter import SynapseConnection
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import configparser
from datetime import datetime
import hashlib

class Utils:
    #synpConn = None
    config = None
    def __init__(self):
        #self.synpConn = SynapseConnection()
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        
    def defaultDict(self, jsonData):
        resultDict = {}
        resultDict['Status'] = jsonData['status']
        resultDict['DocsSucceeded'] = jsonData['itemsProcessed']
        resultDict['DocsFailure'] = jsonData['itemsFailed']
        resultDict['No_Of_Docs'] = int(resultDict['DocsSucceeded']) + int(resultDict['DocsFailure'])
        resultDict['StartTime'] = jsonData['startTime']
        resultDict['EndTime'] = jsonData['endTime']
        return resultDict
    
    def executionErrorDict(self, jsonData, defaultData, indexName, dSourceName, dsBlobName, dsContainerName):
        errorList = []
        last_result = jsonData["lastResult"]
        if "errors" in last_result:
            errors = last_result["errors"]
            for error in errors:
                dataDict ={}
                fileurl = None
                docUUID = None
                filename = error['key']
                if filename != None:
                    decodedurl, newFileName = self.getFileName(str(filename))
                    docUUID = self.get_document_UUID(dsContainerName, decodedurl)
                else:
                    newFileName = None
                error_message = error['errorMessage']
                error_message = error_message[:100]
                dataDict['File_UUID'] = docUUID
                dataDict['Filename'] = newFileName
                dataDict['Error'] = error_message
                dataDict['Warning'] = None
                dataDict['Status'] = defaultData['Status']
                dataDict['Job_start_time'] = defaultData['StartTime']
                dataDict['Job_end_time'] = defaultData['EndTime']
                dataDict['File_Retriggered'] = False
                dataDict['Attempt_cnt'] = 0
                dataDict['File_Reprocessed_Flg'] = False
                dataDict['Data_source_name'] = dSourceName
                dataDict['Folder'] = dsBlobName
                dataDict['Indexer'] = indexName
                dataDict['CapturedOn'] = datetime.now()
                get_hash = self.compute_hash(dataDict)
                dataDict['hash'] = get_hash
                errorList.append(dataDict)
        return errorList
    
    def executionWarnDict(self, jsonData, defaultData, indexName, dSourceName, dsBlobName, dsContainerName):
        warnList = []
        last_result = jsonData["lastResult"]
        if "warnings" in last_result:
            warnings = last_result["warnings"]
            for warning in warnings:
                dataDict = {}
                fileurl = None
                docUUID = None
                filename = warning['key']
                if filename != None:
                    decodedurl, newFileName = self.getFileName(str(filename))
                    docUUID = self.get_document_UUID( dsContainerName, decodedurl)
                else:
                    newFileName = None
                warning_message = warning['message']
                warning_message = warning_message[:100]
                dataDict['File_UUID'] = docUUID
                dataDict['Filename'] = newFileName
                dataDict['Error'] = None
                dataDict['Warning'] = warning_message
                dataDict['Status'] = defaultData['Status']
                dataDict['Job_start_time'] = defaultData['StartTime']
                dataDict['Job_end_time'] = defaultData['EndTime']
                dataDict['File_Retriggered'] = False
                dataDict['Attempt_cnt'] = 0
                dataDict['File_Reprocessed_Flg'] = False
                dataDict['Data_source_name'] = dSourceName
                dataDict['Folder'] = dsBlobName
                dataDict['Indexer'] = indexName
                dataDict['CapturedOn'] = datetime.now()
                get_hash = self.compute_hash(dataDict)
                dataDict['hash'] = get_hash
                warnList.append(dataDict)
        return warnList
    
    def executionSuccessDict(self, defaultData, indexName, dSourceName, dsContainerName):
        dataDict ={}
        dataDict['File_UUID'] = None
        dataDict['Filename'] = None
        dataDict['Error'] = None
        dataDict['Warning'] = None
        dataDict['Status'] = defaultData['Status']
        dataDict['Job_start_time'] = defaultData['StartTime']
        dataDict['Job_end_time'] = defaultData['EndTime']
        dataDict['File_Retriggered'] = False
        dataDict['Attempt_cnt'] = 0
        dataDict['File_Reprocessed_Flg'] = False
        dataDict['Data_source_name'] = dSourceName
        dataDict['Folder'] = dsContainerName
        dataDict['Indexer'] = indexName
        dataDict['CapturedOn'] = datetime.now()
        get_hash = self.compute_hash(dataDict)
        dataDict['hash'] = get_hash
        return dataDict
    
    def getFileName(self, complexURL):
        getURL = complexURL.split('&')[1]
        getURL = getURL.split('=')[1]
        decodeURL = urllib.parse.unquote_plus(urllib.parse.unquote_plus(getURL))
        fileName = decodeURL.split('/')[-1]
        return decodeURL, fileName
    
    
    def blobName(self, someURL, containerName):
        getblob = None
        try:
            containerName = str(containerName) + "/"
            getblob = someURL.split(containerName)[1]
        except Exception as e:
            #print("blob name exception is : ",e)
            getblob = None
        return getblob
    

    def get_document_UUID(self, container_name, blob_name):
        get_UUID = None
        # Azure AD credentials
        tenant_id = self.config.get('AzureADcredentials', 'tenant_id')
        client_id = self.config.get('AzureADcredentials', 'client_id')
        client_secret = self.config.get('AzureADcredentials', 'client_secret')
        # Azure Storage account details
        account_url = self.config.get('StorageAccount', 'accountURL')
        # Authenticate with Azure AD
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        blob_service_client = BlobServiceClient(account_url, credential=credential)
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)
        getBlobName =  self.blobName(blob_name, container_name)
        if getBlobName != None:
            try:
                # Get the blob client
                blob_client = container_client.get_blob_client(getBlobName)
                # Get the blob properties
                blob_properties = blob_client.get_blob_properties()
                # Extract metadata
                metadata = blob_properties.metadata
                if len(metadata)>0:
                    get_UUID = metadata['uuid']
                else:
                    get_UUID = None
            except Exception as e:
                #print("getBlobName  is : ",getBlobName)
                #print("get_blob_client Exception is : ",e)
                get_UUID = None
        return get_UUID
    

    # # Function to compute hash value
    def compute_hash(self, row_dict):
        empty_hash = ""
        Filename =row_dict['Filename']
        Error = row_dict['Error']
        Warning = row_dict['Warning']
        Status = row_dict['Status']
        Job_start_time = row_dict['Job_start_time']
        Job_end_time = row_dict['Job_end_time']
        File_Retriggered = row_dict['File_Retriggered']
        Attempt_cnt = row_dict['Attempt_cnt']
        File_Reprocessed_Flg = row_dict['File_Reprocessed_Flg']
        Data_source_name = row_dict['Data_source_name']
        Folder = row_dict['Folder']
        Indexer = row_dict['Indexer']
        file_UUID = row_dict['File_UUID']
        hash_input = f"""{Filename}{Error}{Warning}{Status}{Job_start_time}{Job_end_time}{File_Retriggered}{Attempt_cnt}{File_Reprocessed_Flg}{Data_source_name}{Folder}{Indexer}{file_UUID}"""
        try:
            return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        except Exception as e:
            print("compute_hash exception is : ",e)
            return empty_hash
