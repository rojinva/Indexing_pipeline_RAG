import requests
import json
import pandas as pd
import re
from factory.utils import Utils
from factory.synpAdapter import SynapseConnection
import configparser


class Index:
    config = None
    utils = None
    synpConn = None

    def __init__(self):
        self.utils = Utils()
        self.synpConn = SynapseConnection()
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')



    def getIndexes(self, baseUrl):
        indexes_list = []
        api_version = self.config.get('DEV','search_api_version')
        
        try:
            url = f"{baseUrl}/indexers?api-version={api_version}"
            payload = {}
            headers = {
                'api-key': self.config.get('DEV','api_key'),
                'Content-Type': self.config.get('DEV','Content_Type')
            }
            response = requests.request("GET", url, headers=headers, data=payload)
            res_json = response.json()
            res = res_json['value']
            for i in res:
                indexes_list.append(i['name'])
            resultRes = self.getIndexers(indexes_list)
            return resultRes
        except Exception as e:
            message = str(e)
            returnJson = {'Status':'Failure', 'StatusCode':500, 'Message':message}
            return json.dumps(returnJson)

    
    def getDataSource(self, indexerName):
        baseUrl = self.config.get('DEV','dev_baseUrl')
        apiVersion = self.config.get('DEV','search_api_version')
        try:
            url = f"{baseUrl}/indexers/{indexerName}?api-version={apiVersion}"
            payload = {}
            headers = {
                'api-key': self.config.get('DEV','api_key'),
                'Content-Type': self.config.get('DEV','Content_Type')
            }
            indexerRes = requests.request("GET", url, headers=headers, data=payload)
            indexerRes_json = indexerRes.json()
            datasource_name = indexerRes_json['dataSourceName']
            return datasource_name
        except Exception as e:
            message = str(e)
            returnJson = {'Status':'Failure', 'StatusCode':500, 'Message':message}
            return json.dumps(returnJson)
        
    def getDatasourceDefinition(self, dsName):
        baseUrl = self.config.get('DEV','dev_baseUrl')
        apiVersion = self.config.get('DEV','search_api_version')
        try:
            url = f"{baseUrl}/datasources/{dsName}?api-version={apiVersion}"
            payload = {}
            headers = {
                'api-key': self.config.get('DEV','api_key'),
                'Content-Type': self.config.get('DEV','Content_Type')
            }
            dsRes = requests.request("GET", url, headers=headers, data=payload)
            dsRes_json = dsRes.json()
            container_name = dsRes_json['container']['name']
            blob_name = dsRes_json['container']['query']
            if len(blob_name) > 995:
                blob_name = blob_name[:994]+'...'
            return container_name, blob_name
        except Exception as e:
            message = str(e)
            returnJson = {'Status':'Failure', 'StatusCode':500, 'Message':message}
            return json.dumps(returnJson)

    
            
    def getIndexers(self, indexList):
        returnJson = {'Status':'Success', 'StatusCode':200, 'Message':'Ran the Indexer successfully.'}
        data_list = []
        baseUrl = self.config.get('DEV','dev_baseUrl')
        api_version = self.config.get('DEV','search_api_version')
        try:
            for i in indexList:
                indexer_name = i
                url = f"{baseUrl}/indexers/{indexer_name}/status?api-version={api_version}"
                payload = {}
                headers = {
                    'api-key': self.config.get('DEV','api_key'),
                'Content-Type': self.config.get('DEV','Content_Type')
                }
                response = requests.request("GET", url, headers=headers, data=payload)
                response_json = response.json()
                IndexName = response_json['name']
                datasourceName = self.getDataSource(indexer_name)
                dsContainer_name, dsBlobName =self.getDatasourceDefinition(datasourceName)
                if "lastResult" in response_json:
                    last_result = response_json["lastResult"]
                    defaultRes = self.utils.defaultDict(last_result)
                    if len(last_result['errors']) > 0:
                        errorRes= self.utils.executionErrorDict(response_json, defaultRes, IndexName, datasourceName, dsBlobName, dsContainer_name)
                        for eachError in errorRes:
                            data_list.append(eachError)            
                    elif len(last_result['warnings']) > 0:
                        warnRes = self.utils.executionWarnDict(response_json, defaultRes, IndexName, datasourceName, dsBlobName, dsContainer_name)
                        for eachWarn in warnRes:
                            data_list.append(eachWarn)           
                    else:
                        successRes = self.utils.executionSuccessDict(defaultRes, IndexName, datasourceName, dsBlobName)
                        data_list.append(successRes)
            df= pd.DataFrame(data_list)
            df = df.drop_duplicates(subset=df.columns.difference(['CapturedOn']))
            #df.to_csv("./sample.csv")
            self.synpConn.insertRow(df)
            return json.dumps(returnJson)
        except Exception as e:
            message = str(e)
            returnJson = {'Status':'Failure', 'StatusCode':500, 'Message':message}
            return json.dumps(returnJson)
        
    def indexerChecks(self):
        customUrl = self.config.get('DEV','dev_baseUrl')
        return self.getIndexes(customUrl)