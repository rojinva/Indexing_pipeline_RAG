from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from collections import Counter
import os

'''
This tool helps us answer the following questions:
- Are the indexes the same at a use case level? (maybe other levels in the future... doc level?)
- What is the count of index entries that are different?
- What is the difference in content between the two indexes
'''

class AzureSearchClient:
    """
    Class for interacting with Azure Search Service
    """
    def __init__(self, azure_search_service, acs_index_name, azure_search_service_key, use_case_folder_name):
        self.azure_search_service = azure_search_service
        self.acs_index_name = acs_index_name
        self.azure_search_service_key = azure_search_service_key
        self.use_case_folder_name = use_case_folder_name
        self.client = self._create_client()

    def _create_client(self):
        client = SearchClient(
            endpoint=f"https://{self.azure_search_service}.search.windows.net",
            index_name=self.acs_index_name,
            credential=AzureKeyCredential(self.azure_search_service_key))
        return client
    
    def fetch_all_chunk_hashes(self):
        filter_condition = f"search.in(use_case, '{self.use_case_folder_name}')"
        results = None
        # Query ACS for retrieving all chunk hashes
        results = self.client.search("*", query_type="full", filter=filter_condition, select=["chunk_hash"])
        chunk_hash_list = [result["chunk_hash"] for result in list(results)]
        return chunk_hash_list
    
    def fetch_chunk_hash_info(self, chunk_hash):
        # Query index where chunk_hash is equal to the provided chunk_hash
        filter_condition = f"search.in(use_case, '{self.use_case_folder_name}') and \
            search.in(chunk_hash, '{chunk_hash}')"
        results = None
        results = self.client.search("*", query_type="full", filter=filter_condition)
        
        return list(results)
    


class ACSCompare:
    """
    Class for comparing two ACS indexes
    """
    def __init__(self, hash_list1, hash_list2):
        self.hash_list1 = hash_list1
        self.hash_list2 = hash_list2
        self.chunk_diff = self._compare_hashes()

    def _compare_hashes(self):
        # Count the number of occurances of each chunk hash
        hash_counter1 = Counter(self.hash_list1)
        hash_counter2 = Counter(self.hash_list2)
        
        # Convert counter to set for comparison (hash + # of occurances e.g. f13j90f_12)
        hash_set1 = set([str(key) + "_" + str(value) for key, value in hash_counter1.items()])
        hash_set2 = set([str(key) + "_" + str(value) for key, value in hash_counter2.items()])
        
        # Find the difference between the two sets of chunk hashes
        chunk_differences = list(set(hash_set1 ^ hash_set2))
        
        if len(chunk_differences) == 0:
            print(f"There are no chunks differences between the two indexes")
        else:
            print(f"There are {len(chunk_differences)} chunk differences between the two indexes")
            print(f"This can mean that a chunk is missing from an index or the quantity of that chunk varies")
        
        return [hash.split('_')[0] for hash in chunk_differences]
    
    def retrieve_chunk_differences(self):
        return self.chunk_diff
    
    @staticmethod
    def retrieve_inconsistent_content(chunk_hash_info1, chunk_hash_info2):
        # Create a mapping of chunk hash + parent filename + chunk id to the entry
        chunk_content_mapping1 = {entry["chunk_hash"] + "_" + entry["parent_filename"] + "_" + entry["chunk_id"].split("_")[-1]: entry for entry in chunk_hash_info1}
        chunk_content_mapping2 = {entry["chunk_hash"] + "_" + entry["parent_filename"] + "_" + entry["chunk_id"].split("_")[-1]: entry for entry in chunk_hash_info2}
        
        # Find the entries that are missing from one index
        entries_in_index1_only = list(set(chunk_content_mapping1) - set(chunk_content_mapping2))
        entries_in_index2_only = list(set(chunk_content_mapping2) - set(chunk_content_mapping1))

        return {
            "missing_from_index2": [chunk_content_mapping1[id] for id in entries_in_index1_only], "missing_from_index1": [chunk_content_mapping2[id] for id in entries_in_index2_only]
            }


if __name__ == "__main__":

    # Run to compare two ACS indexes

    # Load in environment variables
    from dotenv import load_dotenv
    load_dotenv()

    index1_service = os.getenv('INDEX1_SEARCH_SERVICE_NAME')
    index1_name = os.getenv('INDEX1_ACS_INDEX_NAME')
    index1_key = os.getenv('INDEX1_SEARCH_SERVICE_KEY')

    index2_service = os.getenv('INDEX2_SEARCH_SERVICE_NAME')
    index2_name = os.getenv('INDEX2_ACS_INDEX_NAME')
    index2_key = os.getenv('INDEX2_SEARCH_SERVICE_KEY')

    my_use_case = "customerSurvey"
    
    # Instantiate clients with search
    index1_client = AzureSearchClient(index1_service, index1_name, index1_key, my_use_case)
    index2_client = AzureSearchClient(index2_service, index2_name, index2_key, my_use_case)

    # Retrieve a list of all chunk hashes for the chosen use case
    index1_hashes = index1_client.fetch_all_chunk_hashes()
    index2_hashes = index2_client.fetch_all_chunk_hashes()
    
    # Compare number of entries between indexes
    print(f"Number of chunks in index #1: {len(index1_hashes)}")
    print(f"Number of chunks in index #2: {len(index2_hashes)}")

    # Compare the chunk hashes for two indexes
    index_comparison = ACSCompare(index1_hashes, index2_hashes)

    # Retrieve a list of chunk hashes that are different between the two indexes (show first 10)
    list_of_different_chunks = index_comparison.retrieve_chunk_differences()
    print(list_of_different_chunks[:10])

    # View all entries that contain a specific chunk hash (here we are looking at the first chunk hash in the list of different chunks)
    chunk_of_interest = list_of_different_chunks[0]
    chunk_hash_info1 = index1_client.fetch_chunk_hash_info(chunk_of_interest)
    chunk_hash_info2 = index2_client.fetch_chunk_hash_info(chunk_of_interest)
    
    # View the specific entries (docuemnts) that are "missing" from an index
    print(ACSCompare.retrieve_inconsistent_content(chunk_hash_info1, chunk_hash_info2))
