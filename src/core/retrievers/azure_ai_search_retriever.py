import aiohttp
import asyncio
from copy import deepcopy
from ._base_retriever import BaseRetriever


class AzureAISearchRetriever(BaseRetriever):
    """
    Asynchronous retriever for Azure AI search service with built-in rate limiting.
    """

    def __init__(
        self, search_api_base, index_name, search_api_version, api_key, rate_limit=5
    ):
        """
        Initializes the Azure AI search service retriever with rate limiting.
        """
        self.search_api_base = search_api_base
        self.index_name = index_name
        self.search_api_version = search_api_version
        self.url = f"{self.search_api_base}/indexes/{self.index_name}/docs/search?api-version={self.search_api_version}"
        self.headers = {"Content-Type": "application/json", "api-key": api_key}
        self.search_params = {"top": 5}
        self._semaphore = asyncio.Semaphore(
            rate_limit
        )  # Semaphore to control concurrency

    async def retrieve(self, query, fields, search_params=None):
        """
        Retrieves documents asynchronously based on the query and specified fields, with concurrency control.
        """
        search_params = search_params or {}  # Ensure search_params is a dictionary
        try:
            async with self._semaphore:  # Control the concurrency here
                async with aiohttp.ClientSession() as session:
                    payload = deepcopy(self.search_params)
                    payload.update(search_params)
                    if "search" in payload and payload["search"] is not None:
                       payload["search"] = query
                    elif "vectorQueries" in payload:
                       payload['vectorQueries'][0]['text'] = query
                    elif "search" and "vectorQueries" in payload:
                       payload["search"] = query
                       payload['vectorQueries'][0]['text'] = query
                    async with session.post(
                        url=self.url, headers=self.headers, json=payload
                    ) as response:
                        response_content = (
                            await response.json()
                        )  # Parse JSON response regardless of status
                        if response.status != 200:
                            raise Exception(
                                f"Error: {response.status}, {response_content.get('error', {}).get('message', 'No error message provided')}"
                            )
                        response.raise_for_status()  # Check for HTTP errors
                        results = response_content.get("value", [])
                        return [
                            [result.get(field) for field in fields]
                            for result in results
                        ]
        except aiohttp.ClientResponseError as e:
            print(f"HTTP ClientResponse Error: Status {e.status} - {str(e)}")
        except aiohttp.ClientConnectionError as e:
            print(f"Network Connection Error: {str(e)}")
        except Exception as e:
            print(f"Unexpected Error: {str(e)}")
