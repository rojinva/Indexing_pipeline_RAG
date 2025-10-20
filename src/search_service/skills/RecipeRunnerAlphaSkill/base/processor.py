from abc import ABC
from typing import Any
from ..models.request import StorageMetadata


class BaseProcessor(ABC):
    """
    Base class for all processors.
    To create a custom processor, subclass this and implement the process method.
    """

    def __init__(self, processor_name: str = "BaseProcessor"):
        self.processor_name = processor_name

    def process(self, req: StorageMetadata, text_splitter: Any, datasource_client: Any) -> str:
        raise NotImplementedError("Subclasses must implement the process() method.")