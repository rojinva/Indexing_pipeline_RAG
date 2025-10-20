import re
from typing import List, Optional
from ..models.records import BaseSearchIndexRecord
from .ms_word_pages_processor import MSWordProcessor

class TranscriptsMSWordProcessor(MSWordProcessor):
    def _transform_combined_text(self, combined_text: str) -> str:
        """
        Applies custom transformations to the combined text.
        Removes timestamps in the format 'HH:MM' at the beginning of lines.
        :param combined_text: The original combined text.
        :return: Transformed text.
        """
        return re.sub(r'^\d{1,2}:\d{2}\s*', '', combined_text, flags=re.MULTILINE)

    def process(
        self,
        storage_metadata,
        datasource_client,
        text_splitter=None,
        output_path: Optional[str] = None,
    ) -> List[BaseSearchIndexRecord]:
        """
        Processes .docx files synchronously, with custom transformations applied to the text.
        :return: List of BaseSearchIndexRecord objects.
        """
        # Call the parent class's process method
        records = super().process(storage_metadata, datasource_client, text_splitter, output_path)

        # Apply custom transformation to each record's content
        for record in records:
            record.content = self._transform_combined_text(record.content)

        return records

    async def aprocess(
        self,
        storage_metadata,
        datasource_client,
        text_splitter=None,
        output_path: Optional[str] = None,
    ) -> List[BaseSearchIndexRecord]:
        """
        Processes .docx files asynchronously, with custom transformations applied to the text.
        :return: List of BaseSearchIndexRecord objects.
        """
        # Call the parent class's aprocess method
        records = await super().aprocess(storage_metadata, datasource_client, text_splitter, output_path)

        # Apply custom transformation to each record's content
        for record in records:
            record.content = self._transform_combined_text(record.content)

        return records