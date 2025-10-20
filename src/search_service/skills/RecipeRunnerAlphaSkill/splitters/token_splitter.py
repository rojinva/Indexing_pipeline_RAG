import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TokensSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, content: str) -> List[str]:
        """
        Split text into chunks using a RecursiveCharacterTextSplitter from LangChain.
        This implementation cleans extra whitespace/newlines before splitting.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4", 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        # Clean-up content.
        content = re.sub(r"\n{2,}", "\n", content)
        content = re.sub(r"\s{2,}", " ", content)
        regex_cleaned_text = content.strip() if content.strip() else " "
        # Create the chunked documents and return page_content for each chunk.
        chunked_documents = text_splitter.create_documents([regex_cleaned_text])
        chunks = [chunk.page_content for chunk in chunked_documents]
        return chunks