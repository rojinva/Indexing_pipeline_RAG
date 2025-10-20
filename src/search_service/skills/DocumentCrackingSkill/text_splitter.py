import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text_into_chunks(content, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    content = re.sub(r"\n{2,}", "\n", content)
    content = re.sub(r"\s{2,}", " ", content)
    regex_cleaned_text = " " if not len(content.strip()) else content.strip()

    chunked_documents = text_splitter.create_documents([regex_cleaned_text])

    chunks = [chunk.page_content for chunk in chunked_documents]
    return chunks
