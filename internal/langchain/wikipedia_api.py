from dataclasses import dataclass, field
from typing import List, Union
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WikipediaLoader


@dataclass
class WikipediaDocumentLoader:
    """A DocumentLoader that uses WikipediaAPIWrapper to load and split content from a Wikipedia page."""
    page_title: str
    wiki_parser: WikipediaAPIWrapper = field(init=False, default_factory=WikipediaAPIWrapper)
    text_splitter: TokenTextSplitter = field(init=False, default=TokenTextSplitter(chunk_size=512, chunk_overlap=128))

    def load_page(self) -> list[Document]:
        return WikipediaLoader(query=self.page_title).load()

    def split_content(self, raw_data: list[Document]):
        return self.text_splitter.split_documents(raw_data[:3])

    def load(self) -> List[Document]:
        """
        Load content from the Wikipedia page and wrap it in a Document format.
        Returns a list with a single Document containing the full page content.
        """
        content = self.wiki_parser.run(self.page_title)
        return [Document(page_content=content, metadata={"source": self.page_title})]

    def split_document(self, raw_documents: Union[List[Document], Document]) -> List[Document]:
        if isinstance(raw_documents, Document):
            raw_documents = [raw_documents]

        # Check the number of chunks before returning
        chunks = []
        for doc in raw_documents:
            print(f"Document content:\n{doc.page_content[:500]}...")  # Print first 500 characters for debug
            doc_chunks = self.text_splitter.split_documents([doc])
            print(f"Number of chunks: {len(doc_chunks)}")  # Print the number of chunks created
            chunks.extend(doc_chunks)

        return chunks


# Usage
if __name__ == "__main__":
    page_title = "Marcus Aurelius"
    loader = WikipediaDocumentLoader(page_title)

    print(f"Splitted: {loader.split_content(loader.load_page())}")

    # Load raw documents from Wikipedia
    raw_documents = loader.load()

    # Split documents into chunks
    documents = loader.split_document(raw_documents)

    # Print out the chunked document content
    for idx, doc in enumerate(documents):
        print(f"Chunk {idx + 1}:\n{doc.page_content}\n")
