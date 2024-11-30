from dataclasses import dataclass, field
from typing import List

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document


@dataclass
class WikipediaDocumentLoader:
    """A DocumentLoader that uses WikipediaAPIWrapper to load content from a Wikipedia page."""
    page_title: str
    wiki_parser: WikipediaAPIWrapper = field(init=False, default_factory=WikipediaAPIWrapper)
    text_splitter: TokenTextSplitter = field(init=False, default=TokenTextSplitter(chunk_size=512, chunk_overlap=24))

    def load(self) -> List[Document]:
        """Load content from the Wikipedia page."""
        content = self.wiki_parser.run(self.page_title)
        # Wrapping the content in the Document format
        return [Document(page_content=content, metadata={"source": self.page_title})]

    def split_document(self, raw_document: List[Document] | Document) -> List[Document]:
        return self.text_splitter.split_documents(raw_document)


# Usage
if __name__ == "__main__":
    page_title = "Marcus Aurelius"
    loader = WikipediaDocumentLoader(page_title)
    raw_documents = loader.load()
    documents = loader.split_document(raw_documents[:3])


    # Print out the document content
    for doc in documents:
        print(doc.page_content)
