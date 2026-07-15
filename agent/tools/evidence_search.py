from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping fixed-size chunks (simple sliding window, no library needed)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _build_vectorstore() -> FAISS:
    chunks: list[str] = []
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        raw_text = file_path.read_text(encoding="utf-8")
        chunks.extend(_chunk_text(raw_text))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", check_embedding_ctx_length=False)
    return FAISS.from_texts(chunks, embeddings)


_vectorstore = _build_vectorstore()


@tool
def evidence_search(query: str) -> str:
    """Search the seed evidence corpus for text chunks relevant to the query.

    Use this to find supporting or opposing evidence for a claim before
    making an argument.
    """
    results = _vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in results)


if __name__ == "__main__":
    print(evidence_search.invoke({"query": "Does coffee cause anxiety?"}))
