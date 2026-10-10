import re
from rag import Retriever

_retriever = Retriever()

PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)


def rag_search(query: str) -> str:
    results = _retriever.retrieve(query, top_k=3)

    if not results:
        return "NO_RELEVANT_INFO"

    joined = "\n---\n".join(chunk for chunk, _ in results)

    # 只有检索结果里真的出现电话号码格式时，才交给模型
    if not PHONE_PATTERN.search(joined):
        return "NO_RELEVANT_INFO"

    return joined