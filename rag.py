from policy_loader import load_policy_documents
import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at",
    "for", "with", "about", "by", "from", "as", "into", "through",
    "should", "could", "would", "can", "may", "might", "must", "shall",
    "do", "does", "did", "have", "has", "had",
    "exact", "exactly", "please", "call", "get",
}

KEYWORD_BOOST = {
    "phone": 5, "telephone": 5, "number": 4, "contact": 3,
    "service": 2, "desk": 2, "it": 1, "helpdesk": 4, "support": 3,
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]


def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


class Retriever:
    def __init__(self):
        documents = load_policy_documents()
        self.chunks = []
        self.metadata = []
        for doc in documents:
            content = doc.page_content
            pieces = chunk_text(content) if len(content) > 400 else [content]
            for piece in pieces:
                self.chunks.append(piece)
                self.metadata.append(doc.metadata)

    def retrieve(self, query, top_k=3, min_matches=2):
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        scored = []
        for chunk, meta in zip(self.chunks, self.metadata):
            chunk_lower = chunk.lower()
            chunk_tokens = set(tokenize(chunk))

            matched = query_tokens & chunk_tokens
            if len(matched) < min_matches:
                continue

            score = sum(1 for t in matched if t in chunk_tokens)
            for kw, weight in KEYWORD_BOOST.items():
                if kw in query_tokens and kw in chunk_tokens:
                    score += weight

            if score > 0:
                scored.append((score, chunk, meta))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [(c, m) for _, c, m in scored[:top_k]]