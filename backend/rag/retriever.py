from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index("backend/rag/index.faiss")

with open("backend/rag/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

RELEVANCE_THRESHOLD = 0.4  # cosine similarity threshold (0 to 1)

def retrieve(query, k=3):
    """
    Retrieves the most relevant context from the index.
    Uses cosine similarity (higher = more similar).
    """
    q_vec = model.encode([query])
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

    D, I = index.search(q_vec.astype(np.float32), k)

    best_score = D[0][0]  # highest cosine similarity score

    # Use RAG only if best score is above relevance threshold
    use_rag = best_score > RELEVANCE_THRESHOLD

    if use_rag:
        context = "\n\n".join([chunks[i]["text"] for i in I[0]])
    else:
        context = ""

    return context, best_score, use_rag
