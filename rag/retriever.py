from fastembed import TextEmbedding
import faiss
import numpy as np
import pickle

model=TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
index=faiss.read_index("rag/index.faiss")
with open("rag/chunks.pkl", "rb") as f:
    chunks=pickle.load(f)
RELEVANCE_THRESHOLD = 0.65  #Raised from 0.4 - only truly relevant topics trigger RAG
def retrieve(query, k=3):
    """
    Retrieves the most relevant context from the index.
    Uses cosine similarity (higher = more similar).
    """
    q_vec=np.array(list(model.embed([query])))
    q_vec=q_vec/np.linalg.norm(q_vec,axis=1,keepdims=True)
    D,I=index.search(q_vec.astype(np.float32),k)
    best_score=D[0][0]  #The highest cosine similarity score
    #Use RAG only if best score is above relevance threshold
    use_rag=best_score > RELEVANCE_THRESHOLD
    if use_rag:
        context="\n\n".join([chunks[i]["text"] for i in I[0]])
    else:
        context=""
    return context, best_score, use_rag

