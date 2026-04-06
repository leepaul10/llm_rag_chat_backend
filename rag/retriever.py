# from fastembed import TextEmbedding
# import faiss
# import numpy as np
# import pickle

# model=TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
# index=faiss.read_index("rag/index.faiss")
# with open("rag/chunks.pkl", "rb") as f:
#     chunks=pickle.load(f)
# RELEVANCE_THRESHOLD = 0.65  #Raised from 0.4 - only truly relevant topics trigger RAG
# def retrieve(query, k=3):
#     """
#     Retrieves the most relevant context from the index.
#     Uses cosine similarity (higher = more similar).
#     """
#     q_vec=np.array(list(model.embed([query])))
#     q_vec=q_vec/np.linalg.norm(q_vec,axis=1,keepdims=True)
#     D,I=index.search(q_vec.astype(np.float32),k)
#     best_score=D[0][0]  #The highest cosine similarity score
#     #Use RAG only if best score is above relevance threshold
#     use_rag=best_score > RELEVANCE_THRESHOLD
#     if use_rag:
#         context="\n\n".join([chunks[i]["text"] for i in I[0]])
#     else:
#         context=""
#     return context, best_score, use_rag



from fastembed import TextEmbedding
import faiss
import numpy as np
import pickle

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = faiss.read_index("rag/index.faiss")

with open("rag/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

RELEVANCE_THRESHOLD = 0.65  # Only truly relevant topics trigger RAG
AMBIGUITY_GAP = 0.05        # If top scores are too close, query is ambiguous

def retrieve(query, k=3):
    """
    Retrieves the most relevant context from the index.
    Uses cosine similarity (higher = more similar).
    Also detects potential ambiguity when multiple top results are close in score.
    """
    # Step 1: Embed the query
    q_vec = np.array(list(model.embed([query])))
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

    # Step 2: Search index
    D, I = index.search(q_vec.astype(np.float32), k)

    best_score = D[0][0]
    second_score = D[0][1] if len(D[0]) > 1 else 0

    # Step 3: Check ambiguity
    score_gap = abs(best_score - second_score)
    is_ambiguous = score_gap < AMBIGUITY_GAP

    # Step 4: Determine if RAG should be used
    use_rag = best_score > RELEVANCE_THRESHOLD and not is_ambiguous

    # Step 5: Prepare context
    context = "\n\n".join([chunks[i]["text"] for i in I[0]]) if use_rag else ""

    return context, best_score, use_rag, is_ambiguous