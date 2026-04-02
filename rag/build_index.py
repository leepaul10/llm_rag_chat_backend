from fastembed import TextEmbedding
import faiss
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
#Using bge-small-en-v1.5 
model=TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
URLS=[
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Tennis",
    "https://en.wikipedia.org/wiki/Chess",
    "https://en.wikipedia.org/wiki/Video_game",
    "https://en.wikipedia.org/wiki/Social_media",
    "https://en.wikipedia.org/wiki/Twitter",
    "https://en.wikipedia.org/wiki/Facebook",
    "https://en.wikipedia.org/wiki/Google",
    "https://en.wikipedia.org/wiki/Amazon_(company)",
    "https://en.wikipedia.org/wiki/Apple_Inc.",
    "https://en.wikipedia.org/wiki/Microsoft",
    "https://en.wikipedia.org/wiki/Tesla,_Inc.",
    "https://en.wikipedia.org/wiki/Netflix",
    "https://en.wikipedia.org/wiki/Uber",
    "https://en.wikipedia.org/wiki/Airbnb",
]
def scrape(url):
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    res=requests.get(url,timeout=15,headers=headers)
    soup=BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()
    return soup.get_text(separator=" ")
def chunk(text, source_url, size=1000, overlap=200):
    chunks=[]
    for i in range(0, len(text),size-overlap):
        part=text[i:i+size]
        if len(part.strip()) > 50:
            chunks.append({"text": part, "source": source_url})
    return chunks
def main():
    os.makedirs(".",exist_ok=True)
    all_chunks=[]
    print("\n" +"="*60)
    print("Starting RAG Index Build")
    print("="*60 + "\n")
    #Progress bar for WEB scraping
    for url in tqdm(URLS, desc="Scraping URLs", unit="page"):
        try:
            text=scrape(url)
            chunks_from_url=chunk(text, url)
            all_chunks.extend(chunks_from_url)
        except requests.exceptions.Timeout:
            tqdm.write(f"Timeout on {url}, skipping...")
        except requests.exceptions.RequestException as e:
            tqdm.write(f"Network error on {url}: {e}")
        except Exception as e:
            tqdm.write(f"Failed to scrape {url}: {e}")

    print(f"\n✓ Scraped {len(all_chunks)} chunks from {len(URLS)} URLs\n")
    #Embedding with progres Bar
    texts=[c["text"] for c in all_chunks]
    print("Generating embeddings")
    embeddings_list=[]
    for i, embedding in enumerate(model.embed(texts)):
        embeddings_list.append(embedding)
        if (i + 1) % 500==0:
            print(f"  ✓ {i + 1}/{len(texts)} chunks embedded")
    embeddings=np.array(embeddings_list)
    embeddings=embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"✓ Generated {len(embeddings)} embeddings\n")
    #Building FAISS index
    print("Building FAISS index...")
    index=faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    print("✓ FAISS index built\n")
    #Saving  the files
    print("Saving files...")
    faiss.write_index(index, "index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print("✓ Files saved\n")

    print("="*60)
    print(f"Index built successfully!")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Files: index.faiss, chunks.pkl")
    print("="*60 + "\n")
if __name__ == "__main__":
    main()


