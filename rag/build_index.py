from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Neural_network",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Supervised_learning",
    "https://en.wikipedia.org/wiki/Unsupervised_learning",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Generative_adversarial_network",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "https://en.wikipedia.org/wiki/GPT-3",
    "https://en.wikipedia.org/wiki/GPT-4",
    "https://en.wikipedia.org/wiki/ChatGPT",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Word_embedding",
    "https://en.wikipedia.org/wiki/Attention_mechanism_(machine_learning)",
    "https://en.wikipedia.org/wiki/Backpropagation",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Recurrent_neural_network",
    "https://en.wikipedia.org/wiki/Random_forest",
    "https://en.wikipedia.org/wiki/Support_vector_machine",
    "https://en.wikipedia.org/wiki/Decision_tree",
    "https://en.wikipedia.org/wiki/K-means_clustering",
    "https://en.wikipedia.org/wiki/Principal_component_analysis",
    "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
    "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence",
    "https://en.wikipedia.org/wiki/Prompt_engineering",
    "https://en.wikipedia.org/wiki/Gemini_(language_model)",
    "https://en.wikipedia.org/wiki/Claude_(language_model)",
    "https://en.wikipedia.org/wiki/Llama_(language_model)",
    "https://en.wikipedia.org/wiki/Mistral_AI",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/DeepMind",
    "https://en.wikipedia.org/wiki/Anthropic",
    "https://en.wikipedia.org/wiki/Stable_Diffusion",
    "https://en.wikipedia.org/wiki/DALL-E",
    "https://en.wikipedia.org/wiki/Midjourney",
    "https://en.wikipedia.org/wiki/Data_science",
    "https://en.wikipedia.org/wiki/Big_data",
    "https://en.wikipedia.org/wiki/Cloud_computing",
    "https://en.wikipedia.org/wiki/Docker_(software)",
    "https://en.wikipedia.org/wiki/Kubernetes",
    "https://en.wikipedia.org/wiki/Application_programming_interface",
    "https://en.wikipedia.org/wiki/Quantum_computing",
    "https://en.wikipedia.org/wiki/Blockchain",
    "https://en.wikipedia.org/wiki/Cybersecurity",
    "https://en.wikipedia.org/wiki/Algorithm",
    "https://en.wikipedia.org/wiki/Data_mining",
    "https://en.wikipedia.org/wiki/DevOps",
    "https://en.wikipedia.org/wiki/Microservices",
    "https://en.wikipedia.org/wiki/Computer_network",
    "https://en.wikipedia.org/wiki/Internet",
    "https://en.wikipedia.org/wiki/Operating_system",
    "https://en.wikipedia.org/wiki/Software_engineering",
    "https://en.wikipedia.org/wiki/Open-source_software",
    "https://en.wikipedia.org/wiki/Database",
    "https://en.wikipedia.org/wiki/Computer_programming",
    "https://en.wikipedia.org/wiki/Compiler",
    "https://en.wikipedia.org/wiki/Linux",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/JavaScript",
    "https://en.wikipedia.org/wiki/C_(programming_language)",
    "https://en.wikipedia.org/wiki/Java_(programming_language)",
    "https://en.wikipedia.org/wiki/Rust_(programming_language)",
    "https://en.wikipedia.org/wiki/SQL",
    "https://en.wikipedia.org/wiki/NoSQL",
    "https://en.wikipedia.org/wiki/Git",
    "https://en.wikipedia.org/wiki/Statistics",
    "https://en.wikipedia.org/wiki/Linear_algebra",
    "https://en.wikipedia.org/wiki/Calculus",
    "https://en.wikipedia.org/wiki/Probability_theory",
    "https://en.wikipedia.org/wiki/Mathematics",
    "https://en.wikipedia.org/wiki/Physics",
    "https://en.wikipedia.org/wiki/Chemistry",
    "https://en.wikipedia.org/wiki/Biology",
    "https://en.wikipedia.org/wiki/Genetics",
    "https://en.wikipedia.org/wiki/Neuroscience",
    "https://en.wikipedia.org/wiki/Quantum_mechanics",
    "https://en.wikipedia.org/wiki/Theory_of_relativity",
    "https://en.wikipedia.org/wiki/Thermodynamics",
    "https://en.wikipedia.org/wiki/Electromagnetism",
    "https://en.wikipedia.org/wiki/Astronomy",
    "https://en.wikipedia.org/wiki/Space_exploration",
    "https://en.wikipedia.org/wiki/Evolution",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Environmental_science",
    "https://en.wikipedia.org/wiki/Ecology",
    "https://en.wikipedia.org/wiki/Biochemistry",
    "https://en.wikipedia.org/wiki/Molecular_biology",
    "https://en.wikipedia.org/wiki/Cell_biology",
    "https://en.wikipedia.org/wiki/Particle_physics",
    "https://en.wikipedia.org/wiki/Astrophysics",
    "https://en.wikipedia.org/wiki/Cosmology",
    "https://en.wikipedia.org/wiki/Black_hole",
    "https://en.wikipedia.org/wiki/Dark_matter",
    "https://en.wikipedia.org/wiki/Medicine",
    "https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare",
    "https://en.wikipedia.org/wiki/Pharmacology",
    "https://en.wikipedia.org/wiki/Epidemiology",
    "https://en.wikipedia.org/wiki/Vaccine",
    "https://en.wikipedia.org/wiki/Cancer",
    "https://en.wikipedia.org/wiki/Mental_health",
    "https://en.wikipedia.org/wiki/Nutrition",
    "https://en.wikipedia.org/wiki/Human_brain",
    "https://en.wikipedia.org/wiki/Immune_system",
    "https://en.wikipedia.org/wiki/DNA",
    "https://en.wikipedia.org/wiki/Virus",
    "https://en.wikipedia.org/wiki/Antibiotic",
    "https://en.wikipedia.org/wiki/Surgery",
    "https://en.wikipedia.org/wiki/Psychology",
    "https://en.wikipedia.org/wiki/Cognitive_science",
    "https://en.wikipedia.org/wiki/Economics",
    "https://en.wikipedia.org/wiki/Cryptocurrency",
    "https://en.wikipedia.org/wiki/Stock_market",
    "https://en.wikipedia.org/wiki/Inflation",
    "https://en.wikipedia.org/wiki/Globalization",
    "https://en.wikipedia.org/wiki/Entrepreneurship",
    "https://en.wikipedia.org/wiki/Supply_and_demand",
    "https://en.wikipedia.org/wiki/Capitalism",
    "https://en.wikipedia.org/wiki/Poverty",
    "https://en.wikipedia.org/wiki/Democracy",
    "https://en.wikipedia.org/wiki/Socialism",
    "https://en.wikipedia.org/wiki/Keynesian_economics",
    "https://en.wikipedia.org/wiki/Gross_domestic_product",
    "https://en.wikipedia.org/wiki/International_trade",
    "https://en.wikipedia.org/wiki/Microeconomics",
    "https://en.wikipedia.org/wiki/Macroeconomics",
    "https://en.wikipedia.org/wiki/Philosophy",
    "https://en.wikipedia.org/wiki/History_of_science",
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/World_War_I",
    "https://en.wikipedia.org/wiki/Ancient_Rome",
    "https://en.wikipedia.org/wiki/Ancient_Egypt",
    "https://en.wikipedia.org/wiki/Renaissance",
    "https://en.wikipedia.org/wiki/Industrial_Revolution",
    "https://en.wikipedia.org/wiki/Cold_War",
    "https://en.wikipedia.org/wiki/Ethics",
    "https://en.wikipedia.org/wiki/Logic",
    "https://en.wikipedia.org/wiki/Consciousness",
    "https://en.wikipedia.org/wiki/Philosophy_of_mind",
    "https://en.wikipedia.org/wiki/Epistemology",
    "https://en.wikipedia.org/wiki/Metaphysics",
    "https://en.wikipedia.org/wiki/Ancient_Greece",
    "https://en.wikipedia.org/wiki/French_Revolution",
    "https://en.wikipedia.org/wiki/American_Revolution",
    "https://en.wikipedia.org/wiki/Roman_Empire",
    "https://en.wikipedia.org/wiki/Byzantine_Empire",
    "https://en.wikipedia.org/wiki/Mongol_Empire",
    "https://en.wikipedia.org/wiki/Ottoman_Empire",
    "https://en.wikipedia.org/wiki/British_Empire",
    "https://en.wikipedia.org/wiki/Renewable_energy",
    "https://en.wikipedia.org/wiki/Solar_energy",
    "https://en.wikipedia.org/wiki/Nuclear_power",
    "https://en.wikipedia.org/wiki/Electric_vehicle",
    "https://en.wikipedia.org/wiki/Robotics",
    "https://en.wikipedia.org/wiki/Automation",
    "https://en.wikipedia.org/wiki/Internet_of_things",
    "https://en.wikipedia.org/wiki/5G",
    "https://en.wikipedia.org/wiki/Semiconductor",
    "https://en.wikipedia.org/wiki/Nanotechnology",
    "https://en.wikipedia.org/wiki/Biotechnology",
    "https://en.wikipedia.org/wiki/CRISPR",
    "https://en.wikipedia.org/wiki/Aerospace_engineering",
    "https://en.wikipedia.org/wiki/SpaceX",
    "https://en.wikipedia.org/wiki/NASA",
    "https://en.wikipedia.org/wiki/Mars",
    "https://en.wikipedia.org/wiki/International_Space_Station",
    "https://en.wikipedia.org/wiki/Satellite",
    "https://en.wikipedia.org/wiki/Photography",
    "https://en.wikipedia.org/wiki/Film",
    "https://en.wikipedia.org/wiki/Music",
    "https://en.wikipedia.org/wiki/Literature",
    "https://en.wikipedia.org/wiki/Architecture",
    "https://en.wikipedia.org/wiki/Painting",
    "https://en.wikipedia.org/wiki/Art",
    "https://en.wikipedia.org/wiki/Language",
    "https://en.wikipedia.org/wiki/Linguistics",
    "https://en.wikipedia.org/wiki/Sociology",
    "https://en.wikipedia.org/wiki/Anthropology",
    "https://en.wikipedia.org/wiki/Political_science",
    "https://en.wikipedia.org/wiki/Human_rights",
    "https://en.wikipedia.org/wiki/United_Nations",
    "https://en.wikipedia.org/wiki/Geopolitics",
    "https://en.wikipedia.org/wiki/Immigration",
    "https://en.wikipedia.org/wiki/Education",
    "https://en.wikipedia.org/wiki/Religion",
    "https://en.wikipedia.org/wiki/Islam",
    "https://en.wikipedia.org/wiki/Christianity",
    "https://en.wikipedia.org/wiki/Hinduism",
    "https://en.wikipedia.org/wiki/Buddhism",
    "https://en.wikipedia.org/wiki/Judaism",
    "https://en.wikipedia.org/wiki/Mythology",
    "https://en.wikipedia.org/wiki/Sport",
    "https://en.wikipedia.org/wiki/Football",
    "https://en.wikipedia.org/wiki/Cricket",
    "https://en.wikipedia.org/wiki/Olympic_Games",
    "https://en.wikipedia.org/wiki/Basketball",
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    res = requests.get(url, timeout=10, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()
    return soup.get_text(separator=" ")

def chunk(text, source_url, size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), size - overlap):
        part = text[i:i+size]
        if len(part.strip()) > 50:
            chunks.append({"text": part, "source": source_url})
    return chunks

def main():
    os.makedirs(".", exist_ok=True)

    all_chunks = []

    for url in URLS:
        print("Scraping:", url)
        try:
            text = scrape(url)
            all_chunks.extend(chunk(text, url))
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, "index.faiss")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Index built successfully with", len(all_chunks), "chunks")

if __name__ == "__main__":
    main()
