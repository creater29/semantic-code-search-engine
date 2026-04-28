import numpy as np
import faiss

def main():
    print("Loading embeddings from 'embeddings.npy'...")
    embeddings = np.load('embeddings.npy')
    print(f"Loaded embeddings of shape {embeddings.shape}")

    dimension = embeddings.shape[1]
    print("Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)

    print(f"Creating FAISS IndexFlatIP with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings to the index...")
    index.add(embeddings)
    print(f"Index created. Total vectors in index: {index.ntotal}")

    faiss.write_index(index, 'code_index.faiss')
    print("Successfully saved FAISS index to 'code_index.faiss'")

if __name__ == "__main__":
    main()