import json
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("Loading model 'flax-sentence-embeddings/st-codesearch-distilroberta-base'...")
    model = SentenceTransformer('flax-sentence-embeddings/st-codesearch-distilroberta-base')

    print("Loading corpus.json...")
    with open('corpus.json', 'r') as f:
        corpus = json.load(f)

    texts_to_embed = []
    for item in corpus:
        desc = item.get("description", "").strip()
        code = item.get("code", "").strip()

        if desc:
            combined_text = f"{desc}\n\n{code}"
        else:
            combined_text = code

        texts_to_embed.append(combined_text)

    print(f"Generating embeddings for {len(texts_to_embed)} snippets...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    print(f"Embeddings generated with shape: {embeddings.shape}")

    np.save('embeddings.npy', embeddings)
    print("Successfully saved embeddings to 'embeddings.npy'")

if __name__ == "__main__":
    main()