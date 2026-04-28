import json
import faiss
from sentence_transformers import SentenceTransformer

class SemanticSearchEngine:
    def __init__(self, model_name='flax-sentence-embeddings/st-codesearch-distilroberta-base',
                 index_path='code_index.faiss', corpus_path='corpus.json'):
        print(f"Loading model '{model_name}'...")
        self.model = SentenceTransformer(model_name)

        print(f"Loading FAISS index from '{index_path}'...")
        self.index = faiss.read_index(index_path)

        print(f"Loading corpus from '{corpus_path}'...")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for j, i in enumerate(indices[0]):
            if i != -1:
                snippet = self.corpus[i]
                results.append({
                    "score": float(scores[0][j]),
                    "id": snippet["id"],
                    "description": snippet.get("description", ""),
                    "code": snippet["code"]
                })
        return results
