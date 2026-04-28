from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from search import SemanticSearchEngine

engine = SemanticSearchEngine()
app = FastAPI(title="Semantic Code Search Engine API")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    score: float
    id: int
    description: str
    code: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.get("/health")
def health_check():
    return {"status": "healthy", "index_size": engine.index.ntotal}

@app.post("/search", response_model=SearchResponse)
def search_code(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = engine.search(request.query, request.top_k)
    return SearchResponse(query=request.query, results=results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)