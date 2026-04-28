import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["index_size"] == 7

def test_search_endpoint():
    # Test valid query
    query = "function that reads a CSV and returns a dataframe"
    response = client.post("/search", json={"query": query, "top_k": 3})
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == query
    assert len(data["results"]) <= 3
    assert data["results"][0]["id"] == 1 # The CSV load function

def test_search_empty_query():
    # Test empty query
    response = client.post("/search", json={"query": "   ", "top_k": 3})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]

def test_files_exist():
    assert os.path.exists("corpus.json")
    assert os.path.exists("embeddings.npy")
    assert os.path.exists("code_index.faiss")

if __name__ == "__main__":
    print("Running E2E tests...")
    test_files_exist()
    print("Files exist test: PASS")
    test_health_endpoint()
    print("Health endpoint test: PASS")
    test_search_endpoint()
    print("Search endpoint test: PASS")
    test_search_empty_query()
    print("Empty query test: PASS")
    print("All tests passed!")