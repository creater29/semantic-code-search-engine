from search import SemanticSearchEngine

eval_dataset = [
    {"query": "function to connect to database", "expected_id": 2},
    {"query": "read csv file into dataframe", "expected_id": 1},
    {"query": "fetch website content html", "expected_id": 4},
    {"query": "upload data to s3 bucket", "expected_id": 7},
    {"query": "train random forest classifier", "expected_id": 3},
]

def calculate_mrr(engine, dataset):
    total_rr = 0.0
    top_k = 5

    print(f"Evaluating {len(dataset)} queries using MRR@{top_k}...")

    for item in dataset:
        query = item["query"]
        expected_id = item["expected_id"]

        results = engine.search(query, top_k=top_k)

        rank = 0
        for i, res in enumerate(results, 1):
            if res["id"] == expected_id:
                rank = i
                break

        if rank > 0:
            rr = 1.0 / rank
            print(f"Query: '{query:30s}' -> Found at Rank {rank} (RR: {rr:.2f})")
        else:
            rr = 0.0
            print(f"Query: '{query:30s}' -> NOT FOUND in top {top_k} (RR: 0.00)")

        total_rr += rr

    mrr = total_rr / len(dataset)
    return mrr

if __name__ == "__main__":
    engine = SemanticSearchEngine()
    print("-" * 50)
    mrr_score = calculate_mrr(engine, eval_dataset)
    print("-" * 50)
    print(f"Final MRR Score: {mrr_score:.4f}")
    if mrr_score == 1.0:
        print("Perfect retrieval on the evaluation set!")