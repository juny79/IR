import json
import os
from pathlib import Path
from sentence_transformers import CrossEncoder
import torch
from tqdm import tqdm

# Configuration
SOURCE_FILES = [
    "submission_v9_sota.csv",
    "submission_v12_sota.csv",
    "submission_v15_sota.csv",
    "submission_best_9394.csv",
    "submission_bge_m3_sota_v7.csv",
    "submission_v16_gemini_rerank_20251227_130830.csv"
]
DOCUMENTS_FILE = "data/documents.jsonl"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
OUTPUT_FILE = "submission_final_union_rerank_v18.csv"
TOP_K_UNION = 10  # Take top 10 from each file for the union

def load_submissions(files):
    all_data = {}
    for file in files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found. Skipping.")
            continue
        print(f"Loading {file}...")
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                eid = str(obj["eval_id"])
                if eid not in all_data:
                    all_data[eid] = {
                        "eval_id": obj["eval_id"],
                        "standalone_query": obj.get("standalone_query", ""),
                        "candidates": set()
                    }
                # Add top K candidates to the union
                topk = obj.get("topk", [])
                for docid in topk[:TOP_K_UNION]:
                    all_data[eid]["candidates"].add(docid)
                
                # Ensure we have a query
                if not all_data[eid]["standalone_query"] and obj.get("standalone_query"):
                    all_data[eid]["standalone_query"] = obj["standalone_query"]
    return all_data

def load_documents(file_path):
    print(f"Loading documents from {file_path}...")
    docs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs[obj["docid"]] = obj["content"]
    return docs

def main():
    # 1. Load submissions and build union
    eval_data = load_submissions(SOURCE_FILES)
    
    # 2. Load documents
    docs = load_documents(DOCUMENTS_FILE)
    
    # 3. Initialize Reranker
    print(f"Initializing reranker {RERANK_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=device)
    
    # 4. Rerank
    print("Reranking union...")
    results = []
    # Sort by eval_id for consistency
    sorted_eids = sorted(eval_data.keys(), key=lambda x: int(x))
    
    for eid in tqdm(sorted_eids):
        item = eval_data[eid]
        query = item["standalone_query"]
        candidates = list(item["candidates"])
        
        if not query or not candidates:
            print(f"Warning: Missing query or candidates for eval_id {eid}")
            results.append({
                "eval_id": item["eval_id"],
                "standalone_query": query,
                "topk": candidates[:5] # Fallback
            })
            continue
            
        # Prepare pairs for reranking
        pairs = [[query, docs.get(docid, "")] for docid in candidates]
        
        # Predict scores
        scores = reranker.predict(pairs)
        
        # Sort candidates by score
        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        reranked_topk = [c[0] for c in scored_candidates]
        
        results.append({
            "eval_id": item["eval_id"],
            "standalone_query": query,
            "topk": reranked_topk[:5] # We only need top 5 for submission
        })
    
    # 5. Save results
    output_path = OUTPUT_FILE
    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
