import os
import json
from tqdm import tqdm
from pathlib import Path
from models.openai_client import openai_client

# Settings
DOC_PATH = "/root/IR/data/documents.jsonl"
EVAL_PATH = "/root/IR/data/eval.jsonl"
V1_PATH = "/root/IR/submission_surgical_v1.csv"
BREAK_V2_PATH = "/root/IR/submission_final_0.95_break_v2.csv"
SOTA_PATH = "/root/IR/submission_v9_sota.csv"
MASTER_PATH = "/root/IR/submission_final_0.95_master.csv"
CAND_D_PATH = "/root/IR/submission_candidate_D_id271_id303.csv"
CAND_B_PATH = "/root/IR/submission_candidate_B_id271.csv"
OUTPUT_PATH = "/root/IR/submission_super_hybrid_final_v2.csv"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return data

def load_full(path):
    d = {}
    for obj in load_jsonl(path):
        d[str(obj['eval_id'])] = obj
    return d

print("Loading data...")
eval_data = load_full(EVAL_PATH)
documents = {d['docid']: d['content'] for d in load_jsonl(DOC_PATH)}
v1 = load_full(V1_PATH)
break_v2 = load_full(BREAK_V2_PATH)
sota = load_full(SOTA_PATH)
master = load_full(MASTER_PATH)
cand_d = load_full(CAND_D_PATH)
cand_b = load_full(CAND_B_PATH)

# IDs to re-evaluate
target_ids = [26, 31, 37, 43, 84, 85, 106, 214, 215, 243, 246, 250, 270, 271, 303]

def llm_choose_best(messages, candidates):
    """
    candidates: list of (doc_id, doc_content)
    """
    system_prompt = """당신은 검색 결과의 정확도를 판별하는 전문가입니다.
사용자의 질문(대화 맥락 포함)과 여러 개의 검색 결과(Candidate)가 주어집니다.
질문에 대해 가장 정확하고, 직접적이며, 완결된 답변을 제공하는 문서를 하나만 골라주세요.
반드시 JSON 형식으로 {"best_candidate_index": 0} (0부터 시작) 로 답변하세요."""

    user_prompt = f"대화 맥락:\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n\n"
    for i, (doc_id, doc_content) in enumerate(candidates):
        user_prompt += f"Candidate {i} (ID: {doc_id}):\n{doc_content[:2000]}\n\n"
    
    try:
        resp = openai_client._call_with_retry(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(resp)
        return int(parsed.get("best_candidate_index", 0))
    except:
        return 0

print("Starting Super Hybrid Generation...")
final_results = []

all_eval_ids = sorted(eval_data.keys(), key=int)

for eid_str in tqdm(all_eval_ids):
    eid = int(eid_str)
    
    # Default to v1
    base_obj = v1.get(eid_str)
    
    if eid in target_ids:
        print(f"\nRe-evaluating ID {eid}...")
        
        # Collect unique candidates
        candidate_ids = []
        seen_docs = set()
        
        for source in [v1, break_v2, sota, master, cand_d, cand_b]:
            obj = source.get(eid_str)
            if obj and obj.get('topk'):
                tid = obj['topk'][0]
                if tid not in seen_docs:
                    candidate_ids.append(tid)
                    seen_docs.add(tid)
        
        if len(candidate_ids) > 1:
            candidates = [(tid, documents[tid]) for tid in candidate_ids]
            winner_idx = llm_choose_best(eval_data[eid_str]['msg'], candidates)
            winner_tid = candidate_ids[winner_idx]
            
            print(f"  Winner: Candidate {winner_idx} ({winner_tid[:8]})")
            
            # Find which source has this winner as top-1 to use its full topk
            chosen_obj = base_obj
            for source in [v1, break_v2, sota, master, cand_d, cand_b]:
                obj = source.get(eid_str)
                if obj and obj.get('topk') and obj['topk'][0] == winner_tid:
                    chosen_obj = obj
                    break
        else:
            chosen_obj = base_obj
    else:
        chosen_obj = base_obj

    # Construct final object
    final_obj = {"eval_id": eid}
    
    # Preserve standalone_query if it exists
    sq = chosen_obj.get('standalone_query') or (base_obj.get('standalone_query') if base_obj else None)
    if sq:
        final_obj["standalone_query"] = sq
        
    final_obj["topk"] = chosen_obj.get('topk', [])
    final_obj["answer"] = chosen_obj.get('answer', base_obj.get('answer', "") if base_obj else "")
    
    final_results.append(final_obj)

print(f"Saving to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for res in final_results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"\n✅ Super Hybrid file generated: {OUTPUT_PATH}")
