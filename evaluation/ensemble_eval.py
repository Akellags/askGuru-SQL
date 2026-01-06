import httpx
import asyncio
import re
import sqlglot
from typing import List, Dict
import time

API_URL = "http://localhost:8000/generate-sql"
API_KEY = "your-secret-api-key-here" # Update this to match your .env
GOLD_FILE = "data/data_warehouse/oracle_train/gold_standard_sqls.md"

def parse_gold_standard(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by ### which seems to be the separator
    entries = content.split('###')
    results = []
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if not lines:
            continue
            
        # First line is usually the question: "1. Question text"
        question_line = lines[0].strip()
        question = re.sub(r'^\d+\.\s*', '', question_line)
        
        # Remaining lines are the SQL
        gold_sql = "\n".join(lines[1:]).strip()
        
        if question and gold_sql:
            results.append({
                "question": question,
                "gold_sql": gold_sql
            })
    
    return results

def normalize_sql(sql: str) -> str:
    try:
        return sqlglot.transpile(sql, read="oracle", write="oracle", pretty=True)[0].lower()
    except:
        return sql.lower().strip()

async def evaluate():
    gold_data = parse_gold_standard(GOLD_FILE)
    print(f"Loaded {len(gold_data)} test cases.")
    
    headers = {"X-API-Key": API_KEY}
    success_count = 0
    valid_count = 0
    total = len(gold_data)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, item in enumerate(gold_data[:10]): # Test first 10
            print(f"[{i+1}/{total}] Testing: {item['question'][:50]}...")
            
            try:
                response = await client.post(
                    API_URL,
                    json={"question": item["question"]},
                    headers=headers
                )
                
                if response.status_code == 200:
                    res_data = response.json()
                    gen_sql = res_data["sql"]
                    is_valid = res_data["validation_passed"]
                    
                    if is_valid:
                        valid_count += 1
                    
                    # Exact or normalized match (heuristic)
                    if normalize_sql(gen_sql) == normalize_sql(item['gold_sql']):
                        success_count += 1
                        print("  ✓ Exact/Normalized Match")
                    elif is_valid:
                        print("  ⚠ Valid SQL, but differs from Gold")
                    else:
                        print("  ✗ Invalid SQL generated")
                else:
                    print(f"  Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"  Exception: {str(e)}")
            
            time.sleep(1) # Be nice to the GPU
            
    print(f"\n--- Evaluation Results ---")
    print(f"Total Tested: {min(10, total)}")
    print(f"Oracle Valid: {valid_count}")
    print(f"Logic Match:  {success_count}")

if __name__ == "__main__":
    asyncio.run(evaluate())
