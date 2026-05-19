import os
import sys
import json
import warnings
from os.path import join, dirname, exists
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI
import numpy as np

warnings.filterwarnings('ignore')

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = join(PATH, '../')
sys.path.append(ROOT)
import board
from judge.eval import evaluate
from src.reclist_loader import LoadRecList

NUM_WORKERS = board.args.num_workers
client = OpenAI(api_key=board.API_KEY, base_url=board.BASE_URL)

def build_judge_prompt(user_profile: str, user_history: list, target_item: dict) -> tuple:
    system_content = (
        "You are a ruthless fashion recommendation critic.\n"
        "Your goal: Determine if a candidate item is a **perfect match** for the user based on user history and profile.\n\n"
        "### CORE PRINCIPLE:\n"
        "- **Skeptical Default:** Your starting position is **REJECT (0)**. Most recommendations are mediocre, you only approve if the evidence is undeniable.\n"
        "### DECISION FRAMEWORK (0 or 1):\n"
        "- **0 : REJECT (Default)**\n"
        "  - As long as there is any uncertain or lack of evidence, reject it.\n"
        "  - If you only observe a coarse-grained matching (e.g., broad category match like 'Tops') but not finer-grained matching, reject it.\n"
        "- **1 : HIT (Strict Match)**\n"
        "  - The item fits clearly within the user's established clusters of Style or Color.\n"
        "  - Only strong logical extensions (e.g., familiar style in a neutral color) supported by history exist.\n\n"
        "### OUTPUT REQUIREMENT:\n"
        "1. First think about why does this item NOT fit the user? Only if no strong reason to reject exists, can you score it 1.\n"
        "2. You MUST output a JSON object with 'reason' and 'score' fields.\n"
        "3. The 'reason' should be a single, concise sentence in English."
    )
    
    # --- User Prompt ---
    history_str = "\n".join([f"- {item['caption']}" for item in user_history])
    
    user_content = (
        f"### USER PROFILE:\n{user_profile}\n\n"
        f"### USER PURCHASE HISTORY:\n{history_str}\n\n"
        f"### CANDIDATE ITEM TO EVALUATE:\n- {target_item['caption']}\n\n"
        "Based on the profile and history, provide your evaluation in JSON format."
    )
    
    return system_content, user_content

def parse_pointwise_response(response: str) -> Tuple[float, str]:
    try:
        data = json.loads(response)
        score = float(data.get('score', 0.0))
        reason = data.get('reason', "")
        return score, reason
    except Exception as e:
        print(f"[ERROR] Parse failed: {e}")
        return 0.0, "Parse Error"

def call_model(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=board.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=board.args.temp,  
            max_tokens=board.args.max_tokens,  
            seed=board.SEED,
            response_format={"type": "json_object"} 
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[ERROR] API call failed: {exc}")
        return None

def process_one_item_task(task_info: Dict) -> Dict:
    user_id = task_info['user_id']
    item = task_info['target_item']
    user_desc = task_info['user_description']
    
    sys_prompt, user_prompt = build_judge_prompt(user_desc, task_info['history_items'], item)
    raw_response = call_model(sys_prompt, user_prompt)
    
    if raw_response:
        score, reason = parse_pointwise_response(raw_response)
        return {
            'user_id': user_id,
            'item_id': item['item_id'],
            'score': score,
            'reason': reason,
            'raw': raw_response
        }
    else:
        return {
            'user_id': user_id,
            'item_id': item['item_id'],
            'score': 0.0,
            'reason': "API Failed",
            'raw': None
        }

def main():
    print("Loading data...")
    data_loader = LoadRecList(dataset=board.args.dataset)
    rec_list_data = data_loader.get_rec_list(model=board.args.model, topk=max(board.args.topks))
    pos_list_data = data_loader.get_pos_list()

    all_item_tasks = []
    user_id_to_rec_order = {} 

    for rec, pos in zip(rec_list_data, pos_list_data):
        u_id = rec['user_id']
        history = pos['items']
        recommended_items = rec['items']
        user_desc = pos['description']
        
        user_id_to_rec_order[u_id] = [item['item_id'] for item in recommended_items]
        
        for item in recommended_items:
            all_item_tasks.append({
                'user_id': u_id,
                'user_description': user_desc,
                'history_items': history,
                'target_item': item
            })

    total_tasks = len(all_item_tasks)
    print(f"Start Pointwise LLM Inference: {total_tasks} total item-evaluations...")

    item_results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        future_to_task = {pool.submit(process_one_item_task, task): task for task in all_item_tasks}
        
        for future in tqdm(as_completed(future_to_task), total=total_tasks):
            res = future.result()
            item_results.append(res)

    user_grouped_data = defaultdict(dict)
    for res in item_results:
        u_id = res['user_id']
        i_id = res['item_id']
        user_grouped_data[u_id][i_id] = {
            'score': res['score'],
            'reason': res['reason']
        }

    final_output = []
    for u_id, ordered_item_ids in user_id_to_rec_order.items():
        scores_list = []
        reasons_list = []
        
        for i_id in ordered_item_ids:
            item_info = user_grouped_data[u_id].get(i_id, {'score': 0.0, 'reason': "Missing"})
            scores_list.append(item_info['score'])
            reasons_list.append(item_info['reason'])
            
        final_output.append({
            'user_id': u_id,
            'llm_scores': scores_list,
            'reasons': reasons_list 
        })

    output_path = join(board.DATA_PATH, board.args.dataset, board.args.model + f'_pointwise_full_{board.args.seed}.json')
    output_dir = dirname(output_path)
    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    
    print(f"Done. Saved results for {len(final_output)} users to {output_path}")
    evaluate(output_path, board.args.topks)


if __name__ == "__main__":
    main()