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

def build_judge_prompt(user_history: list, target_item: dict) -> tuple:
    # --- System Prompt ---
    system_content = (
        "你是一个极其挑剔的推荐系统评估专家。你的任务是基于用户的历史行为，找出真正能打动用户的物品。\n"
        "请注意：你的默认立场是【不推荐】。绝大多数推荐结果都是平庸的，只有在证据极其充分时才认可推荐。\n\n"
        "### 评分标准 (0 或 1):\n"
        " - 0 : 【拒绝】(默认选项)\n"
        "     - 只要存在任何不确定或证据不足，直接打 0 分。\n"
        "     - 如果物品仅有粗粒度匹配（例如都属于生活题材），但没有更细粒度的匹配证据，打 0 分。\n"
        "     - 如果物品是毫无针对性的热门万金油内容，打 0 分。\n"
        " - 1 : 【命中】(仅限极高匹配)\n"
        "     - 只有当物品与用户历史有强烈的、具体的逻辑关联（包括但不限于：同一话题、同一主人公、小众的同好圈子）时，才允许打 1 分。\n\n"
        "### 输出要求:\n"
        "1. 先从为什么不适合用户开始思考，仅当没有理由否定推荐时，才可判定为命中。\n"
        "2. 最终输出必须是一个 JSON 对象，包含 'reason' 和 'score' 字段，且 'reason' 字段仅输出一句简短的解释。\n"
        "3. 输出 JSON 格式示例:\n"
        "{\n"
        "  \"reason\": \"该物品虽然与用户历史同属科幻题材，但属于大众化热门作品，缺乏具体且强烈的匹配证据\",\n"
        "  \"score\": 0\n"
        "}"
    )

    # --- User Prompt ---
    history_str = "\n".join([f"- {item['caption']}" for item in user_history])
    
    user_content = (
        "### 用户历史行为:\n"
        f"{history_str}\n\n"
        "### 待评估物品:\n"
        f"- {target_item['caption']}\n\n"
        "请基于上述信息进行评估，直接输出 JSON。"
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
    
    sys_prompt, user_prompt = build_judge_prompt(task_info['history_items'], item)
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
        
        user_id_to_rec_order[u_id] = [item['item_id'] for item in recommended_items]
        
        for item in recommended_items:
            all_item_tasks.append({
                'user_id': u_id,
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
    
    output_path = join(board.DATA_PATH, board.args.dataset, board.args.model + f'_pointwise_judge_results_full_{board.args.seed}.json')
    output_dir = dirname(output_path)
    if not exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    
    print(f"Done. Saved results for {len(final_output)} users to {output_path}")
    evaluate(output_path, board.args.topks)


if __name__ == "__main__":
    main()