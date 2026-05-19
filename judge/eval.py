import os
import json
import numpy as np

def evaluate(output_path: str = None, topks: list = [20]):
    if not os.path.exists(output_path):
        print(f"[ERROR] Result file not found at: {output_path}")
        return

    print(f"\nLoading results for evaluation from: {output_path}")
    
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("\n" + "="*40)
    print(f"{'Top-K':<10} | {'Avg Score':<12} | {'Avg NDCG':<12}")
    print("-" * 40)

    for k in topks:
        all_ndcgs = []
        all_scores = []

        for entry in data:
            full_scores = entry.get('llm_scores', [])
            scores = full_scores[:k] 
            
            rel = np.asarray([max(0, s) for s in scores])
            all_scores.extend(rel)

            discounts = np.log2(np.arange(len(rel)) + 2)
            dcg = np.sum(rel / discounts)

            ideal_rel = np.sort(rel)[::-1]
            idcg = np.sum(ideal_rel / discounts)

            ndcg = dcg / idcg if idcg > 0 else 0.0
            all_ndcgs.append(ndcg)

        avg_score = np.mean(all_scores)
        avg_ndcg = np.mean(all_ndcgs)
        
        print(f"Top-{k:<5} | {avg_score:<12.4f} | {avg_ndcg:<12.4f}")

    print("="*40 + "\n")


# def evaluate(test_dict, rec_list_data, output_path: str = None, topks: list = [20]):
#     if not os.path.exists(output_path):
#         print(f"[ERROR] Result file not found at: {output_path}")
#         return

#     print(f"\nLoading results for evaluation from: {output_path}")
    
#     with open(output_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     print("\n" + "="*40)
#     print(f"{'Top-K':<10} | {'Avg Score':<12} | {'Avg NDCG':<12} | {'Hit':<12}")
#     print("-" * 40)

#     for k in topks:
#         all_ndcgs = []
#         all_scores = []

#         user_recalls = []
        
#         for entry, item in zip(data, rec_list_data):
#             user_id = entry.get('user_id')
#             uid = item.get('user_id')
#             if user_id != uid:
#                 print(f"[ERROR] User ID mismatch: {user_id} vs {uid}")
#                 continue
        
#             full_scores = entry.get('llm_scores', [])
#             scores = full_scores[:k] 
            
#             rel = np.asarray([max(0, s) for s in scores])
#             all_scores.extend(rel)

#             discounts = np.log2(np.arange(len(rel)) + 2)
#             dcg = np.sum(rel / discounts)

#             ideal_rel = np.sort(rel)[::-1]
#             idcg = np.sum(ideal_rel / discounts)

#             ndcg = dcg / idcg if idcg > 0 else 0.0
#             all_ndcgs.append(ndcg)
            
#             if user_id not in test_dict:
#                 continue
#             rec_items = [i['item_id'] for i in item.get('items', [])[:k]]
#             true_items = set(test_dict[uid])
#             true_score = [1 if item_id in true_items else 0 for item_id in rec_items]
            
#             hits = sum([1 if sc == true_sc else 0 for sc, true_sc in zip(scores, true_score)])
#             recall_u = hits / k 
#             user_recalls.append(recall_u)

#         avg_score = np.mean(all_scores)
#         avg_ndcg = np.mean(all_ndcgs)
#         avg_recall = np.mean(user_recalls)

#         print(f"Top-{k:<5} | {avg_score:<12.4f} | {avg_ndcg:<12.4f} | {avg_recall:<12.4f}")

#     print("="*40 + "\n")