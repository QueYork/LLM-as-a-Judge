import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Lossless')

    parse.add_argument('--seed', type=int, default=42, help='LLM random seed')
    parse.add_argument('--rec_seed', type=int, default=2021, help='rec model random seed')
    parse.add_argument('--num_workers', type=int, default=50, help='Number of workers for llm inference')
    parse.add_argument('--topks', nargs='+', type=int, default=[20, 50, 100], help='top@k test list') 
    parse.add_argument('--model', type=str, default='gcn', help='rec models to be loaded from [gcn, mf, lrgccf, ngcf, neumf, simgcl, gtn, caged]')  
    parse.add_argument('--caption', type=str, default='item_caption.csv', help='item caption file name')
    parse.add_argument('--dataset', type=str, default='kuairec', help='dataset name [kuairec, coat]')
    
    # LLM parameters
    parse.add_argument('--api_key', type=str, default='', help='LLM API Key')
    parse.add_argument('--base_url', type=str, default='', help='LLM API base url')
    parse.add_argument('--llm_name', type=str, default='/data/Qwen3-30B-A3B-ins', help='llm model name')
    parse.add_argument('--temp', type=float, default=0.0, help='LLM temperature setting')
    parse.add_argument('--max_tokens', type=int, default=2048, help='LLM max tokens setting')
    
    # Model builder parameters
    parse.add_argument('--layers', type=int, default=2, help='the layer number')
    parse.add_argument('--dim', type=int, default=256, help='embedding dimension')   
    parse.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parse.add_argument('--eps', type=float, default=0, help='simgcl noise scale') 
    parse.add_argument('--lam', type=float, default=0, help='simgcl cl loss weight') 
    
    
    return parse.parse_known_args()
